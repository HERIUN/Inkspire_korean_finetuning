"""InkSpire 레이아웃 모델 — Masked Modeling + Conditional Flow Matching (논문 §A, 표 1 최선안).

글자열 C 와 레퍼런스 토큰(첫 줄)의 bbox 가 주어지면 나머지 글자의 bbox
b = [w, h, Δx, Δy](page_w 정규화, 모듈 1 `layout_seq` 규약)를 예측한다.
토큰 = char_emb + pos_emb + line_emb + layout_proj(layout_in) + obs_emb + t_mlp(sinus(t)).
손실 = 마스크 토큰의 v-prediction L1, 추론 = 마스크 토큰만 Euler ODE (관측 토큰은 그대로 반환).

★ 정규화는 필수다. 원값이 0.02~0.06 스케일이라 노이즈 N(0,1) 과 섞으면 x_t ≈ z,
  v 타깃 (z − x0) ≈ z 가 되어 모델이 "노이즈 복사"만 배운다(SNR 붕괴). 그래서 차원별
  `(layout − mean) / std` 로 맞추고, 통계는 `mean`/`std` 버퍼로 state_dict 에 같이 저장한다.
  트레이너는 step 0 에 `set_norm` 을 반드시 호출해야 한다(기본 mean 0/std 1 이면 위 붕괴).

torch + stdlib 만 사용. 데이터 의존 없음 — vocab 크기는 생성자 인자.
self-check: `.venv/bin/python models/inkspire_layout.py --smoke`
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LINES = 64   # line_emb 크기. line_id 는 63 으로 clamp


def _sinus(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """t∈[0,1] → [B,dim] 사인/코사인 임베딩 (t·1000, DiT/diffusers 규약)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class LayoutCFM(nn.Module):
    def __init__(self, vocab_size: int, d: int = 512, n_layers: int = 10, n_heads: int = 8,
                 max_len: int = 2048):
        super().__init__()
        self.kw = dict(vocab_size=vocab_size, d=d, n_layers=n_layers, n_heads=n_heads, max_len=max_len)
        self.char_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        # ponytail: 논문에 없는 line_emb. 마스크 토큰의 Δx/Δy 는 t=1 에서 순수 노이즈라 줄바꿈을
        #   입력에서 읽을 수 없고, 어차피 layout_to_bboxes 가 line_id 를 요구한다. 줄 시작 토큰의
        #   Δx≈−0.9 vs 그 외 ≈0 이라는 이봉 타깃을 결정적으로 만든다. 32k 파라미터.
        self.line_emb = nn.Embedding(MAX_LINES, d)
        self.layout_proj = nn.Linear(4, d)
        self.obs_emb = nn.Embedding(2, d)   # 0 = 관측 GT, 1 = 마스크(노이즈) — Fig 10(c) reference/masked
        # ponytail: adaLN 대신 additive t. 4-d 회귀·10층 pre-LN 잔차엔 충분, adaLN-Zero 는 +8M/30줄.
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 4)
        # ★ 정규화 통계 — state_dict 에 포함되어 ckpt 와 함께 이동. set_norm 으로만 갱신.
        self.register_buffer("mean", torch.zeros(4))
        self.register_buffer("std", torch.ones(4))

    @torch.no_grad()
    def set_norm(self, layout: torch.Tensor, pad_mask: torch.Tensor):
        """비-pad 토큰 [M,4] 로 차원별 mean/std 계산. layout [B,N,4], pad_mask [B,N] True=pad."""
        v = layout[~pad_mask].float()
        self.mean.copy_(v.mean(0))
        self.std.copy_(v.std(0).clamp_min(1e-4))

    def normalize(self, layout):
        return (layout - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def forward(self, char_ids, line_id, layout_in, gen_mask, t, pad_mask):
        """char_ids/line_id [B,N] long, layout_in [B,N,4] (정규화됨), gen_mask/pad_mask [B,N] bool,
        t [B] ∈[0,1] → v [B,N,4]. pad 위치 출력은 무의미."""
        B, N = char_ids.shape
        assert N <= self.kw["max_len"], f"N={N} > max_len={self.kw['max_len']}"
        pos = torch.arange(N, device=char_ids.device)
        h = (self.char_emb(char_ids) + self.pos_emb(pos)[None]
             + self.line_emb(line_id.clamp(max=MAX_LINES - 1))
             + self.layout_proj(layout_in) + self.obs_emb(gen_mask.long())
             + self.t_mlp(_sinus(t))[:, None])
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(self.norm(h))


def make_gen_mask(ref_mask: torch.Tensor, pad_mask: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """샘플별 마스크 [B,N] bool (True = 예측 대상). 논문: 연속 구간 / 토큰별 20% 각 50%.
    ref/pad 는 절대 건드리지 않는다 — 단, 전체 마스크 모드(gen == ~pad)는 예외."""
    ref, pad = ref_mask.cpu(), pad_mask.cpu()
    gen = torch.zeros_like(pad)
    for b in range(ref.shape[0]):
        cand = (~ref[b] & ~pad[b]).nonzero().flatten()
        n = cand.numel()
        u = torch.rand((), generator=rng).item()
        # ponytail: 10% 는 레퍼런스까지 전부 마스크 — 레이아웃 라벨이 없는 실제 손글씨(ref_mask
        #   전부 False)에서도 추론이 되게 하는 논문 편차. 글자 크기 조건이 없으니 크기는 임의.
        if u < 0.1 or n == 0:
            gen[b] = ~pad[b]
            continue
        if u < 0.55:   # 연속 구간: 길이 L~U[1,n], 시작 s (후보 인덱스 공간 기준)
            L = int(torch.randint(1, n + 1, (), generator=rng))
            s = int(torch.randint(0, n - L + 1, (), generator=rng))
            gen[b, cand[s:s + L]] = True
        else:          # 토큰별 i.i.d. 20%, 0 개면 1 개 강제
            g = torch.rand(n, generator=rng) < 0.2
            if not g.any():
                g[int(torch.randint(0, n, (), generator=rng))] = True
            gen[b, cand[g]] = True
    return gen.to(ref_mask.device)


def cfm_loss(model: LayoutCFM, batch: dict, rng: torch.Generator) -> torch.Tensor:
    """batch = layout_collate dict (char_ids, layout, line_id, ref_mask, pad_mask). 마스크 토큰 v-pred L1."""
    dev = model.mean.device
    ids, lay, line = batch["char_ids"].to(dev), batch["layout"].to(dev), batch["line_id"].to(dev)
    ref, pad = batch["ref_mask"].to(dev), batch["pad_mask"].to(dev)
    gen = make_gen_mask(ref, pad, rng)
    x0 = model.normalize(lay)
    # ponytail: t~U(0,1). logit-normal 은 고차원 이미지용 — 4-d 회귀에는 무의미.
    t = torch.rand(ids.shape[0], generator=rng).to(dev)
    z = torch.randn(x0.shape, generator=rng).to(dev)
    tt = t[:, None, None]
    x_t = (1 - tt) * x0 + tt * z
    g3 = gen[..., None]
    v = model(ids, line, torch.where(g3, x_t, x0), gen, t, pad)
    m = g3.float()
    return ((v - (z - x0)).abs() * m).sum() / (m.sum() * 4).clamp_min(1.0)


@torch.no_grad()
def sample(model: LayoutCFM, char_ids, line_id, layout_ref, ref_mask, pad_mask, steps: int = 10,
           seed: int = 0) -> torch.Tensor:
    """마스크(~ref & ~pad) 토큰만 Euler ODE 로 채운 layout [B,N,4] (page_w 단위).
    관측(ref) 토큰은 layout_ref 그대로(bit-identical). ref_mask 전부 False 허용(무레퍼런스 모드)."""
    dev = model.mean.device
    char_ids, line_id, layout_ref = char_ids.to(dev), line_id.to(dev), layout_ref.to(dev)
    ref_mask, pad_mask = ref_mask.to(dev), pad_mask.to(dev)
    gen = ~ref_mask & ~pad_mask
    g3 = gen[..., None]
    z = torch.randn(layout_ref.shape, generator=torch.Generator().manual_seed(seed)).to(dev)
    x = torch.where(g3, z, model.normalize(layout_ref))
    B = char_ids.shape[0]
    for i in range(steps):
        t = torch.full((B,), 1.0 - i / steps, device=dev)
        v = model(char_ids, line_id, x, gen, t, pad_mask)
        x = torch.where(g3, x - v / steps, x)
    return torch.where(g3, model.denormalize(x), layout_ref)


def save_ckpt(model: LayoutCFM, opt, step: int, path):
    """{"model","kw","optimizer","step"} — 트레이너가 `checkpoint_step_%06d.pth` 와
    `checkpoint_last.pth` 두 경로로 호출(train/core.py 규약)."""
    torch.save({"model": model.state_dict(), "kw": model.kw,
                "optimizer": opt.state_dict(), "step": step}, path)


def load_layout(path, device="cpu"):
    """→ (model.eval() on device, ckpt dict). resume 은 ckpt["optimizer"]/["step"] 사용."""
    ck = torch.load(path, map_location=device, weights_only=True)
    model = LayoutCFM(**ck["kw"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


# ---------------------------------------------------------------- smoke
def _smoke():
    torch.manual_seed(0)
    n_full = sum(p.numel() for p in LayoutCFM(2511).parameters())
    print(f"full config params: {n_full / 1e6:.2f}M")
    assert 30e6 < n_full < 36e6

    B, N, V = 2, 50, 100
    model = LayoutCFM(V, d=64, n_layers=2, n_heads=4, max_len=64)
    ids = torch.randint(0, V, (B, N))
    line = torch.arange(N)[None].expand(B, N) // 17
    pad = torch.zeros(B, N, dtype=torch.bool); pad[1, -7:] = True
    ref = (line == 0) & ~pad
    lay = torch.rand(B, N, 4) * 0.05
    lay[..., 2][line != line.roll(1, 1)] = -0.9   # 줄 시작 Δx
    lay[pad] = 0
    batch = dict(char_ids=ids, layout=lay, line_id=line, ref_mask=ref, pad_mask=pad)
    model.set_norm(lay, pad)
    assert model.std.min() > 1e-4

    # forward shape
    gen = ~ref & ~pad
    v = model(ids, line, model.normalize(lay), gen, torch.rand(B), pad)
    assert v.shape == (B, N, 4), v.shape

    # make_gen_mask: 500 회
    rng = torch.Generator().manual_seed(1)
    n_contig = n_iid = n_all = 0
    for _ in range(500):
        g = make_gen_mask(ref, pad, rng)
        for b in range(B):
            if torch.equal(g[b], ~pad[b]):
                n_all += 1
                continue
            assert not (g[b] & ref[b]).any() and not (g[b] & pad[b]).any()
            assert g[b].sum() >= 1
            idx = g[b].nonzero().flatten()
            if idx.numel() == idx[-1] - idx[0] + 1 and idx.numel() > 1:
                n_contig += 1
            else:
                n_iid += 1
    print(f"make_gen_mask: contig={n_contig} iid={n_iid} all={n_all}")
    assert n_contig > 0 and n_iid > 0 and n_all > 0

    # loss + grad
    loss = cfm_loss(model, batch, torch.Generator().manual_seed(2))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    model.zero_grad()

    # sample
    out = sample(model, ids, line, lay, ref, pad, steps=2)
    assert out.shape == lay.shape and torch.isfinite(out).all()
    assert torch.equal(out[ref], lay[ref]) and torch.equal(out[pad], lay[pad])
    out0 = sample(model, ids, line, lay, torch.zeros_like(ref), pad, steps=2)
    assert torch.isfinite(out0).all()

    # 고정 배치 200 step
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    init = None
    for _ in range(200):
        l = cfm_loss(model, batch, torch.Generator().manual_seed(3))   # rng reset → t/z/mask 고정
        init = init or l.item()
        opt.zero_grad(); l.backward(); opt.step()
    print(f"fit: L1 {init:.4f} -> {l.item():.4f}")
    assert l.item() < init / 10

    # ckpt round-trip
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "checkpoint_last.pth")
    save_ckpt(model, opt, 200, p)
    m2, ck = load_layout(p)
    assert ck["step"] == 200 and ck["kw"] == model.kw and "optimizer" in ck
    model.eval()
    with torch.no_grad():
        a = model(ids, line, model.normalize(lay), gen, torch.full((B,), 0.5), pad)
        b_ = m2(ids, line, m2.normalize(lay), gen, torch.full((B,), 0.5), pad)
    assert torch.equal(a, b_) and torch.equal(m2.mean, model.mean)
    print("smoke OK")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        _smoke()
    else:
        print(__doc__)
