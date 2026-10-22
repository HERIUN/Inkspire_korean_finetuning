"""InkSpire 이미지 모델 — FLUX.3-Image-Klein-Base(HF black-forest-labs/flux-3-image-klein-base, 5.5B) + LoRA 백본. models/inkspire.py(FLUX.1-Fill)와 같은 API.

FLUX.1-Fill 과 달리 Klein 에는 인페인팅 채널(마스크된 latent ⊕ 마스크)이 없다. 대신 깨끗한 참조 이미지를
별도 토큰 스트림(`image_cond`, timestep 0)으로 받고, 참조의 위치 id 는 생성 이미지와 같은 (h, w) 에 `l` 축만
다르게 둔다(공식 img2img 규약). 그래서 InkSpire 를 다음처럼 옮긴다:
  생성 토큰  = X 캔버스 (l=0)
  참조 1     = X·(1−mask) + 회색(0)·mask   (l=−1)   — 마스크는 회색 칠로만 전달(채널 없음)
  참조 2     = Xc                           (l=−2)   — 같은 (h, w) id = 논문 APE 가 모델에 원래 들어 있다
  → R-APE 회전·[X|Xc] 연결이 필요 없다. rape/ape/guidance 인자는 API 호환용으로 받고 무시한다.

★ 격자: Flux2 VAE 8× · latent 32ch → patchify 2×2 = 128ch 토큰, 픽셀 16 = 토큰 1 (FLUX.1 과 같다).
   latent 정규화 = vae.bn running 통계(patchify 후 128ch), encode 는 mode() (공식 파이프라인과 동일).
★ σ: shift 6.93 고정(해상도 무관, 공식 FLOW_SHIFT). 학습 u ~ U(0,1) → σ = shift·u / (1 + (shift−1)·u)
   (공식 dreambooth weighting "none"), 추론 linspace(1, 1/steps) 를 같은 shift 로. 학습=추론.
★ 추론: 매 step 후 비마스크 토큰을 (1−σ)·x0 + σ·z 로 되돌린다 — 학습에서 그 토큰들은 늘 GT 의 노이즈 버전이었으므로
   추론도 같게(마스크 채널이 없으니 이게 마스크를 알려주는 두 번째 길). 손실은 마스크 토큰만(식 10-11).
★ 텍스트: 빈 프롬프트("") Qwen3-VL 임베딩을 `empty_prompt_cache()` 가 1회 캐시 → 학습·추론은 그 dict 만 읽는다.
   base 는 CFG 모델이고 "" 가 무조건부 프롬프트라 분포 안이다. CFG 는 쓰지 않는다.

self-check:
  python models/inkspire_klein.py --smoke                 # 다운로드 없음, tiny 모델
  python models/inkspire_klein.py --cache                 # 빈 프롬프트 캐시 생성(텍스트 인코더 1회 로드, = ./train.sh fetch-klein)
  python models/inkspire_klein.py --smoke --with-weights  # 실제 가중치 fwd+bwd·generate·peak 메모리
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from models.inkspire import _attach_lora, _cache_or_exit, save_lora   # noqa: E402,F401  (save_lora 재노출)

REPO = "black-forest-labs/flux-3-image-klein-base"
EMPTY_CACHE = HERE / "model_zoo" / "klein_empty_prompt.pt"
HOWTO = f"""{REPO} 가중치/빈 프롬프트 캐시가 없습니다 (게이트 repo, 비상업 라이선스).
  1) hf auth login            (https://huggingface.co/settings/tokens 의 read 토큰)
  2) https://huggingface.co/{REPO} 에서 라이선스 수락
  3) ./train.sh fetch-klein   (텍스트 인코더 1회 로드 + {EMPTY_CACHE.relative_to(HERE)}, 나머지 가중치는 첫 학습 때 받는다)"""
TOK = 16
SHIFT = 6.93
ROPE_SCALE = 0.5
# 공식 img2img 레시피(attn qkv/out + ff) 를 생성·참조 mode 블록과 joint 블록에. 텍스트 mode 블록은 입력이 상수라 제외.
# 입력 projection(x_embedders) 은 논문 표 8 이 x_embedder 를 포함한 것을 따른다.
LORA_TARGETS = (
    r"(mode_blocks\.(image|image_cond)\.\d+|single_transformer_blocks\.\d+)"
    r"\.(attn\.to_[qkv]|attn\.to_out\.0|ff\.linear_in|ff\.linear_out)"
    r"|x_embedders\.(image|image_cond)"
)


def load_flux(repo=REPO, device="cuda", dtype=torch.bfloat16, lora_rank=32, lora_alpha=32,
              lora_ckpt=None, grad_ckpt=True):
    """(transformer, vae). 이름은 models/inkspire.load_flux 와 맞춘다(호출부 공용)."""
    tr = Flux3Transformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=dtype)
    vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
    return _attach_lora(tr, vae, device, lora_rank, lora_alpha, lora_ckpt, grad_ckpt, LORA_TARGETS)


@torch.no_grad()
def empty_prompt_cache(repo=REPO, path=EMPTY_CACHE, device="cuda"):
    """"" → {prompt_embeds[1,80,20480] bf16, text_ids[80,4]}. 공식 Flux3TextEncoderStep 과 같은 함수로 만든다."""
    from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
    from types import SimpleNamespace
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
    te = Qwen3VLForConditionalGeneration.from_pretrained(repo, subfolder="text_encoder",
                                                         torch_dtype=torch.bfloat16).to(device).eval()
    # AutoProcessor 는 비디오 프로세서 때문에 torchvision 을 요구한다. 텍스트만 쓰므로 토크나이저로 같은 두 메서드를 흉내.
    tok = AutoTokenizer.from_pretrained(repo, subfolder="text_encoder")
    proc = SimpleNamespace(apply_chat_template=tok.apply_chat_template, tokenizer=tok)
    pe = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(te, proc, "", device)
    n = pe.shape[1]
    tids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(n)).float()
    d = {"prompt_embeds": pe.to(torch.bfloat16).cpu(), "text_ids": tids}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, path)
    return d


# ---------------------------------------------------------------- 입력 구성
def _bn(vae, dev):
    m = vae.bn.running_mean.view(1, -1, 1, 1).to(dev)
    s = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(dev)
    return m, s


def _enc(vae, img):
    """[B,1,H,W]∈[-1,1] → 정규화 토큰 latent [B,128,H/16,W/16]."""
    z = vae.encode(img.to(vae.dtype).repeat(1, 3, 1, 1)).latent_dist.mode()
    B, C, h, w = z.shape
    z = z.view(B, C, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, h // 2, w // 2)
    m, s = _bn(vae, z.device)
    return (z.float() - m) / s


def _dec(vae, lat):
    m, s = _bn(vae, lat.device)
    z = lat.float() * s + m
    B, C, h, w = z.shape
    z = z.reshape(B, C // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, h * 2, w * 2)
    return vae.decode(z.to(vae.dtype)).sample.float().mean(1, keepdim=True)


def _pack(z):
    return z.flatten(2).transpose(1, 2)                     # [B,C,h,w] → [B,h·w,C]


def _unpack(t, h, w):
    return t.transpose(1, 2).reshape(t.shape[0], -1, h, w)


def _ids(h, w, l):
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.tensor([l])).float()
    ids[:, 1:3] *= ROPE_SCALE
    return ids


@torch.no_grad()
def build_inputs(x, xc, mask, vae, rape=None, ape=None):
    """x, xc [B,1,H,W]∈[-1,1], mask [B,1,H,W] 1=생성. → x0_packed [B,L,128], cond_packed [B,2L,128],
    img_ids [L,4], cond_ids [2L,4], loss_mask [B,L] bool(16px 칸에 마스크가 하나라도), hw=(h,w) 토큰 격자."""
    B, _, H, W = x.shape
    assert H % TOK == 0 and W % TOK == 0, f"H,W 는 {TOK} 배수: {(H, W)}"
    x, xc, mask = x.float(), xc.float(), mask.float()
    h, w = H // TOK, W // TOK
    cond = torch.cat([_pack(_enc(vae, x * (1 - mask))), _pack(_enc(vae, xc))], 1)   # 마스크 = 회색(0)
    return dict(x0_packed=_pack(_enc(vae, x)), cond_packed=cond, img_ids=_ids(h, w, 0).to(x.device),
                cond_ids=torch.cat([_ids(h, w, -1), _ids(h, w, -2)]).to(x.device),
                loss_mask=F.max_pool2d(mask, TOK).flatten(1) > 0, hw=(h, w))


def _pred(tr, xt, inp, sigma, cache):
    B, dev, dt = xt.shape[0], xt.device, tr.dtype
    return tr(hidden_states=xt.to(dt), timestep=sigma.to(dev, dt),
              encoder_hidden_states=cache["prompt_embeds"].to(dev, dt).expand(B, -1, -1),
              txt_ids=cache["text_ids"].to(dev), img_ids=inp["img_ids"],
              image_cond_hidden_states=inp["cond_packed"].to(dt), image_cond_ids=inp["cond_ids"],
              return_dict=False)[0]


def _shift(u, shift=SHIFT):
    return shift * u / (1 + (shift - 1) * u)


def cfm_loss(transformer, inp, cache, guidance=None):
    """마스크 토큰만의 v-prediction MSE. σ = shift(U(0,1))."""
    x0, m = inp["x0_packed"], inp["loss_mask"].float()
    B = x0.shape[0]
    sigma = _shift(torch.rand(B, device=x0.device))
    z = torch.randn_like(x0)
    s = sigma.view(B, 1, 1)
    v = _pred(transformer, (1 - s) * x0 + s * z, inp, sigma, cache).float()
    err = ((v - (z - x0)) ** 2).mean(-1)
    return (err * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def generate(transformer, vae, x, xc, mask, cache, steps=20, guidance=None, seed=0, rape=None, ape=None):
    """Euler ODE → x·(1−mask) + gen·mask, [B,1,H,W]∈[-1,1]. 비마스크 토큰은 매 step GT 노이즈 버전으로 고정."""
    inp = build_inputs(x, xc, mask, vae)
    x0, keep = inp["x0_packed"], ~inp["loss_mask"][..., None]
    B = x0.shape[0]
    sched = FlowMatchEulerDiscreteScheduler(shift=SHIFT, use_dynamic_shifting=False)
    sched.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps).tolist())
    sig = sched.sigmas.to(x0.device)                        # [steps+1] shift 적용, 끝 0
    g = torch.Generator(device=x0.device).manual_seed(seed)
    z = torch.randn(x0.shape, generator=g, device=x0.device, dtype=torch.float32)
    lat = z.clone()
    for i in range(steps):
        v = _pred(transformer, lat, inp, sig[i].expand(B), cache).float()
        lat = lat + (sig[i + 1] - sig[i]) * v
        lat = torch.where(keep, (1 - sig[i + 1]) * x0 + sig[i + 1] * z, lat)
    gen = _dec(vae, _unpack(lat, *inp["hw"])).clamp(-1, 1)
    mask = mask.float()
    return x.float() * (1 - mask) + gen * mask


# ---------------------------------------------------------------- self-check
def _tiny(device, lora_ckpt=None):
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), layers_per_block=1, latent_channels=4,
                             norm_num_groups=4, mid_block_add_attention=False)
    vae.bn.running_mean.normal_(); vae.bn.running_var.uniform_(0.5, 2)
    tr = Flux3Transformer2DModel(in_channels=16, cond_in_channels=16, num_layers=1, num_single_layers=1,
                                 attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                                 axes_dims_rope=(4, 4, 4, 4))
    cache = {"prompt_embeds": torch.randn(1, 8, 32), "text_ids": torch.cartesian_prod(*(torch.arange(k) for k in (1, 1, 1, 8))).float()}
    return *_attach_lora(tr, vae, device, 4, 4, lora_ckpt, True, LORA_TARGETS), cache


def _smoke(device):
    import re
    import tempfile
    tr, vae, cache = _tiny(device)
    for n, p in tr.named_parameters():
        if "lora_B" in n:
            p.data.normal_(std=0.02)
    names = {re.sub(r"\.\d+\.", ".N.", n) for n, m in tr.named_modules() if hasattr(m, "lora_A")}
    assert not any(n.startswith("txt_mode_blocks") or n == "proj_out" or "context_embedder" in n for n in names), names
    assert {"x_embedders.image", "x_embedders.image_cond", "mode_blocks.image_cond.N.attn.to_q",
            "single_transformer_blocks.N.ff.linear_out"} <= names, names
    # (1) 패치/역패치 왕복 + 정규화 왕복
    z = torch.randn(2, 4, 8, 12, device=device)
    B, C, h, w = z.shape
    zp = z.view(B, C, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, h // 2, w // 2)
    back = zp.reshape(B, C, 2, 2, h // 2, w // 2).permute(0, 1, 4, 2, 5, 3).reshape(B, C, h, w)
    assert torch.equal(back, z)
    # (2) shape · ids · loss_mask
    H, W = 32, 64
    x = torch.rand(1, 1, H, W, device=device) * 2 - 1
    xc = torch.rand(1, 1, H, W, device=device) * 2 - 1
    mask = torch.zeros(1, 1, H, W, device=device); mask[..., 20:, 10:40] = 1
    inp = build_inputs(x, xc, mask, vae)
    L = (H // TOK) * (W // TOK)
    assert inp["x0_packed"].shape == (1, L, 16) and inp["cond_packed"].shape == (1, 2 * L, 16)
    ci = inp["cond_ids"]
    assert torch.equal(ci[:L, 1:3], inp["img_ids"][:, 1:3]) and torch.equal(ci[L:, 1:3], inp["img_ids"][:, 1:3])
    assert (ci[:L, 3] == -1).all() and (ci[L:, 3] == -2).all() and (inp["img_ids"][:, 3] == 0).all()
    assert inp["loss_mask"].sum().item() == F.max_pool2d(mask, TOK).sum().item() == 3   # 행 16~32 × 열 0~48
    # (3) σ shift 범위·단조
    u = torch.linspace(0, 1, 11)
    s = _shift(u)
    assert s[0] == 0 and abs(s[-1] - 1) < 1e-6 and (s.diff() > 0).all() and s[5] > 0.5
    # (4) backward → LoRA 만 grad
    loss = cfm_loss(tr, inp, cache)
    assert torch.isfinite(loss)
    loss.backward()
    for n, p in tr.named_parameters():
        assert (p.grad is not None and p.grad.abs().sum() > 0) if "lora" in n else p.grad is None, n
    # (5) generate: shape · 비마스크 보존
    out = generate(tr, vae, x, xc, mask, cache, steps=2)
    assert out.shape == x.shape and torch.equal(out[mask == 0], x[mask == 0])
    # (6) save → load 왕복
    with tempfile.TemporaryDirectory() as d:
        save_lora(tr, d)
        tr2, _, _ = _tiny(device, lora_ckpt=d)
        with torch.no_grad():
            sg = torch.full((1,), 0.5, device=device)
            a, b = _pred(tr, inp["x0_packed"], inp, sg, cache), _pred(tr2, inp["x0_packed"], inp, sg, cache)
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    print(f"[smoke] ok  device={device}  loss={loss.item():.4f}  lora_module_kinds={len(names)}")


def _with_weights(device):
    if not EMPTY_CACHE.exists():
        print(f"[skip] {EMPTY_CACHE} 없음\n{HOWTO}"); return 2
    tr, vae = load_flux(device=device)
    n = sum(p.numel() for p in tr.parameters() if p.requires_grad)
    cache = torch.load(EMPTY_CACHE)
    for P, B in ((512, 4),):
        x = torch.rand(B, 1, P, P, device=device) * 2 - 1
        mask = torch.zeros(B, 1, P, P, device=device); mask[..., P // 4:, :] = 1
        tr.train(); torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = cfm_loss(tr, build_inputs(x, x, mask, vae), cache)
        loss.backward()
        print(f"[with-weights] trainable={n:,} P={P} B={B} loss={loss.item():.4f} "
              f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
    tr.eval()
    out = generate(tr, vae, x[:1], x[:1], mask[:1], cache, steps=2)
    assert out.shape == x[:1].shape and torch.isfinite(out).all()
    print("[with-weights] generate ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--with-weights", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.smoke:
        _smoke(a.device)
    if a.cache:
        _cache_or_exit(empty_prompt_cache, EMPTY_CACHE, HOWTO, a.device)
    if a.with_weights:
        sys.exit(_with_weights(a.device))
