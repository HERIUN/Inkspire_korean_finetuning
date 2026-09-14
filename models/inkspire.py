"""InkSpire(ICLR 2026, Wang et al.) 이미지 모델 — FLUX.1-Fill-dev + LoRA, 순수 시각 조건 인페인팅.

p(X | Xs, Xc): 스타일 레퍼런스 X(마스크 밖)와 표준폰트 콘텐츠 Xc 를 공간 연결한 캔버스를 FLUX-Fill 로 채운다.
텍스트 인코더는 쓰지 않는다 — 빈 프롬프트("") 임베딩을 `tools/fetch_flux.py` 가 1회 캐시(`EMPTY_CACHE`)하고
학습·추론은 그 dict 만 읽는다(T5-XXL/CLIP 미탑재). 0 을 넣지 않는 이유: context_embedder/temb 에 실제 분포의
빈 프롬프트가 들어가야 12B 백본이 분포 안에서 시작한다.

★ 격자 3층: 픽셀 16 = VAE latent 2×2 = packed 토큰 1. H, W 는 16 의 배수여야 한다(`TOK`).
★ R-APE: x/xc/mask 를 90° 시계방향(rot90 k=-1) 회전 후 폭 방향으로 [X | Xc] 연결. 역회전은 k=+1.
   APE: Xc 토큰의 RoPE id = 같은 자리 X 토큰의 id (X 절반 격자 id 를 폭 방향으로 2회 타일).
★ 마스크는 X 절반만 — Xc 절반은 항상 보이며(0 패딩) 손실에서도 제외된다.
★ transformer 내부에서 timestep·guidance 에 ×1000 → 여기서는 σ∈(0,1] 그대로 넘긴다. img_ids 는 2-D [L,3].
★ 학습 σ 와 추론 σ 는 같은 shift 스케줄(`calculate_shift(L)` → `time_shift`)을 쓴다(train/infer 동일성).
★ guidance=30 고정(학습=추론). Fill-dev 의 인페인팅 동작점; 텍스트가 없으니 상수 조건 스칼라일 뿐이다.
★ grad-ckpt 는 `enable_gradient_checkpointing()` 후 grad 가 켜진 forward 에서만 동작한다
   (diffusers 0.38 의 게이트는 `torch.is_grad_enabled()`; `train()` 여부와 무관). generate 는 no_grad.

체크포인트: `save_lora(tr, dir)` → `dir/pytorch_lora_weights.safetensors`(LoraConfig 메타 포함, diffusers
`load_lora_weights` 호환). 복원은 `load_flux(lora_ckpt=dir)`.

self-check:
  python models/inkspire.py --smoke                  # 다운로드 없음, tiny 모델, 수 초
  python models/inkspire.py --smoke --with-weights   # 실제 가중치(hf 로그인 + fetch_flux 선행) — trainable 115,912,704
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline, calculate_shift
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
)

HERE = Path(__file__).resolve().parents[1]
REPO = "black-forest-labs/FLUX.1-Fill-dev"
EMPTY_CACHE = HERE / "model_zoo" / "flux_fill_empty_prompt.pt"
TOK = 16   # ★ 픽셀/토큰
LAT_C = 16  # Fill-dev VAE latent 채널

# 표 8 의 13 종. full-match 정규식 — suffix 매칭이면 최상위 `transformer.proj_out` 까지 잡힌다(+100k).
LORA_TARGETS = (
    r"x_embedder"
    r"|transformer_blocks\.\d+\.(norm1\.linear|attn\.to_[qkv]|attn\.to_out\.0|ff\.net\.2)"
    r"|single_transformer_blocks\.\d+\.(norm\.linear|proj_mlp|proj_out|attn\.to_[qkv])"
)

_pack = FluxFillPipeline._pack_latents
_unpack = FluxFillPipeline._unpack_latents
_grid_ids = FluxFillPipeline._prepare_latent_image_ids


def _sched():
    # 기본 config = FLUX 스케줄러(base_shift 0.5 / max_shift 1.15, exponential) → 허브 다운로드 불필요
    return FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)


# ---------------------------------------------------------------- 로드 / 저장
def _attach_lora(tr, vae, device, lora_rank, lora_alpha, lora_ckpt, grad_ckpt):
    """base 전부 freeze → LoRA 주입(신규 또는 ckpt 복원) → LoRA 만 fp32·학습 → grad-ckpt."""
    from peft import LoraConfig

    tr.requires_grad_(False)
    vae.requires_grad_(False).eval()
    if lora_ckpt:
        # 메타에서 LoraConfig 복원, add_adapter 불필요. use_safetensors 기본 None 이면 .bin 만 찾으므로 명시.
        # ★ adapter_name="default" — 안 주면 "default_0" 으로 등록돼 resume 후 save_lora(adapter "default")가 실패한다.
        tr.load_lora_adapter(str(lora_ckpt), prefix=None, use_safetensors=True, adapter_name="default")
    else:
        tr.add_adapter(LoraConfig(r=lora_rank, lora_alpha=lora_alpha, init_lora_weights="gaussian",
                                  target_modules=LORA_TARGETS))
    for n, p in tr.named_parameters():
        p.requires_grad_("lora" in n)
    cast_training_params(tr, torch.float32)
    if grad_ckpt:
        tr.enable_gradient_checkpointing()
    return tr.to(device), vae.to(device)


def load_flux(repo=REPO, device="cuda", dtype=torch.bfloat16, lora_rank=32, lora_alpha=32,
              lora_ckpt=None, grad_ckpt=True):
    """(transformer, vae). transformer.dtype == dtype(base), LoRA 파라미터만 fp32/requires_grad."""
    tr = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
    return _attach_lora(tr, vae, device, lora_rank, lora_alpha, lora_ckpt, grad_ckpt)


def save_lora(transformer, out_dir):
    """`out_dir/pytorch_lora_weights.safetensors` (fp32, LoraConfig 메타 포함)."""
    transformer.save_lora_adapter(str(out_dir))


@torch.no_grad()
def empty_prompt_cache(pipe, path=EMPTY_CACHE):
    """빈 프롬프트 임베딩 1회 계산 → {prompt_embeds[1,512,4096], pooled[1,768], text_ids[512,3]} bf16 저장.
    ponytail: 512 토큰 유지. 64 로 줄이면 ~15% 절약되나 Fill-dev 학습 분포와 달라진다."""
    pe, pooled, tids = pipe.encode_prompt(prompt="", prompt_2=None, max_sequence_length=512)
    d = {"prompt_embeds": pe.to(torch.bfloat16).cpu(), "pooled": pooled.to(torch.bfloat16).cpu(),
         "text_ids": tids.cpu()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, path)
    return d


# ---------------------------------------------------------------- 입력 구성
def _rot(t, k):
    return torch.rot90(t, k, dims=(2, 3))


def _enc(vae, img):
    """[B,1,h,w]∈[-1,1] → 정규화 latent [B,16,h/8,w/8]."""
    z = vae.encode(img.to(vae.dtype).repeat(1, 3, 1, 1)).latent_dist.sample()
    return (z - vae.config.shift_factor) * vae.config.scaling_factor


def _dec(vae, lat):
    img = vae.decode(lat.to(vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor).sample
    return img.float().mean(1, keepdim=True)


@torch.no_grad()
def build_inputs(x, xc, mask, vae, rape=True, ape=True):
    """x, xc [B,1,H,W]∈[-1,1], mask [B,1,H,W] 1=생성(X 절반만).
    → x0_packed [B,L,64], cond_packed [B,L,320](masked latent 64 + mask 256), img_ids [L,3],
      loss_mask [B,L] bool, hw=(회전 캔버스 H, W) — L = (H/16)·(2W/16) (rape 시 H,W 가 바뀜)."""
    B, _, H, W = x.shape
    assert H % TOK == 0 and W % TOK == 0, f"H,W 는 {TOK} 배수: {(H, W)}"
    k = -1 if rape else 0                                   # ★ 시계방향; 역은 +1
    x, xc, mask = (_rot(t.float(), k) for t in (x, xc, mask))
    Hr, Wr = x.shape[-2:]
    I = torch.cat([x, xc], -1)                              # [B,1,Hr,2Wr]
    Im = torch.cat([mask, torch.zeros_like(mask)], -1)      # ★ Xc 절반 마스크 0
    h, w = Hr // 8, 2 * Wr // 8
    x0 = _pack(_enc(vae, I), B, LAT_C, h, w)
    masked = _pack(_enc(vae, I * (1 - Im)), B, LAT_C, h, w)
    m = Im[:, 0].view(B, h, 8, w, 8).permute(0, 2, 4, 1, 3).reshape(B, 64, h, w)   # FluxFill 규약
    mp = _pack(m, B, 64, h, w)                              # [B,L,256]
    cond = torch.cat([masked, mp.to(masked.dtype)], -1)
    hh, ww = h // 2, w // 4                                 # 토큰 격자: X 절반 = hh×ww
    if ape:
        ids = _grid_ids(1, hh, ww, x.device, torch.float32).view(hh, ww, 3)
        ids = torch.cat([ids, ids], 1).reshape(-1, 3)       # ★ Xc 토큰 id == 대응 X 토큰 id
    else:
        ids = _grid_ids(1, hh, 2 * ww, x.device, torch.float32)
    return dict(x0_packed=x0, cond_packed=cond, img_ids=ids, loss_mask=mp.amax(-1) > 0, hw=(Hr, 2 * Wr))


def _pred(tr, xt, cond, sigma, cache, guidance, ids):
    B, dev, dt = xt.shape[0], xt.device, tr.dtype
    return tr(hidden_states=torch.cat([xt, cond], -1).to(dt),
              timestep=sigma.to(dev),                       # ★ 내부 ×1000
              guidance=torch.full((B,), float(guidance), device=dev),
              pooled_projections=cache["pooled"].to(dev, dt).expand(B, -1),
              encoder_hidden_states=cache["prompt_embeds"].to(dev, dt).expand(B, -1, -1),
              txt_ids=cache["text_ids"].to(dev), img_ids=ids.to(dev), return_dict=False)[0]


def cfm_loss(transformer, inp, cache, guidance=30.0):
    """식 10-11: 마스크 토큰만의 v-prediction MSE. σ = time_shift(calculate_shift(L), logit_normal u)."""
    x0, cond, m = inp["x0_packed"].float(), inp["cond_packed"], inp["loss_mask"].float()
    B, L, _ = x0.shape
    u = compute_density_for_timestep_sampling("logit_normal", B, logit_mean=0.0, logit_std=1.0, device=x0.device)
    sigma = _sched().time_shift(calculate_shift(L), 1.0, u)
    z = torch.randn_like(x0)
    s = sigma.view(B, 1, 1)
    v = _pred(transformer, (1 - s) * x0 + s * z, cond, sigma, cache, guidance, inp["img_ids"]).float()
    err = ((v - (z - x0)) ** 2).mean(-1)               # [B,L]
    return (err * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def generate(transformer, vae, x, xc, mask, cache, steps=20, guidance=30.0, seed=0, rape=True, ape=True):
    """Euler ODE steps 회 → X 절반 역회전·gray → x·(1−mask) + gen·mask, [B,1,H,W]∈[-1,1]."""
    inp = build_inputs(x, xc, mask, vae, rape, ape)
    x0 = inp["x0_packed"]
    B, L, _ = x0.shape
    sched = _sched()
    sched.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps).tolist(), mu=calculate_shift(L))
    sig = sched.sigmas.to(x0.device)                        # [steps+1], 끝 0
    g = torch.Generator(device=x0.device).manual_seed(seed)
    lat = torch.randn(x0.shape, generator=g, device=x0.device, dtype=torch.float32)
    for i in range(steps):
        v = _pred(transformer, lat, inp["cond_packed"], sig[i].expand(B), cache, guidance, inp["img_ids"]).float()
        lat = lat + (sig[i + 1] - sig[i]) * v
    Hr, W2 = inp["hw"]
    img = _dec(vae, _unpack(lat, Hr, W2, 8))               # [B,1,Hr,W2]
    gen = _rot(img[..., : W2 // 2], 1 if rape else 0).clamp(-1, 1)
    mask = mask.float()
    return x.float() * (1 - mask) + gen * mask


# ---------------------------------------------------------------- self-check
def _tiny(device, lora_ckpt=None):
    """다운로드 없는 소형 VAE(8× 압축, 16ch) + FLUX(1+1 층, in 384/out 64, guidance_embeds)."""
    torch.manual_seed(0)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8),
                        layers_per_block=1, latent_channels=LAT_C, norm_num_groups=8, scaling_factor=0.3611,
                        shift_factor=0.1159, use_quant_conv=False, use_post_quant_conv=False,
                        mid_block_add_attention=False)
    tr = FluxTransformer2DModel(patch_size=1, in_channels=384, out_channels=64, num_layers=1, num_single_layers=1,
                                attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                                pooled_projection_dim=16, guidance_embeds=True, axes_dims_rope=(4, 6, 6))
    cache = {"prompt_embeds": torch.randn(1, 8, 32), "pooled": torch.randn(1, 16), "text_ids": torch.zeros(8, 3)}
    return *_attach_lora(tr, vae, device, 4, 4, lora_ckpt, grad_ckpt=True), cache


def _smoke(device):
    import tempfile

    tr, vae, cache = _tiny(device)
    for n, p in tr.named_parameters():      # lora_B 는 0 초기화 → grad(lora_A)=0, LoRA 출력 기여 0. 검사 의미를 위해 교란
        if "lora_B" in n:
            p.data.normal_(std=0.02)
    # (1) LoRA 대상 = 13 종, 최상위 proj_out 제외
    names = {re.sub(r"\.\d+\.", ".N.", n) for n, m in tr.named_modules() if hasattr(m, "lora_A")}
    expect = {"x_embedder"} | {f"transformer_blocks.N.{s}" for s in
                               ("norm1.linear", "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.2")} \
        | {f"single_transformer_blocks.N.{s}" for s in
           ("norm.linear", "proj_mlp", "proj_out", "attn.to_q", "attn.to_k", "attn.to_v")}
    assert names == expect, names ^ expect
    assert "proj_out" not in names
    # (2) P=64 shape
    P = 64
    x = torch.rand(1, 1, P, P, device=device) * 2 - 1
    xc = torch.rand(1, 1, P, P, device=device) * 2 - 1
    mask = torch.zeros(1, 1, P, P, device=device)
    mask[..., 20:50, 10:40] = 1                             # 16 격자에 안 맞는 사각형
    inp = build_inputs(x, xc, mask, vae)
    ids = inp["img_ids"]
    assert inp["x0_packed"].shape == (1, 32, 64) and inp["cond_packed"].shape == (1, 32, 320), inp["cond_packed"].shape
    assert ids.shape == (32, 3) and inp["loss_mask"].shape == (1, 32)
    # (3) APE: 토큰은 행 우선이라 X/Xc 가 행마다 교차 → 행 안의 앞/뒤 절반이 같아야 함 (ids[:16]==ids[16:] 은 열 타일에선 거짓)
    g = ids.view(4, 8, 3)
    assert torch.equal(g[:, :4], g[:, 4:]) and len(torch.unique(ids, dim=0)) == 16
    assert len(torch.unique(build_inputs(x, xc, mask, vae, ape=False)["img_ids"], dim=0)) == 32
    # (4) loss_mask == 회전 마스크의 16px max-pool, Xc 절반 0
    mask_r = _rot(mask, -1)
    assert inp["loss_mask"].sum().item() == F.max_pool2d(mask_r, TOK).sum().item()
    assert inp["loss_mask"].view(1, 4, 8)[..., 4:].sum() == 0 and inp["loss_mask"].any()
    # (5) rot 왕복
    assert torch.equal(_rot(_rot(x, -1), 1), x)
    # (6) cfm_loss backward → LoRA grad 만
    loss = cfm_loss(tr, inp, cache)
    assert torch.isfinite(loss), loss
    loss.backward()
    for n, p in tr.named_parameters():
        if "lora" in n:
            assert p.grad is not None and p.grad.abs().sum() > 0, n
        else:
            assert p.grad is None, n
    # (7) generate(2 step): shape·범위·비마스크 보존
    out = generate(tr, vae, x, xc, mask, cache, steps=2)
    assert out.shape == (1, 1, P, P) and out.min() >= -1 and out.max() <= 1
    assert torch.equal(out[mask == 0], x[mask == 0])
    # (8) 비정방
    x2 = torch.rand(1, 1, 32, 64, device=device) * 2 - 1
    m2 = torch.zeros_like(x2); m2[..., 16:, :] = 1
    assert generate(tr, vae, x2, x2, m2, cache, steps=2).shape == (1, 1, 32, 64)
    # (9) save_lora → load_flux(lora_ckpt) 왕복: 같은 출력
    with tempfile.TemporaryDirectory() as d:
        save_lora(tr, d)
        assert (Path(d) / "pytorch_lora_weights.safetensors").exists()
        tr2, _, _ = _tiny(device, lora_ckpt=d)
        with torch.no_grad():
            sig = torch.full((1,), 0.5, device=device)
            a = _pred(tr, inp["x0_packed"], inp["cond_packed"], sig, cache, 30.0, ids)
            b = _pred(tr2, inp["x0_packed"], inp["cond_packed"], sig, cache, 30.0, ids)
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
        assert sum(p.numel() for p in tr2.parameters() if p.requires_grad) == \
            sum(p.numel() for p in tr.parameters() if p.requires_grad)
    print(f"[smoke] ok  device={device}  loss={loss.item():.4f}  lora_modules={len(names)}")


def _with_weights(device):
    if device == "cpu":
        print("[skip] --with-weights 는 GPU 필요"); return 2
    if not EMPTY_CACHE.exists():
        print(f"[skip] {EMPTY_CACHE} 없음 — 먼저 `hf auth login` + 라이선스 수락 후 `python tools/fetch_flux.py`"); return 2
    try:
        tr, vae = load_flux(device=device)
    except Exception as e:   # 게이트/미다운로드
        print(f"[skip] FLUX 가중치 로드 실패({type(e).__name__}): {e}\n"
              f"       `hf auth login` → https://huggingface.co/{REPO} 라이선스 수락 → python tools/fetch_flux.py"); return 2
    n = sum(p.numel() for p in tr.parameters() if p.requires_grad)
    assert n == 115_912_704, n
    cache = torch.load(EMPTY_CACHE)
    P = 256
    x = torch.rand(1, 1, P, P, device=device) * 2 - 1
    xc = torch.rand(1, 1, P, P, device=device) * 2 - 1
    mask = torch.zeros(1, 1, P, P, device=device); mask[..., P // 4:, :] = 1
    tr.train()
    torch.cuda.reset_peak_memory_stats()
    loss = cfm_loss(tr, build_inputs(x, xc, mask, vae), cache)
    loss.backward()
    print(f"[with-weights] trainable={n:,} loss={loss.item():.4f} "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB (P={P}, B=1, fwd+bwd)")
    tr.eval()
    out = generate(tr, vae, x, xc, mask, cache, steps=2)
    assert out.shape == x.shape and torch.isfinite(out).all()
    print(f"[with-weights] generate ok, peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--with-weights", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.smoke:
        _smoke(a.device)
    if a.with_weights:
        sys.exit(_with_weights(a.device))
