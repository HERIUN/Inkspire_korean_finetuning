"""InkSpire 이미지 모델 학습 — FLUX.1-Fill-dev + LoRA(r=32, 115.9M), 순수 시각 조건 인페인팅 (models/inkspire.py).

데이터: KoreanPageDataset P×P 패치 {x, xc, mask}(mask_mode rand=R-Mask / top). val 은 mode="line"
(첫 줄 아래 전부 마스크 = one-shot 추론 모사, GT xc = oracle 레이아웃) 고정 배치.
손실: 마스크 영역 latent 토큰만의 CFM MSE (식 10-11). Prodigy lr=1 wd 0.01 batch 4 20k iters(논문) / adamw 옵션.
로그: train_loss.csv(step,loss,it_s,peak_mem_gb,d) · val_loss.csv(step,val_loss) · samples/val_s%06d.png
      열 = [x | xc | x·(1−mask) | x_hat].
ckpt: out/lora_step_%06d/{pytorch_lora_weights.safetensors, trainer.pt(optimizer,step,rng)} + lora_last 심링크.
      resume 는 그 디렉토리(load_flux(lora_ckpt=…) + trainer.pt).

★ grad-ckpt 는 grad 가 켜진 forward 에서만 동작(train() 무관) — 항상 on. P=512 B=4 ≈ 40GB 예상.
  OOM 사다리: vae.enable_tiling() → --batch-size 1 --grad-accum 4 → 캐시 토큰 64.
선행: ./train.sh fetch-flux (hf auth login + 라이선스 수락). 없으면 데이터 구성까지 돌고 한글 안내로 종료.

예:
  ./train.sh inkspire                                                        # configs/inkspire.yaml
  ./train.sh inkspire --out finetune_runs/inkspire_smoke --max-steps 2 --batch-size 1 --P 256 \\
      --num-workers 0 --val-every 1 --val-batches 1 --val-steps 2 --save-every 1
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from configs import loader as _config                                             # noqa: E402
from custom_datasets.korean.page import STD_FONT, KoreanPageDataset                         # noqa: E402
from infer.inkspire import load_flux_or_exit                                       # noqa: E402
from train.core import (PATH_ARGS, data_paths_of, load_rng_state, log_run_config,   # noqa: E402
                        require_optim_state, rng_state)


def make_parser():
    B = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="finetune_runs/inkspire_p512")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None, help="lora_step_XXXXXX 디렉토리 (safetensors + trainer.pt)")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--flux-repo", default="black-forest-labs/FLUX.1-Fill-dev")
    p.add_argument("--empty-cache", default=str(HERE / "model_zoo/flux_fill_empty_prompt.pt"))
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--rape", action=B, default=True, help="90° 회전 후 [X|Xc] 연결")
    p.add_argument("--ape", action=B, default=True, help="Xc 토큰 RoPE id = 대응 X 토큰 id")
    p.add_argument("--guidance", type=float, default=30.0, help="학습=추론 고정")
    p.add_argument("--grad-ckpt", action=B, default=True)
    p.add_argument("--optimizer", choices=["prodigy", "adamw"], default="prodigy")
    p.add_argument("--lr", type=float, default=1.0, help="prodigy 1.0 / adamw 1e-4")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--clip-grad", type=float, default=1.0, help="0 = 해제 (dreambooth 관례, 논문 미기재)")
    for k in PATH_ARGS:   # 입력 자산 4키. korean_fonts_dir = 페이지 렌더 폰트 풀
        p.add_argument(f"--{k.replace('_', '-')}", default=None)
    p.add_argument("--P", type=int, default=512, help="패치 크기(16 배수)")
    p.add_argument("--page-w", type=int, default=1024)
    p.add_argument("--std-font", default=str(STD_FONT), help="Xc 렌더 표준폰트")
    p.add_argument("--mask-mode", choices=["rand", "top"], default="rand")
    p.add_argument("--exclude-fonts", nargs="*", default=None)
    p.add_argument("--aug", action=B, default=True)
    p.add_argument("--val-every", type=int, default=500, help="0=off")
    p.add_argument("--val-batches", type=int, default=2)
    p.add_argument("--val-seed", type=int, default=1234)
    p.add_argument("--val-steps", type=int, default=20, help="generate ODE step")
    p.add_argument("--val-fonts-dir", default=None, help="held-out 폰트(기본 assets/fonts/test)")
    p.add_argument("--allow-fresh-optim", action="store_true")
    p.add_argument("--config", default=str(HERE / "configs/inkspire.yaml"))
    return p


def _u8(t):
    return ((t[0].float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)


def main():
    args = _config.parse_args(make_parser(), default_config=str(HERE / "configs/inkspire.yaml"))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    log_run_config(out, args)
    start_step = 0
    st = None
    if args.resume:
        st = torch.load(Path(args.resume) / "trainer.pt", map_location="cpu", weights_only=False)
        start_step = int(st.get("step", 0))

    # ── 데이터 (FLUX 로드보다 먼저 — 가중치가 없어도 여기까지는 검증된다) ──
    kw = dict(P=args.P, page_w=args.page_w, sampler_cfg=args.sampler, paths=data_paths_of(args),
              exclude_fonts=args.exclude_fonts, aug=args.aug, std_font=args.std_font)
    ds = KoreanPageDataset(mode=args.mask_mode, length=(args.max_steps - start_step) * args.batch_size * args.grad_accum,
                           seed=args.seed + start_step, fonts_dir=args.korean_fonts_dir, **kw)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0)
    print(f"train fonts {len(ds.fonts)}  P={args.P} mask={args.mask_mode}")
    val = []
    if args.val_every > 0:
        vds = KoreanPageDataset(mode="line", length=args.val_batches * args.batch_size, seed=args.val_seed,
                                fonts_dir=args.val_fonts_dir or str(HERE / "assets/fonts/test"), **kw)
        val = list(DataLoader(vds, batch_size=args.batch_size, num_workers=args.num_workers))
        print(f"val fonts {len(vds.fonts)}  batches {len(val)} (mode=line)")

    # ── FLUX + LoRA + 옵티마이저 ──
    tr, vae, cache = load_flux_or_exit(device, args.empty_cache, repo=args.flux_repo, lora_rank=args.lora_rank,
                                       lora_alpha=args.lora_alpha, lora_ckpt=args.resume, grad_ckpt=args.grad_ckpt)
    from models.inkspire import build_inputs, cfm_loss, generate, save_lora
    tr.train()
    params = [p for p in tr.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params):,}")
    if args.optimizer == "prodigy":
        from prodigyopt import Prodigy   # dreambooth-flux 레시피
        opt = Prodigy(params, lr=args.lr, weight_decay=args.weight_decay, safeguard_warmup=True, use_bias_correction=True)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if st is not None:
        require_optim_state(st, opt, args.allow_fresh_optim, str(Path(args.resume) / "trainer.pt"))
        if load_rng_state(st.get("rng")):
            print("  RNG state restored")
        print(f"resume {args.resume} step={start_step}")

    tl = (out / "train_loss.csv").open("a", newline="", encoding="utf-8"); tw = csv.writer(tl)
    if tl.tell() == 0:
        tw.writerow(["step", "loss", "it_s", "peak_mem_gb", "d", "timestamp"])
    vl = (out / "val_loss.csv").open("a", newline="", encoding="utf-8"); vw = csv.writer(vl)
    if vl.tell() == 0:
        vw.writerow(["step", "val_loss"])

    def to_dev(b):
        return (b["x"].to(device), b["xc"].to(device), b["mask"].to(device))

    @torch.no_grad()
    def run_val(step):
        tr.eval()
        losses, rows = [], []
        for b in val:
            x, xc, m = to_dev(b)
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(args.val_seed)      # ★ VAE latent_dist.sample() 도 이 안에 — 밖이면 val loss 에 잡음이 섞인다
                inp = build_inputs(x, xc, m, vae, args.rape, args.ape)
                losses.append(cfm_loss(tr, inp, cache, args.guidance).item())
            if len(rows) < 4:
                xh = generate(tr, vae, x, xc, m, cache, steps=args.val_steps, guidance=args.guidance,
                              seed=args.val_seed, rape=args.rape, ape=args.ape)
                for i in range(min(x.shape[0], 4 - len(rows))):
                    masked = x[i] * (1 - m[i]) + 0.5 * m[i]          # 마스크 영역 회색
                    rows.append(np.hstack([_u8(x[i]), _u8(xc[i]), _u8(masked), _u8(xh[i])]))
        Image.fromarray(np.vstack(rows)).save(out / "samples" / f"val_s{step:06d}.png")
        tr.train()
        v = sum(losses) / len(losses)
        print(f"  [val@{step}] loss={v:.4f}  → samples/val_s{step:06d}.png")
        vw.writerow([step, f"{v:.6f}"]); vl.flush()

    def save(step):
        d = out / f"lora_step_{step:06d}"
        save_lora(tr, d)
        torch.save({"optimizer": opt.state_dict(), "step": step, "rng": rng_state()}, d / "trainer.pt")
        link = out / "lora_last"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(d.name, link)
        print(f"  saved {d} (lora_last → {d.name})")

    print(f"training {start_step} -> {args.max_steps}  (batch {args.batch_size}×{args.grad_accum}, {args.optimizer})")
    step, micro, run_loss, run_n, t0 = start_step, 0, 0.0, 0, time.time()
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for b in loader:                       # 길이 = (max_steps−start)·batch·accum, drop_last → 정확히 max_steps 에서 끝난다
        if step >= args.max_steps:
            break
        x, xc, m = to_dev(b)
        inp = build_inputs(x, xc, m, vae, args.rape, args.ape)          # no_grad(VAE 동결)
        # ★ bf16 autocast = accelerate mixed_precision 과 동일(dreambooth-flux). LoRA 파라미터/grad 는 fp32 유지.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = cfm_loss(tr, inp, cache, args.guidance)
        (loss / args.grad_accum).backward()
        run_loss += loss.item(); run_n += 1; micro += 1
        if micro % args.grad_accum:
            continue
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0 or step == args.max_steps:
            its = (run_n / args.grad_accum) / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            d = opt.param_groups[0].get("d", args.lr)
            print(f"step {step}/{args.max_steps}  loss={run_loss / run_n:.4f}  {its:.2f} it/s  peak={peak:.1f}GB  d={d:.2e}")
            tw.writerow([step, f"{run_loss / run_n:.6f}", f"{its:.3f}", f"{peak:.2f}", f"{d:.3e}",
                         datetime.datetime.now().isoformat(timespec="seconds")]); tl.flush()
            run_loss, run_n, t0 = 0.0, 0, time.time()
        if val and step % args.val_every == 0:
            run_val(step)
        if args.save_every > 0 and step % args.save_every == 0:
            save(step)
    if args.save_every <= 0 or step % args.save_every != 0:
        save(step)
    tl.close(); vl.close()
    print("done.")


if __name__ == "__main__":
    main()
