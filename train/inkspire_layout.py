"""InkSpire 레이아웃 모델 학습 (masked CFM, models/inkspire_layout.py) — 논문 §A 표 1 최선안.

데이터: KoreanLayoutDataset 페이지 전체 토큰 {char_ids, layout[N,4], line_id, ref_mask(첫 줄)} → layout_collate.
손실: 마스크 토큰의 v-prediction L1 (cfm_loss). AdamW 1e-4, batch 160, 20k step ≈ 40분(GPU 1장, 렌더 병목).
val: held-out 폰트 고정 배치의 val loss + sample(steps=10) 으로 채운 생성 토큰 bbox L1(px).

★ step 0 에 첫 norm_batches 배치로 model.set_norm — 안 하면 SNR 붕괴(모듈 docstring). 통계는 ckpt 안에 있어
  resume 땐 건너뛴다. ckpt = checkpoint_step_%06d.pth + checkpoint_last.pth (train/core.py 규약, optimizer 포함).

예:
  ./train.sh inkspire-layout                                   # configs/inkspire_layout.yaml
  ./train.sh inkspire-layout --out finetune_runs/inkspire_layout_smoke --max-steps 2 --batch-size 4 \\
      --num-workers 0 --norm-batches 1 --val-every 1 --val-batches 1 --save-every 1
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import datetime
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from configs import loader as _config                                   # noqa: E402
from custom_datasets.korean.page import KoreanLayoutDataset, layout_collate, vocab   # noqa: E402
from models.inkspire_layout import LayoutCFM, cfm_loss, load_layout, sample, save_ckpt   # noqa: E402
from train.core import PATH_ARGS, data_paths_of, log_run_config, require_optim_state   # noqa: E402


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="finetune_runs/inkspire_layout")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None, help="checkpoint_*.pth (model+optimizer+step+정규화 통계)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--norm-batches", type=int, default=20, help="step 0 정규화 통계용 배치 수(★ 필수)")
    p.add_argument("--batch-size", type=int, default=160)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--clip-grad", type=float, default=1.0, help="0 = 해제")
    p.add_argument("--ema-decay", type=float, default=0.999, help="0 = 끔. checkpoint_ema_* 로 따로 저장")
    for k in PATH_ARGS:   # 입력 자산 4키 (configs paths: 섹션). korean_fonts_dir = 페이지 렌더 폰트 풀
        p.add_argument(f"--{k.replace('_', '-')}", default=None)
    p.add_argument("--P", type=int, default=512, help="페이지 줄 수 결정(ceil(P/pitch)+U(0,3))")
    p.add_argument("--page-w", type=int, default=1024)
    p.add_argument("--exclude-fonts", nargs="*", default=None)
    p.add_argument("--val-every", type=int, default=1000, help="0=off")
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--val-seed", type=int, default=1234)
    p.add_argument("--val-fonts-dir", default=None, help="held-out 폰트(기본 assets/fonts_korean_v2/test)")
    p.add_argument("--allow-fresh-optim", action="store_true", help="resume 시 optimizer 복원 실패 허용")
    p.add_argument("--config", default=str(HERE / "configs/inkspire_layout.yaml"))
    return p


def make_ds(args, seed, length, fonts_dir=None):
    return KoreanLayoutDataset(P=args.P, page_w=args.page_w, mode="rand", length=length, seed=seed,
                               fonts_dir=fonts_dir or args.korean_fonts_dir, sampler_cfg=args.sampler,
                               paths=data_paths_of(args), exclude_fonts=args.exclude_fonts)


@torch.no_grad()
def validate(model, batches, seed, page_w):
    """(val loss, 생성 토큰 bbox L1 px). 고정 rng → step 간 같은 마스크/노이즈."""
    model.eval()
    losses, l1s = [], []
    for b in batches:
        losses.append(cfm_loss(model, b, torch.Generator().manual_seed(seed)).item())
        out = sample(model, b["char_ids"], b["line_id"], b["layout"], b["ref_mask"], b["pad_mask"],
                     steps=10, seed=seed).cpu()
        gen = ~b["ref_mask"] & ~b["pad_mask"]
        l1s.append(((out - b["layout"]).abs() * gen[..., None]).sum().item() / max(1, gen.sum().item() * 4))
    model.train()
    return sum(losses) / len(losses), sum(l1s) / len(l1s) * page_w


def main():
    args = _config.parse_args(make_parser(), default_config=str(HERE / "configs/inkspire_layout.yaml"))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log_run_config(out, args)

    # ── 모델 / 옵티마이저 / resume ──
    if args.resume:
        model, ck = load_layout(args.resume, device)
        start_step = int(ck["step"])
        print(f"resume {args.resume} step={start_step} kw={model.kw}")
    else:
        model = LayoutCFM(len(vocab()), d=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                          max_len=args.max_len).to(device)
        ck, start_step = None, 0
    model.train()
    print(f"LayoutCFM params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  vocab {len(vocab())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if ck is not None:
        require_optim_state(ck, opt, args.allow_fresh_optim, args.resume)
        for g in opt.param_groups:
            g["lr"] = args.lr

    # ── 데이터 (idx 결정적: seed+start_step 으로 resume 시 새 샘플) ──
    ds = make_ds(args, args.seed + start_step, (args.max_steps - start_step) * args.batch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=layout_collate, drop_last=True, persistent_workers=args.num_workers > 0)
    print(f"train fonts {len(ds.fonts)}  ({args.korean_fonts_dir})")
    val_batches = []
    if args.val_every > 0:
        vds = make_ds(args, args.val_seed, args.val_batches * args.batch_size,
                      fonts_dir=args.val_fonts_dir or str(HERE / "assets/fonts_korean_v2/test"))
        val_batches = list(DataLoader(vds, batch_size=args.batch_size, collate_fn=layout_collate,
                                      num_workers=args.num_workers))
        print(f"val fonts {len(vds.fonts)}  batches {len(val_batches)}")

    it = iter(loader)
    pending = []
    if ck is None:   # ★ 정규화 통계 — 첫 norm_batches 배치(학습에도 그대로 씀)
        pending = [next(it) for _ in range(args.norm_batches)]
        v = torch.cat([b["layout"][~b["pad_mask"]] for b in pending])
        model.set_norm(v[None], torch.zeros(1, len(v), dtype=torch.bool))
        print(f"set_norm on {len(v)} tokens: mean {model.mean.tolist()} std {model.std.tolist()}")

    # ponytail: AveragedModel 대신 3줄 EMA. 논문엔 없지만 4-d 회귀라 마지막 step 편차가 그대로 남는다.
    # ★ set_norm 뒤에 떠야 한다 — 먼저 뜨면 fresh run 의 EMA 가 mean 0/std 1 로 굳어 denormalize 가 항등이 된다.
    ema = copy.deepcopy(model).requires_grad_(False) if args.ema_decay > 0 else None

    tl = (out / "train_loss.csv").open("a", newline="", encoding="utf-8"); tw = csv.writer(tl)
    if tl.tell() == 0:
        tw.writerow(["step", "loss", "it_s", "timestamp"])
    vl = (out / "val_loss.csv").open("a", newline="", encoding="utf-8"); vw = csv.writer(vl)
    if vl.tell() == 0:
        vw.writerow(["step", "val_loss", "val_l1_px"])

    def save(step):
        for name in (f"checkpoint_step_{step:06d}.pth", "checkpoint_last.pth"):
            save_ckpt(model, opt, step, out / name)
        if ema is not None:
            for name in (f"checkpoint_ema_step_{step:06d}.pth", "checkpoint_ema_last.pth"):
                save_ckpt(ema, opt, step, out / name)
        print(f"  saved {out / f'checkpoint_step_{step:06d}.pth'} (+ checkpoint_last.pth)")

    # ponytail: rng 상태는 ckpt 에 안 넣는다(save_ckpt 규약 고정). 데이터는 idx 결정적이고
    #   마스크/t/z 는 seed+start_step 로 재시드되므로 resume 도 결정적이다(중단 없이 돌린 것과는 다름).
    rng = torch.Generator().manual_seed(args.seed + start_step)
    step, run_loss, run_n, t0 = start_step, 0.0, 0, time.time()
    print(f"training {start_step} -> {args.max_steps}  (batch {args.batch_size})")
    for b in itertools.chain(pending, it):   # 길이 = max_steps·batch, drop_last → 정확히 max_steps 에서 끝난다
        if step >= args.max_steps:
            break
        loss = cfm_loss(model, b, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm.detach(), 1 - args.ema_decay)
        step += 1
        run_loss += loss.item(); run_n += 1
        if step % args.log_every == 0 or step == args.max_steps:
            its = run_n / (time.time() - t0)
            print(f"step {step}/{args.max_steps}  loss={run_loss / run_n:.4f}  {its:.2f} it/s")
            tw.writerow([step, f"{run_loss / run_n:.6f}", f"{its:.3f}",
                         datetime.datetime.now().isoformat(timespec="seconds")]); tl.flush()
            run_loss, run_n, t0 = 0.0, 0, time.time()
        if val_batches and step % args.val_every == 0:
            vloss, vl1 = validate(model, val_batches, args.val_seed, args.page_w)
            row = [step, f"{vloss:.6f}", f"{vl1:.3f}"]
            msg = f"  [val@{step}] loss={vloss:.4f}  gen bbox L1={vl1:.2f}px"
            if ema is not None:
                eloss, el1 = validate(ema, val_batches, args.val_seed, args.page_w)
                row += [f"{eloss:.6f}", f"{el1:.3f}"]
                msg += f"  | EMA loss={eloss:.4f} L1={el1:.2f}px"
            print(msg)
            vw.writerow(row); vl.flush()
        if args.save_every > 0 and step % args.save_every == 0:
            save(step)
    if args.save_every <= 0 or step % args.save_every != 0:
        save(step)
    tl.close(); vl.close()
    print("done.")


if __name__ == "__main__":
    main()
