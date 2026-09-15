"""트레이너 공용 헬퍼 (Eruku_korean_finetuning 에서 잘라옴).

원본은 Eruku Phase1/2 의 인자 정의와 학습 루프(`make_parser`, `train`, Emuru 모델)를 담고
있었다. InkSpire 트레이너 둘이 쓰는 것은 아래 6개뿐이라 그것만 남긴다 — 그래야
`models/eruku.py` 와 `custom_datasets/korean/handb.py` 의존이 사라진다.
"""
from __future__ import annotations

import datetime
import random
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))

from custom_datasets.korean import fontset as G   # log_run_config 의 sampler_config


#: config/CLI 인자 이름 → korean/split.py DEFAULT_PATHS 키
PATH_ARGS = ("korean_fonts_dir", "corpus_korean", "corpus_english")


def data_paths_of(args) -> dict:
    """args 에서 데이터 경로 override 만 추려 dict 로. 미지정(None)은 기본값 유지."""
    return {k: getattr(args, k, None) for k in PATH_ARGS if getattr(args, k, None)}


def rng_state() -> dict:
    """재개가 '중단 없이 돌린 것'과 같아지려면 Adam 모멘트만으론 부족하다 — RNG 도 같이 옮긴다.
    (dropout·증강·DataLoader 워커 시드가 전부 여기서 갈린다.)"""
    import numpy as _np
    return {"python": random.getstate(), "numpy": _np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def load_rng_state(st: dict | None) -> bool:
    """rng_state() 결과를 되돌린다. 복원했으면 True."""
    if not st:
        return False
    import numpy as _np
    random.setstate(st["python"])
    _np.random.set_state(st["numpy"])
    torch.set_rng_state(torch.as_tensor(st["torch"], dtype=torch.uint8).cpu())
    if st.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(x, dtype=torch.uint8).cpu()
                                      for x in st["cuda"]])
    return True


def require_optim_state(ck: dict, optimizer, allow_fresh: bool, what: str):
    """resume 시 Adam 모멘트를 복원한다. 없거나 실패하면 **기본적으로 중단**한다.

    조용히 새 옵티마이저로 시작하면 모멘트가 0에서 다시 쌓이면서 좋은 지점에서 밀려난다
    (docs/EXPERIMENTS.md §4 의 '연장하면 퇴보' 가 이것이다). 의도한 경우만 --allow-fresh-optim."""
    if "optimizer" not in ck:
        msg = (f"{what} 에 optimizer state 가 없습니다 — 이어서 돌리면 Adam 모멘트가 초기화돼 "
               f"중단 없이 돌린 것과 달라집니다. optimizer 를 포함한 checkpoint_step_*.pth 를 쓰거나, "
               f"의도한 것이면 --allow-fresh-optim 을 주세요.")
        if not allow_fresh:
            raise SystemExit("resume 중단: " + msg)
        print("  [warn] " + msg)
        return False
    try:
        optimizer.load_state_dict(ck["optimizer"])
    except Exception as e:
        if not allow_fresh:
            raise SystemExit(f"resume 중단: optimizer state 복원 실패 ({type(e).__name__}: {e}). "
                             f"의도한 것이면 --allow-fresh-optim.")
        print(f"  [warn] optimizer restore 실패(무시): {e}")
        return False
    print("  optimizer state restored (Adam 모멘트 포함)")
    return True


def log_run_config(out_dir, args):
    """run config 기록: out_dir/train_config.yml 에 run 히스토리 누적 (train/inkspire*.py 도 사용).
    CLI 덮어쓰기까지 반영된 최종 값 + 실제 적용된 샘플러 설정 전체를 남긴다."""
    cfg_path = Path(out_dir) / "train_config.yml"
    runs = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or []) if cfg_path.exists() else []
    cur = {k: (list(v) if isinstance(v, tuple) else v)
           for k, v in vars(args).items() if k != "sampler"}
    cur["sampler"] = G.sampler_config(getattr(args, "sampler", None))   # 실제 적용된 샘플러 설정 전체
    if getattr(args, "resume", None) and runs:    # resume 인데 파라미터가 달라지면 경고
        prev = runs[-1]["args"]
        for k in cur:
            if k not in ("resume", "max_steps", "extra_steps") and prev.get(k) != cur[k]:
                print(f"[config WARN] resume 인데 '{k}' 가 이전 run 과 다름: {prev.get(k)!r} -> {cur[k]!r}")
    runs.append({"run": len(runs) + 1,
                 "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
                 "torch": str(torch.__version__),
                 "args": cur})
    cfg_path.write_text(yaml.safe_dump(runs, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"config logged: {cfg_path} (run #{len(runs)})")
