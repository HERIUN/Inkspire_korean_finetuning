"""폰트 목록 · charset 필터 · 텍스트 샘플러 (Eruku_korean_finetuning 에서 잘라옴).

원본은 `KoreanSplitFontSquare`(upstream OnlineSplitFontSquare 서브클래스)와 Eruku 학습용
`make_dataset`/`split_collate`/`dump_samples` 를 함께 갖고 있었다. InkSpire 는 페이지를 직접
렌더하므로 그 절반이 필요 없고, 그걸 버려야 `custom_datasets/upstream/font_square/*` 의존이
사라진다. 여기 남은 것은 `custom_datasets/korean/page.py` 가 쓰는 5개뿐이다:
DEFAULT_EXCLUDE_FONTS · data_paths · font_files · ensure_font_charsets · build_samplers.
"""
from __future__ import annotations
import sys, random, json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]   # 저장소 루트
ASSETS = HERE / "assets"
sys.path.insert(0, str(HERE))
from custom_datasets.korean import fontset as G


# ─────────────────── 입력 자산 경로 (configs/*.yaml 의 paths: 섹션) ───────────────────
# 코드 기본값 = 저장소 동봉 자산. config/CLI 로 바꿀 수 있다(다른 코퍼스·폰트로 실험).
DEFAULT_PATHS = {
    "corpus_korean": ASSETS / "corpus/korean_lines.txt",    # 한글 어절 공급
    "corpus_english": ASSETS / "corpus/english_words.txt",  # 영어 단어 사전
    "korean_fonts_dir": ASSETS / "fonts_korean_v2/train",   # 한글 writer 풀 (+ fonts_charsets.json)
    "backgrounds_dir": ASSETS / "backgrounds",              # style 종이 배경 패치
}


def data_paths(overrides: dict | None = None) -> dict:
    """DEFAULT_PATHS + overrides. None/빈 값은 기본값 유지."""
    paths = dict(DEFAULT_PATHS)
    for k, v in (overrides or {}).items():
        if k not in paths:
            raise KeyError(f"알 수 없는 데이터 경로: {k!r} (가능: {sorted(paths)})")
        if v:
            paths[k] = Path(v)
    return paths


#: 학습·평가에서 통째로 빼는 폰트(파일명 stem). 추론의 `--exclude-writers` 와 같은 목록이어야
#: 학습과 추론이 어긋나지 않는다.
#:   UlsanJunggu: '델' 글리프 높이 555px (그 폰트 중앙값 74px 의 7.5배). 이 글자가 한 줄에
#:   들어가면 라인 전체를 64px 로 맞출 때 나머지 글자가 ~8px 로 뭉개진다.
#:   → 글자만 charset 에서 빼는 방법도 있었지만, 그러면 '델' 이 전 폰트 **교집합**에서 빠져
#:     rand 음절 풀이 2,350 → 2,349 로 줄어든다. 음절 커버리지가 이미 최대 현안이라(D1)
#:     폰트를 빼고 음절을 지키는 쪽을 택했다. docs/DATA_LIMITATIONS.md D6.
#: 근거: tools/font_audit.py — 83폰트 × 2,350음절 전수 스캔에서 걸린 폰트는 이것 하나뿐이다.
DEFAULT_EXCLUDE_FONTS = ["UlsanJunggu"]


def _excluded(name, exclude) -> bool:
    """폰트 파일명(또는 stem)이 제외 목록에 걸리는지. '#N' 접미사는 무시(추론과 같은 규약)."""
    stem = Path(name).stem.split("#")[0]
    return stem in set(exclude or ())


def font_files(fonts_dir, exclude=None) -> list:
    """디렉토리 → 제외 폰트를 뺀 폰트 경로 리스트."""
    d = Path(fonts_dir)
    return [p for p in sorted(d.iterdir())
            if p.suffix.lower() in G.FONT_EXTS and not _excluded(p.name, exclude)]


def ensure_font_charsets(fonts_dir) -> Path | None:
    """`fonts_dir/fonts_charsets.json` 이 그 디렉토리의 폰트를 **전부** 담도록 갱신한다.

    이 json 이 없거나 어떤 폰트의 키가 빠져 있으면, upstream `make_renderers` 가 그 폰트에
    `charset=None` 을 넘기고 `Render.render()` 의 글자 필터가 통째로 꺼진다 → 두부(□)가
    조용히 학습 데이터에 섞인다(docs/DATA_LIMITATIONS.md D6). 그래서 데이터셋을 만들기 전에
    빠진 폰트만 cmap 을 읽어 채운다.

    - 이미 최신이면 파일을 건드리지 않는다(빠진 폰트 0개 → 즉시 반환).
    - 삭제된 폰트의 키는 남겨 둔다(무해하고, 폰트를 되돌릴 때 재계산이 없다).
    - fonts_dir 이 폰트 경로 **리스트**면 그 부모 디렉토리를 대상으로 본다.
    반환: 갱신 대상 json 경로 (디렉토리를 못 정하면 None)
    """
    if fonts_dir is None:
        return None
    if isinstance(fonts_dir, (list, tuple)):
        if not fonts_dir:
            return None
        fonts_dir = Path(fonts_dir[0]).parent
    d = Path(fonts_dir)
    if not d.is_dir():
        return None
    fonts = [p for p in sorted(d.iterdir()) if p.suffix.lower() in G.FONT_EXTS]
    if not fonts:
        return None
    jf = d / "fonts_charsets.json"
    cs = json.load(open(jf)) if jf.exists() else {}
    missing = [p for p in fonts if p.name not in cs]
    if not missing:
        return jf
    print(f"fonts_charsets 갱신: {jf} — 누락 {len(missing)}개 폰트 cmap 읽는 중 ...")
    added, failed = 0, []
    for p in missing:
        try:
            cs[p.name] = "".join(sorted(chr(c) for c in G.get_cmap(p)))
            added += 1
        except Exception as e:                      # 못 읽는 폰트는 남겨두고 알린다
            failed.append(f"{p.name}: {e}")
    tmp = jf.with_suffix(".json.tmp")               # 원자적 교체(워커/중단 대비)
    tmp.write_text(json.dumps(cs, ensure_ascii=False, indent=4), encoding="utf-8")
    tmp.replace(jf)
    print(f"  +{added}개 추가 → 총 {len(cs)}개 폰트")
    if failed:
        print(f"  ⚠️ cmap 읽기 실패 {len(failed)}개 (charset 필터 꺼진 채 렌더됨 = 두부 위험): "
              + "; ".join(failed[:3]))
    return jf


def charsets_json_for(fonts_dir, fallback_dir):
    """fonts_dir 안의 fonts_charsets.json. 없으면(폰트 리스트를 받았거나 json 부재) fallback."""
    if fonts_dir is not None and not isinstance(fonts_dir, (list, tuple)):
        p = Path(fonts_dir) / "fonts_charsets.json"
        if p.exists():
            return p
    return Path(fallback_dir) / "fonts_charsets.json"


def font_charset_cps(charsets_json, rule="intersection", exclude=None):
    """폰트 charset json → (covered_cps, 한글음절 리스트).

    rule="intersection": 전 폰트가 그릴 수 있는 글자만. 샘플러가 통과시킨 글자를 렌더가 다시
        지우는 일이 없다(= 조용한 삭제 없음). 한글 경로 기본. docs/DATA_LIMITATIONS.md D8.
    rule="union": 하나라도 그리면 통과. 영어 폰트 177종은 교집합이 63자(A-Za-z0-9+공백)뿐이라
        구두점이 전멸하고 gen_number 도 randint 폴백만 남으므로 union 을 쓴다.
    """
    if rule not in ("intersection", "union"):
        raise ValueError(f"charset_rule 은 intersection|union: {rule!r}")
    uni, inter = set(), None
    for name, s in json.load(open(charsets_json)).items():
        if _excluded(name, exclude):        # 제외 폰트는 교집합/합집합 계산에서도 뺀다
            continue
        f = {ord(c) for c in s}
        uni |= f
        inter = f if inter is None else inter & f
    cps = inter if rule == "intersection" else uni
    # rand 음절 풀은 규칙과 무관하게 항상 교집합 — 어떤 폰트에 배정돼도 두부가 안 뜨게
    syls = sorted(chr(c) for c in (inter or set()) if 0xAC00 <= c <= 0xD7A3)
    return cps, syls


def build_samplers(style_range, gen_range, seed=42, sampler_cfg=None, n_english=None,
                   paths=None, fonts_dir=None, exclude_fonts=None):
    """한글 style/gen sampler 2개.

    sampler_cfg: configs/*.yaml 의 data.sampler (미지정 키는 G.SAMPLER_DEFAULTS)
    paths:       configs/*.yaml 의 paths: (미지정은 DEFAULT_PATHS)
    fonts_dir:   실제로 렌더에 쓸 폰트 디렉토리. 그 안의 fonts_charsets.json 을 쓴다
                 → held-out 폰트로 데이터를 만들면 charset 도 그 폰트 기준이 된다
                 (docs/DATA_LIMITATIONS.md D5). 폰트 리스트를 받았거나 json 이 없으면
                 paths['korean_fonts_dir'] 로 폴백.
    n_english:   하위호환 단축 인자."""
    cfg = G.sampler_config(sampler_cfg)
    pth = data_paths(paths)
    if n_english is not None:
        cfg["n_english"] = n_english
    rng = random.Random(seed)
    pools = G.build_pools(pth["corpus_korean"], str(pth["corpus_english"]), cfg["n_english"], rng)
    csj = charsets_json_for(fonts_dir, pth["korean_fonts_dir"])
    exc = DEFAULT_EXCLUDE_FONTS if exclude_fonts is None else exclude_fonts
    cps, syls = font_charset_cps(csj, cfg["charset_rule"], exclude=exc)
    mk = lambda rng_, lo, hi, mc: G.MixedLineSampler(
        pools["ko"], pools["en"], cps, cfg["weights"], lo, hi, mc,
        cfg["punct_prob"], cfg["special_prob"], rng_, rand_syllables=syls,
        wrap_prob=cfg["wrap_prob"], rand_len=cfg["rand_len"])
    style_s = mk(rng, style_range[0], style_range[1], cfg["max_chars"]["style"])
    gen_s = mk(rng, gen_range[0], gen_range[1], cfg["max_chars"]["gen"])
    print(f"samplers: style {style_range} gen {gen_range} | ko {len(pools['ko'])} en {len(pools['en'])} "
          f"rand_syl {len(syls)} cps {len(cps)} ({cfg['charset_rule']}, {csj.parent.name}"
          f"{', 제외 ' + ','.join(exc) if exc else ''}) | "
          f"w {cfg['weights']} punct {cfg['punct_prob']} wrap {cfg['wrap_prob']} "
          f"special {cfg['special_prob']} max_chars {cfg['max_chars']}")
    return style_s, gen_s
