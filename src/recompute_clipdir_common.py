# ─────────────────────────────────────────────────────────────────────────────
# 공통 reference CLIPdir 재계산 (self-referential 문제 해결)
#
# 문제: 기존 clip_dir는 각 방법이 "자기 자신이 생성한 instruction"을 text
#       direction으로 사용 → 상세한 instruction을 만드는 KG+LLM이 구조적으로 유리.
#
# 해결: 모든 방법에 "동일한" text direction을 적용.
#   - Weather: T_src = input_caption, T_tgt = output_caption  (Emu Edit sample별 제공)
#   - Art    : T_src = "a photo",     T_tgt = "an image in {style} style"  (style별 고정)
#
#   CLIPdir_common = cos( E_I(edited) - E_I(input),  E_T(T_tgt) - E_T(T_src) )
#
# edited/input 이미지는 그대로 사용 (재생성 없음). text direction만 공통으로 교체.
# 기존 필드는 보존하고 "*_clip_dir_common" 필드를 새로 추가.
#
# 실행:
#   python recompute_clipdir_common.py weather
#   python recompute_clipdir_common.py art
# ─────────────────────────────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import os, sys, json, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from config import ROOT as BASE, FLUX_OUT, EMU_PATH, DEVICE  # noqa: E402

CFG = {
    "weather": {
        "results": FLUX_OUT / "results.json",
        "emu":     EMU_PATH,
        "methods": ["simple", "llm", "kg", "kg_nofilter"],
        "img_field": {  # 각 방법의 edited 이미지 경로 필드
            "simple": "simple_image_path", "llm": "llm_image_path",
            "kg": "kg_image_path", "kg_nofilter": "kg_nofilter_image_path",
        },
        "key": "hash",
    },
    "art": {
        "results": BASE / "results_art_style_v3" / "results.json",
        "data":    BASE / "art_style_test_dataset_v3.json",
        "methods": ["simple", "llm", "kg", "kg_nofilter"],
        "img_field": {
            "simple": "simple_image_path", "llm": "llm_image_path",
            "kg": "kg_image_path", "kg_nofilter": "kg_nofilter_image_path",
        },
        "key": "id",
    },
}

# Art style 공통 source/target caption (style별 고정, 방법 무관)
# 두 가지 버전을 모두 계산해서 비교 재료로 제공:
#   - simple  : "a photo" → "an image in {style} style"  (간결)
#   - detailed: "a photo" → STYLE_CAPTIONS[style]         (CLIPout과 동일한 상세 caption)
def art_dir_captions_simple(style):
    label = style.replace("_", " ")
    return "a photo", f"an image in {label} style"

# CLIPout에서 쓰는 것과 동일한 상세 style caption
STYLE_CAPTIONS = {
    "oil_painting":     "an oil painting with visible brushstrokes, rich texture, and layered colors",
    "watercolor":       "a watercolor painting with soft transparent washes and bleeding colors",
    "pencil_drawing":   "a pencil drawing with grayscale tones, hatching lines, and visible pencil marks",
    "charcoal_drawing": "a charcoal drawing with deep black tones, smudged textures, and expressive marks",
    "comic_book":       "a comic book illustration with bold outlines, halftone dots, and flat colors",
    "pop_art":          "a pop art image with bold flat primary colors and halftone patterns",
    "cubism":           "a cubist painting with fragmented geometric planes and multiple viewpoints",
    "renaissance":      "a renaissance painting with realistic proportions, sfumato shading, and classical composition",
    "abstract":         "an abstract painting with non-representational forms, bold color fields, and gestural marks",
}
def art_dir_captions_detailed(style):
    label = style.replace("_", " ")
    return "a photo", STYLE_CAPTIONS.get(style, f"an image in {label} style")

def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "weather"
    cfg = CFG[domain]

    with open(cfg["results"]) as f:
        results = json.load(f)

    # weather: emu에서 input/output caption 얻기 (이미 results에 있으면 그대로 사용)
    emu_lookup = {}
    if domain == "weather":
        with open(cfg["emu"], encoding="utf-8") as f:
            emu = json.load(f)
        emu_lookup = {s["hash"]: s for s in emu["samples"]}

    print("CLIP 로딩...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print("✓ CLIP 로드\n")

    @torch.no_grad()
    def img_embed(img):
        inp = proc(images=img, return_tensors="pt").to("cuda")
        e = clip.get_image_features(**inp)
        return (e / e.norm(dim=-1, keepdim=True))[0]

    @torch.no_grad()
    def txt_embed(text):
        inp = proc(text=[text], return_tensors="pt", padding=True,
                   truncation=True, max_length=77).to("cuda")
        e = clip.get_text_features(**inp)
        return (e / e.norm(dim=-1, keepdim=True))[0]

    # text direction 캐시 (weather는 sample별 1버전, art는 style별 2버전)
    txt_dir_cache = {}

    def get_text_dirs(rec):
        """(variant_name, direction_embedding) 리스트 반환.
        weather는 [("", dir)] 하나, art는 [("", simple_dir), ("_detailed", detailed_dir)]."""
        if domain == "weather":
            h = rec["hash"]
            src = rec.get("input_caption")  or emu_lookup.get(h, {}).get("input_caption", "")
            tgt = rec.get("output_caption") or emu_lookup.get(h, {}).get("output_caption", "")
            ckey = ("w", h)
            if ckey not in txt_dir_cache:
                d = txt_embed(tgt) - txt_embed(src); d = d / d.norm()
                txt_dir_cache[ckey] = [("", d)]
            return txt_dir_cache[ckey]
        else:
            style = rec["art_style"]
            ckey = ("a", style)
            if ckey not in txt_dir_cache:
                variants = []
                for suffix, capfn in [("", art_dir_captions_simple),
                                      ("_detailed", art_dir_captions_detailed)]:
                    src, tgt = capfn(style)
                    d = txt_embed(tgt) - txt_embed(src); d = d / d.norm()
                    variants.append((suffix, d))
                txt_dir_cache[ckey] = variants
            return txt_dir_cache[ckey]

    # input 이미지 embedding 캐시 (sample별 1회)
    input_emb_cache = {}

    def get_input_embed(rec):
        k = rec[cfg["key"]]
        if k not in input_emb_cache:
            if domain == "weather":
                s = emu_lookup.get(rec["hash"], {})
                ipath = BASE / s["image_path"]
            else:
                # art: input_image 경로가 results 또는 data에 있음
                ipath = BASE / rec.get("input_image", "")
            input_emb_cache[k] = img_embed(Image.open(str(ipath)).convert("RGB"))
        return input_emb_cache[k]

    # art는 input_image 경로가 results에 없을 수 있어 data에서 보강
    if domain == "art":
        with open(cfg["data"]) as f:
            art_data = json.load(f)
        art_lookup = {s["id"]: s for s in art_data}
        for r in results:
            if "input_image" not in r:
                r["input_image"] = art_lookup.get(r["id"], {}).get("input_image", "")

    # backup
    bpath = cfg["results"].with_suffix(".before_clipdir_common.json")
    if not bpath.exists():
        with open(bpath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 백업: {bpath}\n")

    todo = [r for r in results if f"{cfg['methods'][0]}_clip_dir_common" not in r]
    print(f"재계산 대상: {len(todo)} records\n")

    for i, rec in enumerate(todo):
        try:
            in_e   = get_input_embed(rec)
            t_dirs = get_text_dirs(rec)   # [(suffix, dir), ...]
        except Exception as e:
            print(f"  ⚠ {rec[cfg['key']][:16]} 준비 실패: {e}")
            continue

        for m in cfg["methods"]:
            ipath = rec.get(cfg["img_field"][m])
            if not ipath or not Path(ipath).exists():
                for suffix, _ in t_dirs:
                    rec[f"{m}_clip_dir_common{suffix}"] = None
                continue
            ed_e = img_embed(Image.open(ipath).convert("RGB"))
            d_img = ed_e - in_e
            n = d_img.norm()
            if float(n) < 1e-8:
                for suffix, _ in t_dirs:
                    rec[f"{m}_clip_dir_common{suffix}"] = 0.0
                continue
            d_img = d_img / n
            for suffix, t_dir in t_dirs:
                rec[f"{m}_clip_dir_common{suffix}"] = round(float((d_img * t_dir).sum()), 4)

        if (i + 1) % 20 == 0 or i == len(todo) - 1:
            with open(cfg["results"], "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [{i+1:03d}/{len(todo)}] 저장")

    with open(cfg["results"], "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 요약: 기존 self-ref vs 공통 reference 비교 ──
    print(f"\n{'='*70}")
    print(f"[{domain}] CLIPdir 비교: 기존(self-ref) → 공통(common)")
    print(f"{'='*70}")
    by_seed = {}
    for r in results:
        by_seed.setdefault(r.get("seed", 42), []).append(r)

    # 어떤 common variant들이 있는지 감지
    sample_rec = results[0]
    variants = [""]
    if f"{cfg['methods'][0]}_clip_dir_common_detailed" in sample_rec:
        variants.append("_detailed")

    for suffix in variants:
        vlabel = "common(detailed)" if suffix == "_detailed" else "common(simple)"
        print(f"\n--- text direction: {vlabel} ---")
        for m in cfg["methods"]:
            old_key = f"{m}_clip_dir"
            new_key = f"{m}_clip_dir_common{suffix}"
            old_seed_means, new_seed_means = [], []
            for s in by_seed:
                rows = by_seed[s]
                ov = [r[old_key] for r in rows if r.get(old_key) is not None]
                nv = [r[new_key] for r in rows if r.get(new_key) is not None]
                if ov: old_seed_means.append(np.mean(ov))
                if nv: new_seed_means.append(np.mean(nv))
            om, osd = (np.mean(old_seed_means), np.std(old_seed_means)) if old_seed_means else (0,0)
            nm, nsd = (np.mean(new_seed_means), np.std(new_seed_means)) if new_seed_means else (0,0)
            print(f"  {m:14}  self-ref {om:.4f}±{osd:.4f}   →   {vlabel} {nm:.4f}±{nsd:.4f}")

    print(f"\n✅ 완료: {cfg['results']} 에 *_clip_dir_common 필드 추가됨")

if __name__ == "__main__":
    main()
