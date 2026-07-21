# ─────────────────────────────────────────────────────────────────────────────
# Weather IP2P — common CLIPdir 재계산 (Emu independent caption 기반)
#
# 기존 IP2P results.json 의 clip_dir 는 각 방법이 자기 instruction 을 text
# direction 으로 쓴 self-referential 값. 여기서는 Weather FLUX 와 동일하게
# Emu Edit 의 sample별 input/output caption 으로 공통 direction 을 만들어 재계산.
#
#   CLIPdir_common = cos( E_I(edited) - E_I(input),  E_T(output_cap) - E_T(input_cap) )
#
# 이미지 경로 (run_comparison_ip2p_v4.py 확인):
#   results_comparison_ip2p_v4/{method}_images/{hash[:40]}.png
#   method 폴더: simple_images, mgie_style_images, llm_only_images,
#                kg_llm_images, kg_llm_nofilter_images
#   single seed (42) → seed suffix 없음
#
# input 이미지/caption: emu_edit_weather_final.json (image_path, input_caption, output_caption)
#
# CPU로 돌리려면 아래 .to("cuda") 두 곳을 .to("cpu") 로 바꾸세요.
#
# 실행:  python recompute_ip2p_weather_common.py
# ─────────────────────────────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import os, json, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel



from config import ROOT as BASE, IP2P_OUT, EMU_PATH, DEVICE  # noqa: E402
RES_DIR  = IP2P_OUT
RESULTS  = RES_DIR / "results.json"
EMU      = EMU_PATH

# results.json 의 nested 키 → 이미지 폴더명
METHOD_DIR = {
    "simple":          "simple_images",
    "mgie_style":      "mgie_style_images",
    "llm_only":        "llm_only_images",
    "kg_llm":          "kg_llm_images",
    "kg_llm_nofilter": "kg_llm_nofilter_images",
}

def main():
    with open(RESULTS) as f:
        results = json.load(f)
    with open(EMU, encoding="utf-8") as f:
        emu = json.load(f)
    emu_lookup = {s["hash"]: s for s in emu["samples"]}

    print(f"CLIP 로딩... (device={DEVICE})")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print("✓ CLIP 로드\n")

    @torch.no_grad()
    def img_embed(img):
        inp = proc(images=img, return_tensors="pt").to(DEVICE)
        e = clip.get_image_features(**inp)
        return (e / e.norm(dim=-1, keepdim=True))[0]

    @torch.no_grad()
    def txt_embed(text):
        inp = proc(text=[text], return_tensors="pt", padding=True,
                   truncation=True, max_length=77).to(DEVICE)
        e = clip.get_text_features(**inp)
        return (e / e.norm(dim=-1, keepdim=True))[0]

    # backup
    bpath = RESULTS.with_suffix(".before_ip2p_common.json")
    if not bpath.exists():
        with open(bpath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 백업: {bpath}\n")

    todo = [r for r in results if "simple_clip_dir_common" not in r]
    print(f"재계산 대상: {len(todo)} records\n")

    skipped_same = 0
    input_emb_cache = {}

    for i, rec in enumerate(todo):
        h = rec["hash"]
        s = emu_lookup.get(h)
        if not s:
            print(f"  ⚠ {h[:16]} emu 없음 - 스킵")
            continue

        in_cap, out_cap = s.get("input_caption",""), s.get("output_caption","")
        if not in_cap or not out_cap or in_cap.strip() == out_cap.strip():
            skipped_same += 1
            for m in METHOD_DIR:
                rec[f"{m}_clip_dir_common"] = None
            continue

        t_dir = txt_embed(out_cap) - txt_embed(in_cap)
        t_dir = t_dir / t_dir.norm()

        if h not in input_emb_cache:
            ipath = BASE / s["image_path"]
            input_emb_cache[h] = img_embed(Image.open(str(ipath)).convert("RGB"))
        in_e = input_emb_cache[h]

        tag = h[:40]
        for m, folder in METHOD_DIR.items():
            ipath = RES_DIR / folder / f"{tag}.png"
            if not ipath.exists():
                rec[f"{m}_clip_dir_common"] = None
                continue
            ed_e = img_embed(Image.open(str(ipath)).convert("RGB"))
            d_img = ed_e - in_e
            n = d_img.norm()
            if float(n) < 1e-8:
                rec[f"{m}_clip_dir_common"] = 0.0
                continue
            d_img = d_img / n
            rec[f"{m}_clip_dir_common"] = round(float((d_img * t_dir).sum()), 4)

        if (i+1) % 20 == 0 or i == len(todo)-1:
            with open(RESULTS, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [{i+1:03d}/{len(todo)}] 저장")

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 요약 ──
    print(f"\n{'='*72}")
    print(f"Weather IP2P CLIPdir: self-ref → Emu independent")
    print(f"(동일/빈 caption으로 스킵: {skipped_same})")
    print(f"{'='*72}")

    def agg_old(m):
        vals = [r[m]["clip_dir"] for r in results if m in r and r[m].get("clip_dir") is not None]
        return np.mean(vals) if vals else float('nan')
    def agg_new(m):
        vals = [r[f"{m}_clip_dir_common"] for r in results
                if r.get(f"{m}_clip_dir_common") is not None
                and not (isinstance(r[f"{m}_clip_dir_common"],float) and np.isnan(r[f"{m}_clip_dir_common"]))]
        return (np.mean(vals), len(vals)) if vals else (float('nan'), 0)

    print(f"\n{'method':<16} {'self-ref':>12}   {'Emu-independent':>18}")
    order = []
    for m in METHOD_DIR:
        om = agg_old(m)
        nm, n = agg_new(m)
        order.append((m, nm))
        print(f"{m:<16} {om:>10.4f}     {nm:>12.4f} (N={n})")

    order.sort(key=lambda x: x[1])
    print(f"\nEmu-independent 순위: " + " < ".join(f"{k}({v:.3f})" for k,v in order))
    print(f"\n✅ 완료: results_comparison_ip2p_v4/results.json 에 *_clip_dir_common 추가")

if __name__ == "__main__":
    main()
