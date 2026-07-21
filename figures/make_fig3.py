"""
Figure 3 (main paper): 3-row qualitative comparison.
Cross-domain: 2 weather rows + 1 art row, all in one PNG.
4 columns: Input | Simple | LLM-only | SK+LLM (ours).
KG+filter excluded.

Usage:
  python make_fig3_final.py
    --weather-hashes h1,h2 --art-ids id1
    [--seed 42] [--cell-size 384] [--out figure3_final.png]
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from config import ROOT as BASE, FLUX_OUT  # noqa: E402

CFG = {
    "weather": {
        "results_dir": BASE / "results_emu_edit_v6",
        "data_path":   BASE / "emu_edit_weather_final.json",
        "key": "hash", "input_field": "image_path",
        "label_field": "kg_condition", "raw_inst": "instruction",
    },
    "art": {
        "results_dir": BASE / "results_art_style_v3",
        "data_path":   BASE / "art_style_test_dataset_v3.json",
        "key": "id", "input_field": "input_image",
        "label_field": "art_style", "raw_inst": "instruction",
    },
}

KEYS = {
    "Simple"        : {"dir": "simple_clip_dir",      "out": "simple_clip_out"},
    "LLM-only"      : {"dir": "llm_clip_dir",         "out": "llm_clip_out"},
    "SK+LLM (ours)" : {"dir": "kg_nofilter_clip_dir", "out": "kg_nofilter_clip_out"},
}
IMG_DIR = {
    "Simple"        : "simple_images",
    "LLM-only"      : "llm_images",
    "SK+LLM (ours)" : "kg_nofilter_images",
}

def load(domain):
    cfg = CFG[domain]
    with open(cfg["results_dir"] / "results.json") as f:
        results = json.load(f)
    with open(cfg["data_path"]) as f:
        data = json.load(f)
    samples = data["samples"] if domain == "weather" else data
    return results, {s[cfg["key"]]: s for s in samples}, cfg

def find_record(results, key_field, kv, seed):
    for r in results:
        if r[key_field] == kv and r["seed"] == seed:
            return r
    return None

def get_font(size, bold=False):
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for c in cands:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

def load_img(p, size):
    return Image.open(str(p)).convert("RGB").resize(size, Image.LANCZOS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather-hashes", default="",
                    help="comma-separated weather hashes (in row order)")
    ap.add_argument("--art-ids", default="",
                    help="comma-separated art ids (in row order)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell-size", type=int, default=384)
    ap.add_argument("--out", default="figure3_final.png")
    args = ap.parse_args()

    weather_hs = [h.strip() for h in args.weather_hashes.split(",") if h.strip()]
    art_ids    = [i.strip() for i in args.art_ids.split(",")        if i.strip()]

    method_order = ["Simple", "LLM-only", "SK+LLM (ours)"]
    cols = ["Input"] + method_order
    cs   = args.cell_size
    gap  = 8
    header_h = 30
    cap_h    = 38

    n_cols = len(cols)
    W = n_cols * cs + (n_cols - 1) * gap

    scale    = cs / 384.0          # 기존 기본 셀 크기 기준으로 폰트 비례 확대
    header_h = int(round(header_h * scale))
    cap_h    = int(round(cap_h * scale))
    gap      = int(round(gap * scale))
    W        = n_cols * cs + (n_cols - 1) * gap
    font_hdr = get_font(max(16, int(round(16 * scale))), bold=True)
    font_cap = get_font(max(12, int(round(12 * scale))), bold=False)

    # 모든 row 정보 먼저 모음
    all_rows = []
    for hs in weather_hs:
        results, lookup, cfg = load("weather")
        r = find_record(results, cfg["key"], hs, args.seed)
        if r is None: print(f"  ⚠ weather {hs[:20]} not found"); continue
        all_rows.append((r, lookup, cfg, "weather"))
    for sid in art_ids:
        results, lookup, cfg = load("art")
        r = find_record(results, cfg["key"], sid, args.seed)
        if r is None: print(f"  ⚠ art {sid[:20]} not found"); continue
        all_rows.append((r, lookup, cfg, "art"))

    n_rows = len(all_rows)
    if n_rows == 0:
        print("no rows found"); return

    H = header_h + n_rows * (cs + cap_h) + (n_rows - 1) * gap
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    # Header
    for i, name in enumerate(cols):
        x  = i * (cs + gap)
        bb = draw.textbbox((0, 0), name, font=font_hdr)
        tw = bb[2] - bb[0]
        draw.text((x + (cs - tw)/2, int(4*scale)), name, fill="black", font=font_hdr)

    y = header_h
    for ri, (r, lookup, cfg, domain) in enumerate(all_rows):
        kv     = r[cfg["key"]]
        sample = lookup[kv]
        cond   = sample.get(cfg["label_field"], "?")
        raw    = sample.get(cfg["raw_inst"], "")

        # Input
        try:
            inp = load_img(BASE / sample[cfg["input_field"]], (cs, cs))
        except Exception as e:
            print(f"  ⚠ input load failed {kv[:20]}: {e}")
            inp = Image.new("RGB", (cs, cs), "lightgray")
        canvas.paste(inp, (0, y))

        # Method imgs
        suf = f"{kv[:40]}_s{r['seed']}.png"
        for i, m in enumerate(method_order, start=1):
            ip = cfg["results_dir"] / IMG_DIR[m] / suf
            if ip.exists():
                img = load_img(ip, (cs, cs))
            else:
                print(f"  ⚠ {ip.name} missing"); img = Image.new("RGB",(cs,cs),"lightgray")
            canvas.paste(img, (i * (cs + gap), y))
        y += cs

        # Caption
        ds = "Weather" if domain == "weather" else "Art"
        s_dir   = r[KEYS["Simple"]["dir"]]
        l_dir   = r[KEYS["LLM-only"]["dir"]]
        o_dir   = r[KEYS["SK+LLM (ours)"]["dir"]]
        s_out   = r[KEYS["Simple"]["out"]]
        l_out   = r[KEYS["LLM-only"]["out"]]
        o_out   = r[KEYS["SK+LLM (ours)"]["out"]]
        cap = (f"({chr(ord('a')+ri)}) {ds} \u2013 {cond}  |  "
               f"user request: \u201C{raw}\u201D")
        cap2 = (f"      CLIPdir:  Simple {s_dir:.3f} / LLM-only {l_dir:.3f} / "
                f"Ours {o_dir:.3f}     "
                f"CLIPout:  Simple {s_out:.3f} / LLM-only {l_out:.3f} / "
                f"Ours {o_out:.3f}")
        lh = int(round(16 * scale))
        draw.text((int(6*scale), y + int(2*scale)),  cap,  fill="#333333", font=font_cap)
        draw.text((int(6*scale), y + int(2*scale) + lh), cap2, fill="#333333", font=font_cap)
        y += cap_h
        if ri < n_rows - 1:
            y += gap

    canvas.save(args.out)
    print(f"\n\u2713 saved: {args.out}  ({canvas.width}x{canvas.height})  rows={n_rows}")


if __name__ == "__main__":
    main()