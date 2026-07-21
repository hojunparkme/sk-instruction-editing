# ── Weather 결과에 art style과 동일한 5개 metric 추가 계산 ────────────
# - 기존 results.json의 이미지 경로를 읽어서 metric 재계산
# - FLUX 재실행 없음 (저장된 이미지 그대로 사용)
# - art style과 같은 형식: clip_dir, clip_im, clip_out, l1, dino
# - 결과를 results.json에 추가 필드로 저장

import os, json, torch
os.environ["CUDA_VISIBLE_DEVICES"]    = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoModel, AutoImageProcessor,
)

from config import (  # noqa: E402
    ROOT as BASE, EMU_PATH, REPOSITORY_PATH, FLUX_SAMPLE_IDS,
    LLAVA_PATH, FLUX_PATH, IP2P_PATH, MGIE_CODE, MGIE_LLAVA, MGIE_CKPT,
    FLUX_OUT, IP2P_OUT, DEVICE,
)
RESULTS_DIR = FLUX_OUT
RESULTS_PATH = RESULTS_DIR / "results.json"

# ── 1. 데이터 로드 ────────────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    all_results = json.load(f)
with open(EMU_PATH, encoding="utf-8") as f:
    emu_data = json.load(f)
sample_lookup = {s["hash"]: s for s in emu_data["samples"]}

# 이미 5-metric 다 계산됐는지 확인
already_done = sum(1 for r in all_results if "kg_clip_dir" in r)
print(f"총 records: {len(all_results)}")
print(f"이미 5-metric 계산됨: {already_done}")
print(f"계산 필요: {len(all_results) - already_done}")

# ── 2. Weather용 condition caption (art style의 STYLE_CAPTIONS와 같은 역할) ─
COND_CAPTIONS = {
    "rainy" : "a rainy scene with rain falling, dark storm clouds, wet surfaces, and puddles",
    "snowy" : "a snowy scene with heavy snowfall, snow accumulation, and a cold winter atmosphere",
    "foggy" : "a foggy scene with dense fog, low visibility, and a misty hazy atmosphere",
    "night" : "a nighttime scene with darkness, artificial lights, and deep shadows",
    "clear" : "a clear sunny day with a bright blue sky, sunlight, and high visibility",
}

# ── 3. CLIP, DINO 로드 ───────────────────────────────────────────────────
print("\n모델 로딩...")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14").to("cuda").eval()
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("✓ CLIP 로드")

dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()
dino_proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
print("✓ DINOv2 로드")

# ── 4. Metric 함수들 (art style 코드와 동일) ──────────────────────────────
@torch.no_grad()
def clip_dir_score(orig_img, edited_img, instruction):
    inputs = clip_proc(
        images=[orig_img, edited_img], text=[instruction],
        return_tensors="pt", padding=True,
        truncation=True, max_length=77,
    ).to("cuda")
    out = clip_model(**inputs)
    img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_e = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    delta_img = img_e[1] - img_e[0]
    delta_img = delta_img / delta_img.norm()
    return float((delta_img * txt_e[0]).sum())

@torch.no_grad()
def clip_im_score(orig_img, edited_img):
    inputs = clip_proc(images=[orig_img, edited_img], return_tensors="pt").to("cuda")
    out = clip_model.get_image_features(**inputs)
    e = out / out.norm(dim=-1, keepdim=True)
    return float((e[0] * e[1]).sum())

@torch.no_grad()
def clip_out_score(edited_img, target_text):
    inputs = clip_proc(images=edited_img, text=[target_text],
                       return_tensors="pt", padding=True,
                       truncation=True, max_length=77).to("cuda")
    out = clip_model(**inputs)
    img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_e = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img_e * txt_e).sum())

def l1_score(orig_img, edited_img, size=512):
    a = np.array(orig_img.resize((size, size))).astype(float)
    b = np.array(edited_img.resize((size, size))).astype(float)
    return float(np.mean(np.abs(a - b)) / 255.0)

@torch.no_grad()
def dino_score(orig_img, edited_img):
    inputs = dino_proc(images=[orig_img, edited_img], return_tensors="pt").to("cuda")
    out    = dino_model(**inputs)
    cls    = out.last_hidden_state[:, 0]
    cls    = cls / cls.norm(dim=-1, keepdim=True)
    return float((cls[0] * cls[1]).sum())

def compute_metrics(orig_img, edited_img, instruction, target_text):
    return {
        "clip_dir": round(clip_dir_score(orig_img, edited_img, instruction), 4),
        "clip_im" : round(clip_im_score(orig_img, edited_img), 4),
        "clip_out": round(clip_out_score(edited_img, target_text), 4),
        "l1"      : round(l1_score(orig_img, edited_img), 4),
        "dino"    : round(dino_score(orig_img, edited_img), 4),
    }

# ── 5. 백업 ───────────────────────────────────────────────────────────────
backup_path = RESULTS_PATH.with_suffix(".before_5metric.json")
if not backup_path.exists():
    with open(backup_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ 백업 저장: {backup_path}")

# ── 6. 메트릭 계산 루프 ──────────────────────────────────────────────────
todo = [r for r in all_results if "kg_clip_dir" not in r]
print(f"\n계산 시작: {len(todo)}개 records")

for i, r in enumerate(todo):
    h    = r["hash"]
    cond = r["kg_condition"]
    s    = sample_lookup.get(h)
    if not s:
        print(f"  ⚠ {h[:20]} sample 못 찾음 - 스킵")
        continue

    orig            = Image.open(str(BASE / s["image_path"])).convert("RGB")
    kg_img          = Image.open(r["kg_image_path"]).convert("RGB")
    kg_nofilter_img = Image.open(r["kg_nofilter_image_path"]).convert("RGB")
    llm_img         = Image.open(r["llm_image_path"]).convert("RGB")
    simple_img      = Image.open(r["simple_image_path"]).convert("RGB")

    target_text = COND_CAPTIONS.get(cond, f"a {cond} scene")

    kg_m   = compute_metrics(orig, kg_img,          r["kg_instruction"],          target_text)
    kgn_m  = compute_metrics(orig, kg_nofilter_img, r["kg_nofilter_instruction"], target_text)
    llm_m  = compute_metrics(orig, llm_img,         r["llm_instruction"],         target_text)
    simp_m = compute_metrics(orig, simple_img,      r["simple_prompt"],           target_text)

    for key, m in [("kg", kg_m), ("kg_nofilter", kgn_m),
                   ("llm", llm_m), ("simple", simp_m)]:
        for met, val in m.items():
            r[f"{key}_{met}"] = val

    print(f"  [{i+1:03d}/{len(todo)}] {cond:<6} s={r['seed']:<4} h={h[:16]} "
          f"KG_dir={kg_m['clip_dir']:.4f}  KGn_dir={kgn_m['clip_dir']:.4f}  "
          f"LLM_dir={llm_m['clip_dir']:.4f}")

    # 매번 저장 (안전)
    if (i+1) % 20 == 0 or i == len(todo)-1:
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# ── 7. 요약 ──────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("Weather 5-metric 결과 (seed별 / condition별)")
print(f"{'='*80}")

metrics_list = ["clip_dir", "clip_im", "clip_out", "l1", "dino"]
SEEDS = sorted(set(r["seed"] for r in all_results))

for seed in SEEDS:
    rows_s = [r for r in all_results if r["seed"] == seed]
    print(f"\n[seed={seed}]  N={len(rows_s)}")
    for m in metrics_list:
        kg   = np.mean([r[f"kg_{m}"]          for r in rows_s])
        kgn  = np.mean([r[f"kg_nofilter_{m}"] for r in rows_s])
        llm  = np.mean([r[f"llm_{m}"]         for r in rows_s])
        simp = np.mean([r[f"simple_{m}"]      for r in rows_s])
        arrow = "↑" if m != "l1" else "↓"
        print(f"  {m:<10}{arrow}  KG={kg:.4f}  KGn={kgn:.4f}  "
              f"LLM={llm:.4f}  Simp={simp:.4f}")

print(f"\n[ALL SEEDS — mean ± std (over seeds)]")
by_seed = {}
for r in all_results:
    by_seed.setdefault(r["seed"], []).append(r)

for m in metrics_list:
    arrow = "↑" if m != "l1" else "↓"
    for col, label in [(f"kg_{m}","KG "),  (f"kg_nofilter_{m}","KGn"),
                       (f"llm_{m}","LLM"), (f"simple_{m}",     "Simp")]:
        seed_means = [np.mean([r[col] for r in by_seed[s]]) for s in by_seed]
        if not seed_means: continue
        mn, sd = np.mean(seed_means), np.std(seed_means)
        print(f"  {m:<10}{arrow}  {label}: {mn:.4f} ± {sd:.4f}")
    print()

print(f"\n✅ 완료: results.json에 5-metric 추가됨")
