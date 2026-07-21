# ── v6 파이프라인: Filtering Ablation + Multi-seed 지원 ──────────────────
# 변경점 (v5 대비)
#   1. LLM deterministic (do_sample=False)
#   2. KG no-filter prompt 추가 → 4-way 비교 (Simple / LLM / KG / KG_no_filter)
#   3. edit_image(seed) 가변 → 여러 seed로 돌릴 수 있음
#   4. SEEDS 환경변수로 seed 리스트 제어
#       예) SEEDS=42        → 단일 seed (기존 호환)
#           SEEDS=42,123,777 → multi-seed
#   5. 결과 폴더/파일명에 seed 포함
import os, json, re, torch
os.environ["CUDA_VISIBLE_DEVICES"]    = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    AutoProcessor, LlavaForConditionalGeneration,
    CLIPProcessor, CLIPModel,
)
from diffusers import FluxKontextPipeline

# ── 0. Seed 설정 ──────────────────────────────────────────────────────────
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42").split(",") if x.strip()]
print(f"실험할 seeds: {SEEDS}")

# ── 1. 경로 설정 ──────────────────────────────────────────────────────────
from config import (  # noqa: E402
    ROOT as BASE, EMU_PATH, REPOSITORY_PATH, FLUX_SAMPLE_IDS,
    LLAVA_PATH, FLUX_PATH, IP2P_PATH, MGIE_CODE, MGIE_LLAVA, MGIE_CKPT,
    FLUX_OUT, IP2P_OUT, DEVICE,
)

KG_PATH = REPOSITORY_PATH
with open(FLUX_SAMPLE_IDS, encoding="utf-8") as f:
    _keep = set(json.load(f)["hashes"])
samples = [s for s in samples if s["hash"] in _keep]
print(f"실험 대상: {len(samples)}개")

RESULTS_PATH = OUTPUT_DIR / "results.json"
all_results  = []
if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)
# (hash, seed) 튜플로 done 관리 → seed별 별도 진행
done_keys = {(r["hash"], r["seed"]) for r in all_results}
print(f"이미 완료: {len(done_keys)}개 (hash×seed)")

# ── 3. KG Retriever ───────────────────────────────────────────────────────
SECTIONS   = ["condition", "environment", "season", "time_of_day", "weather"]
SLOT_ORDER = ["global","lighting","surfaces","atmospheric_effects","objects_details"]

def extract_keywords(text):
    text = text.lower()
    out  = {s: [] for s in SECTIONS}
    for s in SECTIONS:
        for k, node in kg.get(s, {}).items():
            triggers = [k.lower()] + [a.lower() for a in node.get("aliases", [])]
            if any(t in text for t in triggers):
                out[s].append(k)
    return out

def retrieve_candidates(keywords, max_total=50):
    candidates, seen = [], set()
    for section, keys in keywords.items():
        for key in keys:
            node = kg.get(section, {}).get(key)
            if not isinstance(node, dict): continue
            for slot in SLOT_ORDER + [s for s in node.get("positives",{})
                                      if s not in SLOT_ORDER]:
                for item in node.get("positives",{}).get(slot,[])[:6]:
                    c = item.strip() if isinstance(item, str) else str(item)
                    if c and c not in seen:
                        candidates.append(c); seen.add(c)
    return candidates[:max_total]

# ── 4. LLaVA 캡셔닝 ───────────────────────────────────────────────────────
caption_cache_path = OUTPUT_DIR / "captions.json"
if caption_cache_path.exists():
    with open(caption_cache_path) as f:
        captions = json.load(f)
    print(f"✓ 캡션 캐시 로드: {len(captions)}개")
else:
    captions = {}

todo_caption = [s for s in samples if s["hash"] not in captions]
print(f"캡셔닝 필요: {len(todo_caption)}개")

if todo_caption:
    lava_proc  = AutoProcessor.from_pretrained(str(LLAVA_PATH))
    lava_model = LlavaForConditionalGeneration.from_pretrained(
        str(LLAVA_PATH), torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
    ).eval()
    print("✓ LLaVA 로드")

    CAPTION_PROMPT = (
        "Describe this image in 3–5 sentences. Include:\n"
        "1. All visible people and their clothing, accessories, and actions\n"
        "2. All visible objects, vehicles, and surfaces\n"
        "3. Lighting, atmosphere, and overall mood\n"
        "Be specific and concrete. Avoid vague language."
    )
    for i, s in enumerate(todo_caption):
        img    = Image.open(str(BASE / s["image_path"])).convert("RGB")
        prompt = f"USER: <image>\n{CAPTION_PROMPT}\nASSISTANT:"
        inputs = lava_proc(images=img, text=prompt, return_tensors="pt").to(
            lava_model.device, torch.float16)
        with torch.no_grad():
            out = lava_model.generate(**inputs, max_new_tokens=200, do_sample=False)
        cap = lava_proc.decode(out[0][inputs["input_ids"].shape[-1]:],
                               skip_special_tokens=True).strip()
        captions[s["hash"]] = cap
        if (i+1) % 10 == 0:
            with open(caption_cache_path, "w") as f:
                json.dump(captions, f, ensure_ascii=False)
        print(f"  cap [{i+1:03d}/{len(todo_caption)}]")
    with open(caption_cache_path, "w") as f:
        json.dump(captions, f, ensure_ascii=False)

    import gc
    del lava_model; gc.collect(); torch.cuda.empty_cache()
    print("✓ LLaVA 해제")

# ── 5. DeepSeek instruction 생성 ──────────────────────────────────────────
inst_cache_path = OUTPUT_DIR / "instructions.json"
if inst_cache_path.exists():
    with open(inst_cache_path) as f:
        instructions = json.load(f)
    print(f"✓ instruction 캐시 로드: {len(instructions)}개")
else:
    instructions = {}

# v6은 kg_nofilter_instruction이 추가로 필요
def needs_inst(h):
    if h not in instructions: return True
    rec = instructions[h]
    return ("kg_nofilter_instruction" not in rec
            or "kg_instruction" not in rec
            or "llm_instruction" not in rec)

todo_inst = [s for s in samples if needs_inst(s["hash"])]
print(f"instruction 생성 필요: {len(todo_inst)}개")

if todo_inst:
    DS_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    quant    = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(DS_MODEL)
    ds_model  = AutoModelForCausalLM.from_pretrained(
        DS_MODEL, device_map="auto", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, quantization_config=quant,
    ).eval()
    print("✓ DeepSeek 로드")

    @torch.no_grad()
    def generate(system_prompt, user_prompt, max_new_tokens=512):
        messages  = [{"role":"system","content":system_prompt},
                     {"role":"user",  "content":user_prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_tensors="pt",
        ).to(next(ds_model.parameters()).device)
        out = ds_model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens,
            do_sample=False,                                 # ← deterministic
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0, input_ids.shape[-1]:],
                                skip_special_tokens=False).strip()
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        text = re.sub(r"<\|.*?\|>|<｜.*?｜>", "", text).strip()
        if "\n\n" in text:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            thinking_starts = (
                "alright","okay","so i","let me","first","i need",
                "i should","the user","looking at","so the user",
                "let's","now,","wait",
            )
            for para in reversed(paragraphs):
                if not para.lower().startswith(thinking_starts):
                    text = para
                    break
        return text.strip('"').strip()

    SYSTEM_PROMPT = (
        "You are an expert at writing image editing instructions for FLUX Kontext.\n"
        "Output ONLY the final instruction text — no analysis, no JSON, no preamble, no reasoning."
    )

    # KG + filtering (기존 v5 방식)
    def build_kg_prompt(scene, request, candidates):
        return (
            f"Image scene description:\n{scene}\n\n"
            f"User editing request:\n{request}\n\n"
            f"KG candidates:\n{', '.join(candidates)}\n\n"
            "Write ONE image editing instruction:\n"
            "- 2–3 sentences maximum.\n"
            "- Only use KG cues that match objects VISIBLE in the scene description above.\n"
            "- Do NOT use cues referencing elements absent from the scene "
            "(e.g. no road/asphalt visible → never use 'wet asphalt', 'tire spray', "
            "'road puddles', 'lane markings').\n"
            "- Do NOT include reasoning — just the instruction.\n"
            "Return only the instruction text."
        )

    # KG without filtering (ablation)
    def build_kg_nofilter_prompt(scene, request, candidates):
        return (
            f"Image scene description:\n{scene}\n\n"
            f"User editing request:\n{request}\n\n"
            f"KG candidates:\n{', '.join(candidates)}\n\n"
            "Write ONE image editing instruction:\n"
            "- 2–3 sentences maximum.\n"
            "- Use the KG cues to enrich the editing instruction.\n"
            "- Do NOT include reasoning — just the instruction.\n"
            "Return only the instruction text."
        )

    def build_llm_prompt(scene, request):
        return (
            f"Image scene description:\n{scene}\n\n"
            f"User editing request:\n{request}\n\n"
            "Write ONE image editing instruction:\n"
            "- 2–3 sentences maximum.\n"
            "- Only reference objects already visible in the scene description.\n"
            "- Do NOT include reasoning — just the instruction.\n"
            "Return only the instruction text."
        )

    for i, s in enumerate(todo_inst):
        h          = s["hash"]
        scene      = captions[h]
        request    = s["instruction"]
        candidates = retrieve_candidates(extract_keywords(request))

        prev = instructions.get(h, {})
        kg_inst = prev.get("kg_instruction") or generate(
            SYSTEM_PROMPT, build_kg_prompt(scene, request, candidates))
        kg_nofilter_inst = prev.get("kg_nofilter_instruction") or generate(
            SYSTEM_PROMPT, build_kg_nofilter_prompt(scene, request, candidates))
        llm_inst = prev.get("llm_instruction") or generate(
            SYSTEM_PROMPT, build_llm_prompt(scene, request))

        instructions[h] = {
            "kg_instruction"          : kg_inst,
            "kg_nofilter_instruction" : kg_nofilter_inst,
            "llm_instruction"         : llm_inst,
            "simple_prompt"           : request,
            "candidates"              : candidates,
            "scene"                   : scene,
        }
        print(f"  [{i+1:03d}/{len(todo_inst)}] {s['kg_condition']:<8} "
              f"KG: {kg_inst[:60]}...")

        if (i+1) % 10 == 0:
            with open(inst_cache_path, "w") as f:
                json.dump(instructions, f, ensure_ascii=False)

    with open(inst_cache_path, "w") as f:
        json.dump(instructions, f, ensure_ascii=False)
    print("✓ instruction 생성 완료")

    import gc
    del ds_model; gc.collect(); torch.cuda.empty_cache()
    print("✓ DeepSeek 해제")

# ── 6. FLUX + CLIP ────────────────────────────────────────────────────────
pipe = FluxKontextPipeline.from_pretrained(
    str(FLUX_PATH), torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
print("✓ FLUX 로드")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14").to("cuda").eval()
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("✓ CLIP 로드")

@torch.no_grad()
def clip_score(image, text):
    inputs = clip_proc(images=image, text=[text],
                       return_tensors="pt", padding=True).to("cuda")
    out    = clip_model(**inputs)
    img_e  = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_e  = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img_e * txt_e).sum()) * 100

def edit_image(image, instruction, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(image=image, prompt=instruction,
                num_inference_steps=28, guidance_scale=2.5,
                generator=gen).images[0]

# ── 7. FLUX 편집 + 평가 (seed 루프) ──────────────────────────────────────
todo_flux = [(s, seed) for s in samples for seed in SEEDS
             if (s["hash"], seed) not in done_keys]
print(f"\nFLUX 편집 필요: {len(todo_flux)}개 (sample × seed)")

for i, (s, seed) in enumerate(todo_flux):
    h     = s["hash"]
    inst  = instructions[h]
    orig  = Image.open(str(BASE / s["image_path"])).convert("RGB")
    tag   = h[:40]
    suf   = f"{tag}_s{seed}"

    kg_path          = OUTPUT_DIR / "kg_images"          / f"{suf}.png"
    kg_nofilter_path = OUTPUT_DIR / "kg_nofilter_images" / f"{suf}.png"
    llm_path         = OUTPUT_DIR / "llm_images"         / f"{suf}.png"
    simple_path      = OUTPUT_DIR / "simple_images"      / f"{suf}.png"

    kg_img          = edit_image(orig, inst["kg_instruction"],          seed)
    kg_nofilter_img = edit_image(orig, inst["kg_nofilter_instruction"], seed)
    llm_img         = edit_image(orig, inst["llm_instruction"],         seed)
    simple_img      = edit_image(orig, inst["simple_prompt"],           seed)

    kg_img.save(kg_path)
    kg_nofilter_img.save(kg_nofilter_path)
    llm_img.save(llm_path)
    simple_img.save(simple_path)

    out_cap          = s["output_caption"]
    kg_clip_gt       = clip_score(kg_img,          out_cap)
    kgn_clip_gt      = clip_score(kg_nofilter_img, out_cap)
    llm_clip_gt      = clip_score(llm_img,         out_cap)
    simp_clip_gt     = clip_score(simple_img,      out_cap)

    record = {
        "hash"                  : h,
        "seed"                  : seed,
        "kg_condition"          : s["kg_condition"],
        "simple_prompt"         : inst["simple_prompt"],
        "kg_instruction"        : inst["kg_instruction"],
        "kg_nofilter_instruction": inst["kg_nofilter_instruction"],
        "llm_instruction"       : inst["llm_instruction"],
        "input_caption"         : s["input_caption"],
        "output_caption"        : out_cap,
        "kg_clip_gt"            : round(kg_clip_gt,   4),
        "kg_nofilter_clip_gt"   : round(kgn_clip_gt,  4),
        "llm_clip_gt"           : round(llm_clip_gt,  4),
        "simple_clip_gt"        : round(simp_clip_gt, 4),
        "kg_image_path"         : str(kg_path),
        "kg_nofilter_image_path": str(kg_nofilter_path),
        "llm_image_path"        : str(llm_path),
        "simple_image_path"     : str(simple_path),
    }
    all_results.append(record)
    done_keys.add((h, seed))

    print(f"  [{i+1:03d}/{len(todo_flux)}] s={seed} {s['kg_condition']:<8} "
          f"KG={kg_clip_gt:.2f} KGn={kgn_clip_gt:.2f} "
          f"LLM={llm_clip_gt:.2f} Simp={simp_clip_gt:.2f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

# ── 8. 최종 요약 (seed별 평균 + 전체 평균) ────────────────────────────────
print(f"\n{'='*80}")
print("v6 RESULTS — seed별 / condition별")
print(f"{'='*80}")

for seed in SEEDS:
    rows_s = [r for r in all_results if r["seed"] == seed]
    if not rows_s: continue
    print(f"\n[seed={seed}]  N={len(rows_s)}")
    for cond in ["rainy","snowy","foggy","night","clear"]:
        rs = [r for r in rows_s if r["kg_condition"] == cond]
        if not rs: continue
        kg   = np.mean([r["kg_clip_gt"]          for r in rs])
        kgn  = np.mean([r["kg_nofilter_clip_gt"] for r in rs])
        llm  = np.mean([r["llm_clip_gt"]         for r in rs])
        simp = np.mean([r["simple_clip_gt"]      for r in rs])
        print(f"  {cond:<8} N={len(rs):>3}  "
              f"KG={kg:.2f}  KGn={kgn:.2f}  LLM={llm:.2f}  Simp={simp:.2f}")

print(f"\n[ALL SEEDS — mean ± std]")
for cond in ["rainy","snowy","foggy","night","clear","TOTAL"]:
    if cond == "TOTAL":
        rows = all_results
    else:
        rows = [r for r in all_results if r["kg_condition"] == cond]
    if not rows: continue
    # seed별 평균을 먼저 구하고, seed들 사이의 mean/std 계산
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    for col, label in [("kg_clip_gt","KG "), ("kg_nofilter_clip_gt","KGn"),
                       ("llm_clip_gt","LLM"), ("simple_clip_gt","Simp")]:
        seed_means = [np.mean([r[col] for r in by_seed[s]]) for s in by_seed]
        m, sd = np.mean(seed_means), np.std(seed_means)
        print(f"  {cond:<8} {label}: {m:.3f} ± {sd:.3f}")
    print()
