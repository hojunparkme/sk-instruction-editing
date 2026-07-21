"""
Fair Comparison v4: MGIE vs KG+LLM (with/without filter) vs LLM-only vs Simple
- 편집 모델: InstructPix2Pix (IP2P) 로 통일
- MGIE: tsujuifu/ml-mgie official checkpoint (mllm.pt)
- instruction 생성 방법만 다르게 5-way 비교
- 변경점 (v3 대비):
    1. KGn (KG without scene-grounded filtering) 추가
    2. LLM-only 추가
    3. Prefill trick으로 DeepSeek reasoning 봉쇄 + deterministic
    4. 기존 v3 cache 그대로 호환 (mgie_instructions, kg_instructions, scene_descs)
"""

import os, sys, json, re, torch, gc
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    AutoModel, AutoImageProcessor, AutoConfig,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────
from config import (  # noqa: E402
    ROOT as BASE, EMU_PATH, REPOSITORY_PATH, FLUX_SAMPLE_IDS,
    LLAVA_PATH, FLUX_PATH, IP2P_PATH, MGIE_CODE, MGIE_LLAVA, MGIE_CKPT,
    FLUX_OUT, IP2P_OUT, DEVICE,
)
DATA_PATH = EMU_PATH
KG_PATH = REPOSITORY_PATH

OUTPUT_DIR = IP2P_OUT

MGIE_CODE = str(MGIE_CODE)
PATH_LLAVA = str(MGIE_LLAVA)
PATH_MGIE_CKPT = str(MGIE_CKPT)

OUTPUT_DIR.mkdir(exist_ok=True)
for m in ["simple", "mgie_style", "llm_only", "kg_llm", "kg_llm_nofilter"]:
    (OUTPUT_DIR / f"{m}_images").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 데이터 로드 + 필터링
# ─────────────────────────────────────────────────────────────
EXCLUDE_HASHES = {
    "e7d4f466fdcfdbca4aafb26fe1cf8690_0e0311c66655333f77b6ed7411904e739fe1d6a2a03a9d31bca596fa9fd2bf27",
    "8eec8c073e04434436843c075645c1c5_b501df7f2ba8eb7822d2dae9ca9f14c774ecb7e85d4679d7e5600765a5f1db21",
    "6025e8117319a015a18108aa3656175e_f10bc2e991c6faa08d5f256a5683d0d65d4425afa435a27bf0406f184afa88d0",
    "ee8550c538b90211f72fab420b173c40_bfe5c912fee05f83d30d54cb279a59d9997ba06bfec12b7376198bb41278241f",
    "cbba5e54a015b42f9856be678df0871f_810106564569c5bf931d07847bddd2b7772b54036a6a5352ef78018e99686287",
    "c165327917aadba99cecf6f72ecc15fd_af79f6c66384989bd183812a5531fc93f47e3a33f62b51c46e5da896b5fbb91d",
    "f2c3701874e5ca46cc0c016fc8322ed1_36c230f70b5e93b678af60b97bbdb814894357efbb5f4efaef2e9266767b1f09",
    "94941ee652a890a6c6cf5b3f1e0b8203_f222f574e26b5e5808370784211e3e42d8905881ffb4e11c439bd2a104e6c20c",
}

with open(DATA_PATH, encoding="utf-8") as f:
    emu_data = json.load(f)
with open(KG_PATH, encoding="utf-8") as f:
    kg = json.load(f)

all_samples = emu_data["samples"]
samples = [s for s in all_samples if s["hash"] not in EXCLUDE_HASHES]
print(f"Total: {len(all_samples)} -> After filtering: {len(samples)}")

# ─────────────────────────────────────────────────────────────
# 캐시 로드
# ─────────────────────────────────────────────────────────────
CACHE_PATH   = OUTPUT_DIR / "cache.json"
RESULTS_PATH = OUTPUT_DIR / "results.json"

# v3 cache 가져오기 옵션 (mgie_instructions / scene_descs 재사용)
V3_CACHE = BASE / "results_comparison_ip2p" / "cache.json"

if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
elif V3_CACHE.exists():
    print(f"v3 cache 발견 → import: {V3_CACHE}")
    with open(V3_CACHE) as f:
        cache = json.load(f)
    # 기존 v3의 kg_instructions는 do_sample=True + 후처리만 (오염 위험)
    # → prefill 일관성을 위해 폐기, KG/KGn/LLM 모두 prefill로 새로 생성
    if "kg_instructions" in cache:
        print(f"  v3 kg_instructions 폐기 ({len(cache['kg_instructions'])}개) "
              f"→ prefill로 재생성 예정")
        cache["kg_instructions"] = {}
else:
    cache = {}

# 5개 cache key 보장
for key in ["mgie_instructions", "kg_instructions",
            "kg_nofilter_instructions", "llm_instructions",
            "scene_descs"]:
    if key not in cache:
        cache[key] = {}

def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

all_results = []
if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)
done_hashes = {r["hash"] for r in all_results}
print(f"Already done: {len(done_hashes)}")

def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────
# KG cue 추출
# ─────────────────────────────────────────────────────────────
def get_kg_cues(condition, max_cues=50):
    cond_key = condition.lower()
    if cond_key not in kg:
        return ""
    node = kg[cond_key]
    cues = []
    for category, items in node.items():
        if isinstance(items, list):
            cues.extend(items)
        elif isinstance(items, dict):
            for sub_items in items.values():
                if isinstance(sub_items, list):
                    cues.extend(sub_items)
    return ", ".join(cues[:max_cues])

# ─────────────────────────────────────────────────────────────
# Phase 1: MGIE (official checkpoint) + scene_desc
# ─────────────────────────────────────────────────────────────
todo_mgie = [s for s in samples
             if s["hash"] not in cache["mgie_instructions"]
             or s["hash"] not in cache["scene_descs"]]

print(f"\n[Phase 1] MGIE needed: {len(todo_mgie)}")

if todo_mgie:
    import transformers

    # mgie_llava.py에서 exist_ok=True 패치 후 로드
    import importlib.util
    spec = importlib.util.spec_from_file_location("mgie_llava", f"{MGIE_CODE}/mgie_llava.py")
    mgie_llava_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mgie_llava_mod)
    LlavaLlamaForCausalLM = mgie_llava_mod.LlavaLlamaForCausalLM

    # conv_templates: LLaVA submodule에서 로드 (MPT import 제거된 상태)
    sys.path.insert(0, os.path.join(MGIE_CODE, "LLaVA"))
    from llava.conversation import conv_templates

    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN    = "<im_start>"
    DEFAULT_IM_END_TOKEN      = "<im_end>"

    # 모델 로드
    mgie_tok = transformers.AutoTokenizer.from_pretrained(PATH_LLAVA)
    mgie_model = LlavaLlamaForCausalLM.from_pretrained(
        PATH_LLAVA, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True
    ).cuda()
    mgie_image_processor = transformers.CLIPImageProcessor.from_pretrained(
        mgie_model.config.mm_vision_tower, torch_dtype=torch.float16
    )

    mgie_tok.padding_side = "left"
    mgie_tok.add_tokens(
        ["[IMG0]","[IMG1]","[IMG2]","[IMG3]","[IMG4]","[IMG5]","[IMG6]","[IMG7]"],
        special_tokens=True
    )
    mgie_model.resize_token_embeddings(len(mgie_tok))

    # official mllm.pt 로드
    ckpt = torch.load(PATH_MGIE_CKPT, map_location="cpu")
    mgie_model.load_state_dict(ckpt, strict=False)

    mm_use_im_start_end = getattr(mgie_model.config, "mm_use_im_start_end", False)
    mgie_tok.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        mgie_tok.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = mgie_model.get_model().vision_tower[0]
    vision_tower = transformers.CLIPVisionModel.from_pretrained(
        vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).cuda()
    mgie_model.get_model().vision_tower[0] = vision_tower
    vision_config = vision_tower.config
    vision_config.im_patch_token = mgie_tok.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = mgie_tok.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    _ = mgie_model.eval()
    print(f"MGIE loaded  (image_token_len={image_token_len})")

    def remove_alter(s):
        if "ASSISTANT:" in s: s = s[s.index("ASSISTANT:")+10:].strip()
        if "</s>" in s: s = s[:s.index("</s>")].strip()
        if "alternative" in s.lower(): s = s[:s.lower().index("alternative")]
        if "[IMG0]" in s: s = s[:s.index("[IMG0]")]
        s = ".".join([x.strip() for x in s.split(".")[:2]])
        if s and s[-1] != ".": s += "."
        return s.strip()

    @torch.inference_mode()
    def run_mgie(image_path, instruction):
        img = Image.open(str(BASE / image_path)).convert("RGB")
        img_tensor = mgie_image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]

        txt = "what will this image be like if '%s'" % instruction
        txt = (txt + "\n" + DEFAULT_IM_START_TOKEN
               + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
               + DEFAULT_IM_END_TOKEN)
        conv = conv_templates["vicuna_v1_1"].copy()
        conv.append_message(conv.roles[0], txt)
        conv.append_message(conv.roles[1], None)
        txt = conv.get_prompt()
        tok = mgie_tok(txt)
        txt_ids = torch.as_tensor(tok["input_ids"])
        mask    = torch.as_tensor(tok["attention_mask"])

        out = mgie_model.generate(
            txt_ids.unsqueeze(0).cuda(),
            images=img_tensor.half().unsqueeze(0).cuda(),
            attention_mask=mask.unsqueeze(0).cuda(),
            do_sample=False, max_new_tokens=96, num_beams=1,
            no_repeat_ngram_size=3,
            eos_token_id=mgie_tok.eos_token_id,
            pad_token_id=mgie_tok.eos_token_id,
            use_cache=False,
            return_dict_in_generate=True, output_hidden_states=False,
        )
        raw = mgie_tok.decode(out["sequences"][0].tolist())
        return remove_alter(raw)

    @torch.inference_mode()
    def run_scene_desc(image_path):
        """LLaVA-1.5 (transformers built-in) 대신 MGIE 모델로 scene 묘사"""
        img = Image.open(str(BASE / image_path)).convert("RGB")
        img_tensor = mgie_image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]

        txt = ("Describe this image in detail. "
               "Include visible objects, people, weather, lighting, and atmosphere.\n"
               + DEFAULT_IM_START_TOKEN
               + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
               + DEFAULT_IM_END_TOKEN)
        conv = conv_templates["vicuna_v1_1"].copy()
        conv.append_message(conv.roles[0], txt)
        conv.append_message(conv.roles[1], None)
        txt = conv.get_prompt()
        tok = mgie_tok(txt)
        txt_ids = torch.as_tensor(tok["input_ids"])
        mask    = torch.as_tensor(tok["attention_mask"])

        out = mgie_model.generate(
            txt_ids.unsqueeze(0).cuda(),
            images=img_tensor.half().unsqueeze(0).cuda(),
            attention_mask=mask.unsqueeze(0).cuda(),
            do_sample=False, max_new_tokens=256, num_beams=1,
            no_repeat_ngram_size=3,
            eos_token_id=mgie_tok.eos_token_id,
            pad_token_id=mgie_tok.eos_token_id,
            use_cache=False,
            return_dict_in_generate=False,
        )
        raw = mgie_tok.decode(out[0].tolist(), skip_special_tokens=True)
        if "ASSISTANT:" in raw:
            raw = raw[raw.index("ASSISTANT:")+10:].strip()
        return raw

    for i, s in enumerate(tqdm(todo_mgie, desc="MGIE + scene")):
        h = s["hash"]
        if h not in cache["mgie_instructions"]:
            cache["mgie_instructions"][h] = run_mgie(s["image_path"], s["instruction"])
        if h not in cache["scene_descs"]:
            cache["scene_descs"][h] = run_scene_desc(s["image_path"])
        if (i + 1) % 10 == 0:
            save_cache()

    save_cache()
    del mgie_model, vision_tower, ckpt
    gc.collect(); torch.cuda.empty_cache()
    print("MGIE unloaded")
else:
    print("  Skipped (all cached)")

# ─────────────────────────────────────────────────────────────
# Phase 2: DeepSeek (KG+LLM instruction)
# ─────────────────────────────────────────────────────────────
def needs_all_inst(h):
    return (h not in cache["kg_instructions"]
            or h not in cache["kg_nofilter_instructions"]
            or h not in cache["llm_instructions"])

todo_ds = [s for s in samples if needs_all_inst(s["hash"])]
print(f"\n[Phase 2] DeepSeek needed: {len(todo_ds)} (KG / KGn / LLM 3 변종)")

if todo_ds:
    DS_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ds_tok = AutoTokenizer.from_pretrained(DS_MODEL, use_fast=True)
    if ds_tok.pad_token is None:
        ds_tok.pad_token = ds_tok.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    ds_model = AutoModelForCausalLM.from_pretrained(
        DS_MODEL, device_map="auto", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, quantization_config=quant,
    ).eval()
    print("DeepSeek loaded")

    SYSTEM_PROMPT = (
        "You are an expert at writing image editing instructions for InstructPix2Pix.\n"
        "Output ONLY the final instruction text — no analysis, no JSON, no preamble, no reasoning."
    )

    @torch.no_grad()
    def generate(user_prompt, max_new_tokens=768):
        """
        Prefill trick: assistant 응답 시작에 <think>\\n\\n</think>\\n\\n 를
        미리 삽입해서 reasoning 단계를 강제 종료. + deterministic.
        """
        prompt_text = ds_tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": user_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_text = prompt_text + "<think>\n\n</think>\n\n"

        inputs = ds_tok(prompt_text, return_tensors="pt").to(
            next(ds_model.parameters()).device)
        out = ds_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                     # deterministic
            pad_token_id=ds_tok.eos_token_id,
        )
        text = ds_tok.decode(out[0, inputs["input_ids"].shape[-1]:],
                             skip_special_tokens=True).strip()
        # 안전장치
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        text = re.sub(r"<\|.*?\|>|<｜.*?｜>", "", text).strip()
        text = text.strip('"').strip("'").strip()
        text = text.split("\n\n")[0].strip()
        return text

    def build_kg_prompt(scene, request, kg_cues):
        return (
            f"Image scene description:\n{scene}\n\n"
            f"User editing request:\n{request}\n\n"
            f"KG candidates:\n{kg_cues}\n\n"
            "Write ONE image editing instruction:\n"
            "- 2-3 sentences maximum.\n"
            "- Only use KG cues that match objects VISIBLE in the scene description above.\n"
            "- Do NOT use cues referencing elements absent from the scene.\n"
            "- Do NOT include reasoning -- just the instruction.\n"
            "Return only the instruction text."
        )

    def build_kg_nofilter_prompt(scene, request, kg_cues):
        return (
            f"Image scene description:\n{scene}\n\n"
            f"User editing request:\n{request}\n\n"
            f"KG candidates:\n{kg_cues}\n\n"
            "Write ONE image editing instruction:\n"
            "- 2-3 sentences maximum.\n"
            "- Use the KG cues to enrich the editing instruction.\n"
            "- Do NOT include reasoning -- just the instruction.\n"
            "Return only the instruction text."
        )

    def build_llm_prompt(scene, request):
        return (
            f"Image scene description:\n{scene}\n\n"
            f"User editing request:\n{request}\n\n"
            "Write ONE image editing instruction:\n"
            "- 2-3 sentences maximum.\n"
            "- Only reference objects already visible in the scene description.\n"
            "- Do NOT include reasoning -- just the instruction.\n"
            "Return only the instruction text."
        )

    for i, s in enumerate(tqdm(todo_ds, desc="DeepSeek KG/KGn/LLM")):
        h       = s["hash"]
        scene   = cache["scene_descs"].get(h, s.get("input_caption", ""))
        kg_cues = get_kg_cues(s["kg_condition"])

        if h not in cache["kg_instructions"]:
            cache["kg_instructions"][h] = generate(
                build_kg_prompt(scene, s["instruction"], kg_cues))
        if h not in cache["kg_nofilter_instructions"]:
            cache["kg_nofilter_instructions"][h] = generate(
                build_kg_nofilter_prompt(scene, s["instruction"], kg_cues))
        if h not in cache["llm_instructions"]:
            cache["llm_instructions"][h] = generate(
                build_llm_prompt(scene, s["instruction"]))

        if (i + 1) % 10 == 0:
            save_cache()

    save_cache()
    del ds_model; gc.collect(); torch.cuda.empty_cache()
    print("DeepSeek unloaded")
else:
    print("  Skipped (all cached)")

# ─────────────────────────────────────────────────────────────
# Phase 3: IP2P 편집 + 5개 지표
# ─────────────────────────────────────────────────────────────
todo_edit = [s for s in samples if s["hash"] not in done_hashes]
print(f"\n[Phase 3] IP2P editing needed: {len(todo_edit)}")

if todo_edit:
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    import clip as CLIP

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        str(IP2P_PATH), torch_dtype=torch.float16,
        safety_checker=None, local_files_only=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    print("IP2P loaded")

    clip_model, clip_prep = CLIP.load("ViT-L/14", device="cuda")
    print("CLIP loaded")

    dino_proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
    print("DINOv2 loaded")

    def edit_image(image_path, instruction, seed=42):
        image = Image.open(str(BASE / image_path)).convert("RGB").resize((512, 512))
        gen   = torch.Generator("cuda").manual_seed(seed)
        return pipe(
            instruction, image=image,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            generator=gen,
        ).images[0]

    @torch.no_grad()
    def compute_metrics(orig_path, edited_img, instruction, target_caption):
        orig_img = Image.open(str(BASE / orig_path)).convert("RGB").resize((512, 512))

        orig_t     = clip_prep(orig_img).unsqueeze(0).cuda()
        edit_t     = clip_prep(edited_img).unsqueeze(0).cuda()
        instr_tok  = CLIP.tokenize([instruction],    truncate=True).cuda()
        target_tok = CLIP.tokenize([target_caption], truncate=True).cuda()

        f_orig   = clip_model.encode_image(orig_t)
        f_edit   = clip_model.encode_image(edit_t)
        f_instr  = clip_model.encode_text(instr_tok)
        f_target = clip_model.encode_text(target_tok)

        delta_img  = f_edit - f_orig
        delta_img  = delta_img / delta_img.norm(dim=-1, keepdim=True)
        delta_text = f_instr  / f_instr.norm(dim=-1, keepdim=True)
        clip_dir   = float((delta_img * delta_text).sum())

        f_orig_n = f_orig / f_orig.norm(dim=-1, keepdim=True)
        f_edit_n = f_edit / f_edit.norm(dim=-1, keepdim=True)
        clip_im  = float((f_orig_n * f_edit_n).sum())

        f_edit_n2  = f_edit   / f_edit.norm(dim=-1, keepdim=True)
        f_target_n = f_target / f_target.norm(dim=-1, keepdim=True)
        clip_out   = float((f_edit_n2 * f_target_n).sum())

        orig_arr = np.array(orig_img).astype(np.float32) / 255.0
        edit_arr = np.array(edited_img.resize((512, 512))).astype(np.float32) / 255.0
        l1 = float(np.mean(np.abs(orig_arr - edit_arr)))

        dino_inputs = dino_proc(
            images=[orig_img, edited_img.resize((512, 512))],
            return_tensors="pt"
        ).to("cuda")
        dino_out = dino_model(**dino_inputs)
        feats = dino_out.last_hidden_state[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        dino  = float((feats[0] * feats[1]).sum())

        return {
            "clip_dir": round(clip_dir, 4),
            "clip_im":  round(clip_im,  4),
            "clip_out": round(clip_out, 4),
            "l1":       round(l1,       4),
            "dino":     round(dino,     4),
        }

    for i, s in enumerate(tqdm(todo_edit, desc="IP2P")):
        h            = s["hash"]
        simple_instr = s["instruction"]
        mgie_instr   = cache["mgie_instructions"].get(h, simple_instr)
        llm_instr    = cache["llm_instructions"].get(h, simple_instr)
        kg_instr     = cache["kg_instructions"].get(h, simple_instr)
        kgn_instr    = cache["kg_nofilter_instructions"].get(h, simple_instr)
        target_cap   = s.get("output_caption", simple_instr)
        tag          = h[:40]

        simple_img = edit_image(s["image_path"], simple_instr)
        mgie_img   = edit_image(s["image_path"], mgie_instr)
        llm_img    = edit_image(s["image_path"], llm_instr)
        kg_img     = edit_image(s["image_path"], kg_instr)
        kgn_img    = edit_image(s["image_path"], kgn_instr)

        simple_img.save(OUTPUT_DIR / "simple_images"          / f"{tag}.png")
        mgie_img.save(  OUTPUT_DIR / "mgie_style_images"      / f"{tag}.png")
        llm_img.save(   OUTPUT_DIR / "llm_only_images"        / f"{tag}.png")
        kg_img.save(    OUTPUT_DIR / "kg_llm_images"          / f"{tag}.png")
        kgn_img.save(   OUTPUT_DIR / "kg_llm_nofilter_images" / f"{tag}.png")

        simple_m = compute_metrics(s["image_path"], simple_img, simple_instr, target_cap)
        mgie_m   = compute_metrics(s["image_path"], mgie_img,   simple_instr, target_cap)
        llm_m    = compute_metrics(s["image_path"], llm_img,    llm_instr,    target_cap)
        kg_m     = compute_metrics(s["image_path"], kg_img,     kg_instr,     target_cap)
        kgn_m    = compute_metrics(s["image_path"], kgn_img,    kgn_instr,    target_cap)

        record = {
            "hash":                     h,
            "kg_condition":             s["kg_condition"],
            "simple_instruction":       simple_instr,
            "mgie_instruction":         mgie_instr,
            "llm_instruction":          llm_instr,
            "kg_instruction":           kg_instr,
            "kg_nofilter_instruction":  kgn_instr,
            "simple":                   simple_m,
            "mgie_style":               mgie_m,
            "llm_only":                 llm_m,
            "kg_llm":                   kg_m,
            "kg_llm_nofilter":          kgn_m,
        }
        all_results.append(record)
        done_hashes.add(h)

        print(f"  [{i+1:03d}/{len(todo_edit)}] {s['kg_condition']:<8} "
              f"KG={kg_m['clip_dir']:.4f} KGn={kgn_m['clip_dir']:.4f} "
              f"LLM={llm_m['clip_dir']:.4f} "
              f"MGIE={mgie_m['clip_dir']:.4f} Simp={simple_m['clip_dir']:.4f}")

        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*95}")
print("RESULTS v4 (IP2P backbone, 5-way: Simple / MGIE / LLM / KG / KGn-ours)")
print(f"{'='*95}")

methods      = ["simple", "mgie_style", "llm_only", "kg_llm", "kg_llm_nofilter"]
method_label = {"simple":"Simple", "mgie_style":"MGIE", "llm_only":"LLM",
                "kg_llm":"KG", "kg_llm_nofilter":"KGn(ours)"}
metric_names = ["clip_dir", "clip_im", "clip_out", "l1", "dino"]
direction    = {"clip_dir":"up","clip_im":"up","clip_out":"up","l1":"down","dino":"up"}

header = f"{'Metric':<10} {'dir':>4}"
for method in methods:
    header += f" {method_label[method]:>11}"
header += f" {'dKGn-S':>9} {'dKGn-M':>9} {'dKGn-LLM':>10}"
print(header)
print("-" * len(header))

for m in metric_names:
    vals = {}
    for method in methods:
        v = [r[method][m] for r in all_results if method in r and m in r[method]]
        vals[method] = float(np.mean(v)) if v else 0.0
    line = f"{m:<10} {direction[m]:>4}"
    for method in methods:
        line += f" {vals[method]:>11.4f}"
    line += (f" {vals['kg_llm_nofilter']-vals['simple']:>+9.4f}"
             f" {vals['kg_llm_nofilter']-vals['mgie_style']:>+9.4f}"
             f" {vals['kg_llm_nofilter']-vals['llm_only']:>+10.4f}")
    print(line)

print(f"\n[CLIP_dir by condition]")
for cond in ["rainy", "snowy", "foggy", "night", "clear"]:
    rows = [r for r in all_results if r["kg_condition"] == cond]
    if not rows:
        continue
    vals = {}
    for method in methods:
        v = [r[method]["clip_dir"] for r in rows if method in r]
        vals[method] = float(np.mean(v)) if v else 0.0
    parts = [f"{method_label[m]}={vals[m]:.4f}" for m in methods]
    print(f"  {cond:<8} N={len(rows):>3}: " + "  ".join(parts))

summary = {}
for m in metric_names:
    summary[m] = {}
    for method in methods:
        v = [r[method][m] for r in all_results if method in r and m in r[method]]
        summary[m][method] = float(np.mean(v)) if v else 0.0
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone: {OUTPUT_DIR}/")
