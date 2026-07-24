"""Generate IP2P edits under the editor-specific reported configuration.

The IP2P evaluation independently generates scene descriptions and instructions
using the MGIE-associated LLaVA model and an IP2P-specific DeepSeek system
prompt. It does not reuse FLUX instructions. The same structured repository and
four instruction conditions are retained, and the IP2P editor is not fine-tuned
or architecturally modified.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path

from sk_editing.io import (
    atomic_write_json,
    load_json,
    load_selected_samples,
    resolve_image_path,
    require_files,
)
from sk_editing.prompts import IP2P_PROMPTS, Method, build_instruction_prompt
from sk_editing.retrieval import StructuredRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ip2p"))
    parser.add_argument("--annotations", type=Path, default=Path("data/emu_edit_weather_final.json"))
    parser.add_argument("--sample-ids", type=Path, default=Path("data/ip2p_sample_ids.json"))
    parser.add_argument("--repository", type=Path, default=Path("data/structured_repository.json"))
    parser.add_argument("--ip2p", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--llm", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--mgie-code", type=Path, required=True)
    parser.add_argument("--mgie-llava", type=Path, required=True)
    parser.add_argument("--mgie-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-mgie-baseline", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    return parser.parse_args()


def load_or_empty(path: Path) -> dict:
    return load_json(path) if path.exists() else {}


def load_mgie(args: argparse.Namespace):
    import torch
    import transformers

    require_files(
        [args.mgie_code / "mgie_llava.py", args.mgie_checkpoint], label="MGIE artifact"
    )
    spec = importlib.util.spec_from_file_location("mgie_llava", args.mgie_code / "mgie_llava.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load MGIE's mgie_llava.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = module.LlavaLlamaForCausalLM

    sys.path.insert(0, str(args.mgie_code / "LLaVA"))
    from llava.conversation import conv_templates  # type: ignore

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.mgie_llava))
    model = model_class.from_pretrained(
        str(args.mgie_llava),
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_cache=True,
    ).to(args.device)
    image_processor = transformers.CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16
    )
    tokenizer.padding_side = "left"
    tokenizer.add_tokens([f"[IMG{i}]" for i in range(8)], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    checkpoint = torch.load(args.mgie_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    patch_token, start_token, end_token = "<im_patch>", "<im_start>", "<im_end>"
    use_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([patch_token], special_tokens=True)
    if use_start_end:
        tokenizer.add_tokens([start_token, end_token], special_tokens=True)
    vision_tower = model.get_model().vision_tower[0]
    vision_tower = transformers.CLIPVisionModel.from_pretrained(
        vision_tower.config._name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.get_model().vision_tower[0] = vision_tower
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(patch_token)
    vision_config.use_im_start_end = use_start_end
    if use_start_end:
        vision_config.im_start_token = tokenizer.convert_tokens_to_ids(start_token)
        vision_config.im_end_token = tokenizer.convert_tokens_to_ids(end_token)
    image_token_length = (vision_config.image_size // vision_config.patch_size) ** 2
    model.eval()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "conversation_templates": conv_templates,
        "patch_token": patch_token,
        "start_token": start_token,
        "end_token": end_token,
        "use_start_end": use_start_end,
        "image_token_length": image_token_length,
        "checkpoint": checkpoint,
    }


def clean_mgie_instruction(text: str) -> str:
    """Apply the same two-sentence cleanup used in the archived MGIE run."""
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[1].strip()
    if "</s>" in text:
        text = text.split("</s>", 1)[0].strip()
    lower = text.lower()
    if "alternative" in lower:
        text = text[: lower.index("alternative")]
    if "[IMG0]" in text:
        text = text.split("[IMG0]", 1)[0]
    sentences = [part.strip() for part in text.split(".") if part.strip()]
    text = ".".join(sentences[:2]).strip()
    if text and not text.endswith("."):
        text += "."
    return text


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_selected_samples(args.annotations, args.sample_ids)
    if args.limit is not None:
        samples = samples[: args.limit]
    repository = StructuredRepository.from_path(args.repository)

    cache_path = args.output_dir / "instruction_cache.json"
    manifest_path = args.output_dir / "manifest.json"
    cache = load_or_empty(cache_path)
    for key in ("scene_descriptions", "mgie", "llm_only", "sk_filter", "sk_llm"):
        cache.setdefault(key, {})
    manifest = load_json(manifest_path) if manifest_path.exists() else []

    # Stage 1: MGIE-associated LLaVA scene descriptions and optional MGIE baseline.
    pending_scene = [s for s in samples if s["hash"] not in cache["scene_descriptions"]]
    pending_mgie = [s for s in samples if s["hash"] not in cache["mgie"]]
    if pending_scene or (pending_mgie and not args.skip_mgie_baseline):
        import torch
        from PIL import Image

        mgie = load_mgie(args)
        tokenizer = mgie["tokenizer"]
        model = mgie["model"]
        processor = mgie["image_processor"]
        conv_templates = mgie["conversation_templates"]
        patch_tokens = mgie["patch_token"] * mgie["image_token_length"]
        image_wrapped = (
            f"{mgie['start_token']}{patch_tokens}{mgie['end_token']}"
            if mgie["use_start_end"]
            else patch_tokens
        )

        def run_prompt(sample: dict, text: str, *, max_new_tokens: int) -> str:
            image = Image.open(resolve_image_path(args.image_root, sample["image_path"])).convert("RGB")
            image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            conversation = conv_templates["vicuna_v1_1"].copy()
            conversation.append_message(conversation.roles[0], f"{text}\n{image_wrapped}")
            conversation.append_message(conversation.roles[1], None)
            tokenized = tokenizer(conversation.get_prompt())
            input_ids = torch.as_tensor(tokenized["input_ids"]).unsqueeze(0).to(args.device)
            attention = torch.as_tensor(tokenized["attention_mask"]).unsqueeze(0).to(args.device)
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    images=image_tensor.half().unsqueeze(0).to(args.device),
                    attention_mask=attention,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            if "ASSISTANT:" in decoded:
                decoded = decoded.split("ASSISTANT:", 1)[1]
            return decoded.strip()

        targets = {sample["hash"]: sample for sample in [*pending_scene, *pending_mgie]}
        for index, sample in enumerate(targets.values(), 1):
            sample_hash = sample["hash"]
            if sample_hash not in cache["scene_descriptions"]:
                cache["scene_descriptions"][sample_hash] = run_prompt(
                    sample, IP2P_PROMPTS.scene_prompt, max_new_tokens=256
                )
            if not args.skip_mgie_baseline and sample_hash not in cache["mgie"]:
                raw = run_prompt(
                    sample,
                    f"what will this image be like if '{sample['instruction']}'",
                    max_new_tokens=96,
                )
                cache["mgie"][sample_hash] = clean_mgie_instruction(raw)
            if index % 10 == 0 or index == len(targets):
                atomic_write_json(cache_path, cache)
            print(f"[MGIE/VLM {index:03d}/{len(targets)}] {sample_hash[:12]}")
        del mgie
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 2: editor-specific DeepSeek instruction generation.
    pending = [
        sample
        for sample in samples
        if any(sample["hash"] not in cache[key] for key in ("llm_only", "sk_filter", "sk_llm"))
    ]
    if pending:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from sk_editing.text_generation import generate_instruction

        tokenizer = AutoTokenizer.from_pretrained(args.llm, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.llm,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=quantization,
        ).eval()

        for index, sample in enumerate(pending, 1):
            sample_hash = sample["hash"]
            scene = cache["scene_descriptions"][sample_hash]
            by_request = repository.retrieve(sample["instruction"])
            by_label = repository.retrieve_label(sample["kg_condition"])
            retrieval = by_label if by_label.cues else by_request
            if not retrieval.cues:
                raise RuntimeError(f"No cues found for {sample_hash}")
            if by_request.matched_entries and set(by_request.matched_entries) != set(by_label.matched_entries):
                print(
                    f"[warning] request/label retrieval differ for {sample_hash[:12]}: "
                    f"request={by_request.matched_entries}, label={by_label.matched_entries}"
                )
            for method in (Method.LLM_ONLY, Method.SK_FILTER, Method.SK_LLM):
                if sample_hash in cache[method.value]:
                    continue
                prompt = build_instruction_prompt(
                    method,
                    scene=scene,
                    request=sample["instruction"],
                    cues=retrieval.cues,
                    editor="ip2p",
                )
                cache[method.value][sample_hash] = generate_instruction(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=IP2P_PROMPTS.system_prompt,
                    user_prompt=prompt,
                    max_new_tokens=IP2P_PROMPTS.max_new_tokens,
                    reasoning_mode=IP2P_PROMPTS.reasoning_mode,
                )
            if index % 10 == 0 or index == len(pending):
                atomic_write_json(cache_path, cache)
            print(f"[instruction {index:03d}/{len(pending)}] {sample_hash[:12]}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if args.skip_images:
        print("Stopped after instruction generation (--skip-images).")
        return

    # Stage 3: InstructPix2Pix editing.
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
    from PIL import Image

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.ip2p,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(args.device)

    done = {(row["hash"], row["method"]) for row in manifest}
    methods = [Method.SIMPLE, Method.LLM_ONLY, Method.SK_FILTER, Method.SK_LLM]
    if not args.skip_mgie_baseline:
        method_names = ["simple", "mgie", "llm_only", "sk_filter", "sk_llm"]
    else:
        method_names = [method.value for method in methods]

    for sample_index, sample in enumerate(samples, 1):
        sample_hash = sample["hash"]
        source_path = resolve_image_path(args.image_root, sample["image_path"])
        source = Image.open(source_path).convert("RGB").resize((512, 512))
        instructions = {
            "simple": sample["instruction"],
            "mgie": cache["mgie"].get(sample_hash, sample["instruction"]),
            "llm_only": cache["llm_only"][sample_hash],
            "sk_filter": cache["sk_filter"][sample_hash],
            "sk_llm": cache["sk_llm"][sample_hash],
        }
        for method_name in method_names:
            if (sample_hash, method_name) in done:
                continue
            output_path = args.output_dir / "images" / method_name / f"{sample_hash}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            edited = pipeline(
                instructions[method_name],
                image=source,
                num_inference_steps=50,
                image_guidance_scale=1.5,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
            edited.save(output_path)
            manifest.append(
                {
                    "hash": sample_hash,
                    "seed": args.seed,
                    "method": method_name,
                    "condition": sample["kg_condition"],
                    "instruction": instructions[method_name],
                    "input_caption": sample["input_caption"],
                    "output_caption": sample["output_caption"],
                    "source_image": str(source_path),
                    "edited_image": str(output_path),
                }
            )
            done.add((sample_hash, method_name))
            atomic_write_json(manifest_path, manifest)
        print(f"[edit {sample_index:03d}/{len(samples)}] {sample_hash[:12]}")


if __name__ == "__main__":
    main()
