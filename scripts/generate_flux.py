"""Generate FLUX Kontext edits for the four instruction conditions.

This is a cleaned reference implementation of the documented FLUX experiment.
It preserves the reported editor-specific prompt configuration while replacing
workspace-specific paths and fragile checkpoint writes with explicit CLI
arguments and atomic JSON checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

from sk_editing.io import (
    atomic_write_json,
    load_json,
    load_selected_samples,
    resolve_image_path,
)
from sk_editing.prompts import FLUX_PROMPTS, Method, build_instruction_prompt
from sk_editing.retrieval import StructuredRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/flux"))
    parser.add_argument("--annotations", type=Path, default=Path("data/emu_edit_weather_final.json"))
    parser.add_argument("--sample-ids", type=Path, default=Path("data/flux_sample_ids.json"))
    parser.add_argument("--repository", type=Path, default=Path("data/structured_repository.json"))
    parser.add_argument("--llava", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--llm", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--flux", default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-images", action="store_true", help="Generate captions/instructions only")
    return parser.parse_args()


def load_or_empty(path: Path) -> dict:
    return load_json(path) if path.exists() else {}


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_selected_samples(args.annotations, args.sample_ids)
    if args.limit is not None:
        samples = samples[: args.limit]
    repository = StructuredRepository.from_path(args.repository)

    captions_path = args.output_dir / "captions.json"
    instructions_path = args.output_dir / "instructions.json"
    manifest_path = args.output_dir / "manifest.json"
    captions = load_or_empty(captions_path)
    instructions = load_or_empty(instructions_path)
    manifest = load_json(manifest_path) if manifest_path.exists() else []

    # Stage 1: LLaVA scene descriptions.
    pending = [sample for sample in samples if sample["hash"] not in captions]
    if pending:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.llava)
        model = LlavaForConditionalGeneration.from_pretrained(
            args.llava,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()
        for index, sample in enumerate(pending, 1):
            image = Image.open(resolve_image_path(args.image_root, sample["image_path"])).convert("RGB")
            prompt = f"USER: <image>\n{FLUX_PROMPTS.scene_prompt}\nASSISTANT:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            caption = processor.decode(
                output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            ).strip()
            captions[sample["hash"]] = caption
            if index % 10 == 0 or index == len(pending):
                atomic_write_json(captions_path, captions)
            print(f"[caption {index:03d}/{len(pending)}] {sample['hash'][:12]}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 2: DeepSeek instruction generation.
    required_methods = (Method.LLM_ONLY, Method.SK_FILTER, Method.SK_LLM)
    pending = [
        sample
        for sample in samples
        if any(method.value not in instructions.get(sample["hash"], {}) for method in required_methods)
    ]
    if pending:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from sk_editing.text_generation import generate_instruction

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.llm, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.llm,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=quantization,
        ).eval()

        for index, sample in enumerate(pending, 1):
            sample_hash = sample["hash"]
            scene = captions[sample_hash]
            retrieval = repository.retrieve(sample["instruction"])
            if not retrieval.cues:
                raise RuntimeError(
                    f"No structured cues matched request for {sample_hash}: {sample['instruction']}"
                )
            record = instructions.setdefault(sample_hash, {})
            record.update(
                {
                    "simple": sample["instruction"],
                    "scene_description": scene,
                    "matched_entries": [list(item) for item in retrieval.matched_entries],
                    "cues": list(retrieval.cues),
                }
            )
            for method in required_methods:
                if method.value in record:
                    continue
                user_prompt = build_instruction_prompt(
                    method,
                    scene=scene,
                    request=sample["instruction"],
                    cues=retrieval.cues,
                    editor="flux",
                )
                record[method.value] = generate_instruction(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=FLUX_PROMPTS.system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=FLUX_PROMPTS.max_new_tokens,
                    reasoning_mode=FLUX_PROMPTS.reasoning_mode,
                )
            if index % 10 == 0 or index == len(pending):
                atomic_write_json(instructions_path, instructions)
            print(f"[instruction {index:03d}/{len(pending)}] {sample_hash[:12]}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if args.skip_images:
        print("Stopped after caption and instruction generation (--skip-images).")
        return

    # Stage 3: FLUX edits. Metrics are computed separately by recompute_metrics.py.
    import torch
    from diffusers import FluxKontextPipeline
    from PIL import Image

    pipeline = FluxKontextPipeline.from_pretrained(args.flux, torch_dtype=torch.bfloat16).to(args.device)
    done = {(row["hash"], row["seed"], row["method"]) for row in manifest}
    method_order = (Method.SIMPLE, Method.LLM_ONLY, Method.SK_FILTER, Method.SK_LLM)

    total = len(samples) * len(args.seeds) * len(method_order)
    completed = 0
    for sample in samples:
        sample_hash = sample["hash"]
        source_path = resolve_image_path(args.image_root, sample["image_path"])
        source = Image.open(source_path).convert("RGB")
        for seed in args.seeds:
            for method in method_order:
                completed += 1
                key = (sample_hash, seed, method.value)
                if key in done:
                    continue
                instruction = instructions[sample_hash][method.value]
                output_path = args.output_dir / "images" / method.value / f"{sample_hash}_s{seed}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                generator = torch.Generator(device=args.device).manual_seed(seed)
                edited = pipeline(
                    image=source,
                    prompt=instruction,
                    num_inference_steps=28,
                    guidance_scale=2.5,
                    generator=generator,
                ).images[0]
                edited.save(output_path)
                manifest.append(
                    {
                        "hash": sample_hash,
                        "seed": seed,
                        "method": method.value,
                        "condition": sample["kg_condition"],
                        "instruction": instruction,
                        "input_caption": sample["input_caption"],
                        "output_caption": sample["output_caption"],
                        "source_image": str(source_path),
                        "edited_image": str(output_path),
                    }
                )
                done.add(key)
                atomic_write_json(manifest_path, manifest)
                print(f"[edit {completed:04d}/{total}] seed={seed} {method.value} {sample_hash[:12]}")


if __name__ == "__main__":
    main()
