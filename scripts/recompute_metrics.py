"""Compute CLIPdir, CLIPout, CLIPim, L1, and DINO for a generated run.

The directional metric uses the same sample-specific Emu Edit source and target
captions for every method. This avoids scoring each method against its own
instruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sk_editing.io import atomic_write_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Manifest produced by a generation script")
    parser.add_argument("--output", type=Path, help="Defaults to metrics.json beside the manifest")
    parser.add_argument("--captions", type=Path, default=Path("data/clipout_captions.json"))
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.manifest.with_name("metrics.json")
    records = load_json(args.manifest)
    target_captions = load_json(args.captions)["captions"]
    existing = load_json(output_path) if output_path.exists() else []
    done = {(row["hash"], row.get("seed"), row["method"]) for row in existing}

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    clip = CLIPModel.from_pretrained(args.clip_model).to(args.device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    dino = AutoModel.from_pretrained(args.dino_model).to(args.device).eval()
    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)

    @torch.no_grad()
    def clip_image(image: Image.Image) -> torch.Tensor:
        inputs = clip_processor(images=image, return_tensors="pt").to(args.device)
        features = clip.get_image_features(**inputs)
        return functional.normalize(features, dim=-1)[0]

    @torch.no_grad()
    def clip_text(text: str) -> torch.Tensor:
        inputs = clip_processor(
            text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(args.device)
        features = clip.get_text_features(**inputs)
        return functional.normalize(features, dim=-1)[0]

    image_cache: dict[str, torch.Tensor] = {}
    dino_cache: dict[str, torch.Tensor] = {}
    text_direction_cache: dict[tuple[str, str], torch.Tensor | None] = {}
    target_text_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def dino_feature(image: Image.Image, cache_key: str | None = None) -> torch.Tensor:
        if cache_key is not None and cache_key in dino_cache:
            return dino_cache[cache_key]
        inputs = dino_processor(images=image, return_tensors="pt").to(args.device)
        feature = dino(**inputs).last_hidden_state[:, 0]
        feature = functional.normalize(feature, dim=-1)[0]
        if cache_key is not None:
            dino_cache[cache_key] = feature
        return feature

    pending = [
        row
        for row in records
        if (row["hash"], row.get("seed"), row["method"]) not in done
    ]
    for index, row in enumerate(pending, 1):
        source_path = Path(row["source_image"])
        edited_path = Path(row["edited_image"])
        if not source_path.exists() or not edited_path.exists():
            raise FileNotFoundError(f"Missing source or edited image for {row['hash']}")

        source = Image.open(source_path).convert("RGB")
        edited = Image.open(edited_path).convert("RGB")
        source_key = str(source_path.resolve())
        if source_key not in image_cache:
            image_cache[source_key] = clip_image(source)
        source_embedding = image_cache[source_key]
        edited_embedding = clip_image(edited)

        condition = row["condition"]
        target_caption = target_captions[condition]
        if condition not in target_text_cache:
            target_text_cache[condition] = clip_text(target_caption)
        target_embedding = target_text_cache[condition]

        caption_pair = (row["input_caption"], row["output_caption"])
        if caption_pair not in text_direction_cache:
            if caption_pair[0].strip() == caption_pair[1].strip():
                text_direction_cache[caption_pair] = None
            else:
                direction = clip_text(caption_pair[1]) - clip_text(caption_pair[0])
                text_direction_cache[caption_pair] = functional.normalize(direction, dim=0)
        text_direction = text_direction_cache[caption_pair]

        image_direction = edited_embedding - source_embedding
        if float(image_direction.norm()) < 1e-8 or text_direction is None:
            clip_dir = None
        else:
            image_direction = functional.normalize(image_direction, dim=0)
            clip_dir = float((image_direction * text_direction).sum())

        clip_out = float((edited_embedding * target_embedding).sum())
        clip_im = float((source_embedding * edited_embedding).sum())
        source_array = np.asarray(source.resize((512, 512)), dtype=np.float32) / 255.0
        edited_array = np.asarray(edited.resize((512, 512)), dtype=np.float32) / 255.0
        l1 = float(np.abs(source_array - edited_array).mean())
        source_dino = dino_feature(source, source_key)
        edited_dino = dino_feature(edited)
        dino_score = float((source_dino * edited_dino).sum())

        existing.append(
            {
                **row,
                "clip_dir_common": None if clip_dir is None else round(clip_dir, 6),
                "clip_out": round(clip_out, 6),
                "clip_im": round(clip_im, 6),
                "l1": round(l1, 6),
                "dino": round(dino_score, 6),
            }
        )
        done.add((row["hash"], row.get("seed"), row["method"]))
        if index % args.checkpoint_every == 0 or index == len(pending):
            atomic_write_json(output_path, existing)
        print(f"[{index:04d}/{len(pending)}] {row['method']} {row['hash'][:12]}")

    print(f"Wrote {len(existing)} metric records to {output_path}")


if __name__ == "__main__":
    main()
