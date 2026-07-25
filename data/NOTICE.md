# Data licensing and attribution

The repository `LICENSE` covers source code only. Data and result records retain the terms described below.

## Emu Edit annotations

**File:** `data/emu_edit_weather_final.json`

This file is a derivative of the Emu Edit test set released by Meta AI.

- Source: `facebook/emu_edit_test_set` on Hugging Face
- Paper: S. Sheynin et al., “Emu Edit: Precise Image Editing via Recognition and Generation Tasks,” CVPR 2024.
- License: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

The derivative may not be used for commercial purposes unless separate permission or alternative licensing is obtained.

### Modifications

1. The release was filtered to 136 environmental-condition samples: rainy, snowy, foggy, night, and clear.
2. `kg_condition` and `all_conditions` were added as project annotations.
3. The original `idx`, `hash`, `instruction`, `input_caption`, and `output_caption` fields were retained.
4. `image_path` is a local relative reference used by the experiment scripts.

### Image redistribution scope

The repository includes one composite manuscript panel, `assets/qualitative_results.png`, containing two Emu Edit source images and the corresponding generated outputs for non-commercial research demonstration. The panel is reproduced unchanged from the manuscript and remains subject to the Emu Edit **CC BY-NC 4.0** terms.

No standalone source-image collection is included. Obtain the full source images from the original Emu Edit release and arrange them so the paths in the annotation file can be resolved by `--image-root`.

The exact experimental subsets are identified by:

- `data/flux_sample_ids.json`
- `data/ip2p_sample_ids.json`

## Project-authored files

The following are project-authored materials:

| File | Description |
|---|---|
| `data/structured_repository.json` | 21 condition entries and 498 visual cues |
| `data/clipout_captions.json` | Fixed per-condition CLIPout captions |
| `data/repository_stats.json` | Repository summary metadata |
| `data/excluded_samples.json` | CLIPdir validity counts and notes |
| `docs/prompts.md` | Prompt templates and recovered repository-construction prompt |

## Result records

`results/raw/flux_results.json` and `results/raw/ip2p_results.json` contain metric values and generated instruction text from runs over the Emu Edit evaluation samples. They do not contain source or generated image pixels. Because the records are derived from an evaluation built on Emu Edit annotations, downstream users should conservatively retain the CC BY-NC attribution and non-commercial restriction when redistributing derived datasets.

## Third-party components

The following are not redistributed and retain their original licenses:

- FLUX.1 Kontext Dev
- InstructPix2Pix
- MGIE and its LLaVA checkpoint
- LLaVA-1.5
- DeepSeek-R1-Distill-Qwen-32B
- CLIP
- DINOv2

BDD100K and ACDC were conceptual references for selecting condition categories. No images, annotations, or dataset statistics from either dataset are included.
