# Data licensing and attribution

The `LICENSE` file at the root of this repository covers **code only**. The
files in `data/` and `results/` are governed by the terms below.

---

## Emu Edit annotations

**File:** `data/emu_edit_weather_final.json`

This file is a derivative of the **Emu Edit test set**, released by Meta AI.

- Source: https://huggingface.co/datasets/facebook/emu_edit_test_set
- Paper: S. Sheynin et al., "Emu Edit: Precise Image Editing via Recognition and
  Generation Tasks," CVPR 2024.
- **License: Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0)** — https://creativecommons.org/licenses/by-nc/4.0/

**This means the file may not be used for commercial purposes.**

### Changes made to the original

As required by the attribution clause, the modifications are:

1. **Filtered.** Only environmental-condition samples (rainy, snowy, foggy,
   night, clear) were retained: 136 of the original examples.
2. **Fields added.** `kg_condition` and `all_conditions` are our own annotations
   labelling the target condition of each sample; they are not part of the
   original release.
3. **Fields carried over unchanged** from Emu Edit: `idx`, `hash`,
   `instruction`, `input_caption`, `output_caption`.

### What is *not* redistributed

No images from the Emu Edit test set are included in this repository. The
`image_path` field is a local filename reference only. To run the generation
scripts you must obtain the images from the original release above.

The sample ID lists (`data/flux_sample_ids.json`, `data/ip2p_sample_ids.json`)
contain only the `hash` identifiers needed to reconstruct our subsets.

---

## Our own contributions

The following are our own work and are covered by the repository `LICENSE`:

| File | Notes |
|---|---|
| `data/structured_repository.json` | The structured knowledge repository (21 condition entries, 498 cues), constructed as described in the paper |
| `data/clipout_captions.json` | The five fixed per-condition CLIPout target captions |
| `data/repository_stats.json`, `data/excluded_samples.json` | Summary metadata |
| `docs/prompts.md` | Prompt templates |

`results/flux_results.json` and `results/ip2p_results.json` contain metric values
we measured. They were produced by running models over the Emu Edit test images,
so if you build further datasets from them, apply the CC BY-NC 4.0 terms above
to be safe.

---

## Third-party components (not redistributed here)

Each keeps its own license; obtain them from the original source.

| Component | Source |
|---|---|
| MGIE | Apple — https://github.com/apple/ml-mgie |
| FLUX.1 Kontext Dev | Black Forest Labs |
| InstructPix2Pix | Brooks et al. |
| LLaVA-1.5 | Liu et al. |
| DeepSeek-R1-Distill-Qwen-32B | DeepSeek |
| CLIP, DINOv2 | OpenAI, Meta |
| BDD100K, ACDC | Used only as conceptual references for the condition categories; no data from either is included |

---

## Summary

- **Code** (`src/`, `analysis/`, `figures/`, `tools/`, `config.py`) → repository `LICENSE`
- **Emu Edit derivative** (`data/emu_edit_weather_final.json`) → **CC BY-NC 4.0, non-commercial only**
- **Model weights and third-party code** → not included; see their own terms

If you intend to use this repository commercially, remove
`data/emu_edit_weather_final.json` and obtain the Emu Edit data under terms that
permit your use. Everything else in `data/` is our own work.
