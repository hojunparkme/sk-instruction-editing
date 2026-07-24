# Reproducibility and release scope

## What can be reproduced immediately

The repository includes the sample-level metric records behind the reported FLUX and IP2P tables. The following commands require no image-generation model:

```bash
python analysis/statistics.py
python scripts/export_results.py
python scripts/self_check.py
```

These commands verify sample counts, seed handling, overall means, confidence intervals, Wilcoxon tests, the matched subset, and the absence of significant IP2P pairwise differences.

## What requires external assets

Regenerating edited images requires:

- Emu Edit source images, obtained from the original dataset release.
- FLUX.1 Kontext Dev.
- InstructPix2Pix.
- LLaVA-1.5-7B for the FLUX scene descriptions.
- DeepSeek-R1-Distill-Qwen-32B for instruction generation.
- The official MGIE code, LLaVA weights, and checkpoint for the IP2P configuration and MGIE baseline.

The source images, model weights, MGIE code, and generated images are not redistributed.

## Clean reference implementation versus archived outputs

`results/raw/` contains archived outputs from the original experiment workspace. The scripts under `scripts/` are cleaned reference implementations:

- workspace-specific absolute paths were removed;
- fragile JSON writes were replaced with atomic checkpoints;
- the released nested repository is handled explicitly;
- method names use `SK+Filter` and `SK+LLM` rather than legacy `KG` variable names;
- editor-specific prompt configurations are documented instead of being presented as identical.

Because model serving, library versions, quantization kernels, and hardware can affect generated text and images, a fresh run should not be expected to reproduce every instruction or pixel bit-for-bit. The included sample-level records are the source of truth for the reported numerical tables.

## Cross-backbone interpretation

The FLUX and IP2P experiments share the structured repository and comparison logic but use editor-specific VLM and prompt configurations. Cross-backbone results are therefore descriptive. The matched subset controls sample composition only.

## Archived metric precision

The sample-level JSON records store metrics to four decimal places. One displayed value sits on a third-decimal rounding boundary: the archived IP2P SK+Filter L1 records average to `0.183484`, whereas the manuscript table reports `0.184`. The full-precision values used before archival rounding are unavailable. No archived record has been altered to force agreement.
