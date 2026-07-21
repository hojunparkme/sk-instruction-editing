# Training-Free Structured Knowledge-Augmented Instruction Generation for Image Editing

Code and data for the paper. The pipeline expands a short, underspecified
editing request into a detailed, scene-grounded instruction using a structured
knowledge repository, and passes that instruction to an off-the-shelf editing
backbone. Nothing is fine-tuned.

```
user request q ──┐
                 ├─► knowledge cue retrieval ──┐
input image I ───┴─► VLM scene description ────┴─► LLM ──► instruction ──► editor
```

## What is here

| Path | Contents |
|---|---|
| `data/structured_repository.json` | The knowledge repository: 21 condition entries, 5 typed slots, 498 cues |
| `data/flux_sample_ids.json` | The 110 sample IDs used in Table 2 |
| `data/ip2p_sample_ids.json` | The 128 sample IDs used in Table 3 |
| `data/clipout_captions.json` | The five fixed per-condition CLIPout captions |
| `src/` | Experiment scripts, unchanged from the runs that produced the reported numbers |
| `analysis/statistics.py` | Every paired statistic reported in the paper |
| `results/` | Per-sample metric records behind Tables 2 and 3 |
| `figures/` | Figure generation |
| `docs/prompts.md` | All prompt templates, verbatim |

**Not included.** MGIE is third-party code and is not vendored here; clone it
separately (below). Generated images are not redistributed. Emu Edit source
images are not redistributed — we ship sample IDs so you can assemble the same
subset from the original release.

## Setup

```bash
git clone <repo-url> && cd sk-instruction-editing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then edit `config.py`, or export the paths you need:

```bash
export MODEL_ROOT=/path/to/models      # llava-1.5-7b, flux-kontext-dev, instruct-pix2pix
export SK_RESULTS=/path/to/results     # generated images are large
```

A single 48 GB GPU is enough. The generation steps need it; every recompute and
analysis step runs on CPU (`export DEVICE=cpu`).

## Reproducing the paper

### The cheap part: statistics and figures

The per-sample metric records are included, so every number in the paper can be
checked without re-running any model:

```bash
python analysis/statistics.py results/flux_results.json results/ip2p_results.json
```

Expected (abridged):

```
FLUX  clip_dir_common
  SK+LLM vs LLM-only    N=107  diff=+0.0250  95% CI [+0.0127, +0.0379]  p=0.002158
  SK+LLM vs Simple      N=107  diff=+0.0245  95% CI [+0.0126, +0.0364]  p=0.000382
  SK+LLM vs SK+Filter   N=107  diff=+0.0018  95% CI [-0.0077, +0.0114]  p=0.699945
FLUX  clip_out
  SK+LLM vs LLM-only    N=110  diff=+0.0219  95% CI [+0.0153, +0.0289]  p<0.000001
IP2P  all pairwise, clip_dir_common
  significant pairs at alpha=0.05: 0/10
Matched subset (102 shared samples, 99 valid for CLIPdir)
  FLUX  SK+LLM vs LLM-only  diff=+0.0274  p=0.000961
  IP2P  SK+LLM vs LLM-only  diff=-0.0033  p=0.552814
```

Seed handling matters. For FLUX the three seed-level scores of each sample are
averaged **first**, and the paired tests then run over samples (N=107 for
CLIPdir, N=110 for CLIPout). Treating the 330 seed-level rows as independent
observations narrows the confidence intervals and understates the p-values.

Inspect retrieval on its own:

```bash
python src/retrieval.py          # matched entries and cues for a few example requests
```

### The expensive part: regenerating everything

**FLUX Kontext (Table 2)**

```bash
python src/run_gen_data1_v6.py             # 1. instructions + edited images, 3 seeds
python src/recompute_weather_5metrics.py   # 2. CLIPout, CLIPim, L1, DINO
python src/recompute_clipdir_common.py weather   # 3. shared-reference CLIPdir
```

**InstructPix2Pix and MGIE (Table 3)**

```bash
python src/run_comparison_ip2p_v4.py            # needs the MGIE checkout, see below
python src/recompute_ip2p_weather_common.py     # shared-reference CLIPdir
```

Step 3 in each case is what produces the `*_clip_dir_common` fields. The
original `clip_dir` field scores each method against **its own** generated
instruction, which favours methods that emit longer, more cue-laden text; the
paper reports the shared-reference version instead, in which every method is
scored against the same Emu Edit source and target captions.

**Figures**

```bash
python figures/make_fig1.py                          # pipeline diagram (PNG + PDF)
python figures/make_fig2.py                          # CLIPdir bar charts
python figures/make_fig3.py --cell_size 640          # qualitative grid
```

### MGIE

MGIE is compared under IP2P only. It is not redistributed here:

```bash
git clone https://github.com/apple/ml-mgie $MODEL_ROOT/ml-mgie-code
# then fetch the official LLaVA-7B-v1 weights and mgie_7b/mllm.pt checkpoint
export MGIE_CODE=$MODEL_ROOT/ml-mgie-code
```

Apple's license applies to that code. If you skip MGIE, the IP2P script still
runs the other four methods.

## Data

`data/emu_edit_weather_final.json` is the environmental-condition subset of the
Emu Edit test set: instructions that transform a photograph to rainy, snowy or
foggy conditions, to a night/low-light setting, or back to a clear daytime
setting.

`data/emu_edit_weather_final.json` is a derivative of that release: filtered to
the 136 environmental-condition samples and extended with our own
`kg_condition` / `all_conditions` labels. It carries the original **CC BY-NC 4.0**
license — **non-commercial use only** — and contains annotations only, no images.
See `data/NOTICE.md`.

Source images are not redistributed. Obtain them from
https://huggingface.co/datasets/facebook/emu_edit_test_set and place them where
`config.EMU_PATH` expects; `flux_sample_ids.json` and `ip2p_sample_ids.json`
identify the exact subsets we used.

Selection rules, stated explicitly so they can be checked:

- **FLUX, 110 samples** (night 41, rainy 29, snowy 20, clear 15, foggy 5). The
  original script derived this set from an earlier run; `flux_sample_ids.json`
  makes the resulting list explicit.
- **IP2P, 128 samples** (rainy 32, night 45, snowy 25, clear 21, foggy 5) — all
  environmental-condition samples for which valid IP2P and MGIE outputs were
  available. Eight hashes are excluded in `run_comparison_ip2p_v4.py`.
- **CLIPdir exclusions.** Three FLUX samples have identical source and target
  captions and define no text-space direction, so CLIPdir uses 107 of 110; under
  IP2P, 99 of 128 define a valid direction. All other metrics use the full sets.

The two sets overlap in 102 samples; `analysis/statistics.py` reports the
matched-subset comparison built from that overlap.

## The knowledge repository

`data/structured_repository.json` is a nested structure keyed by condition. Each
entry holds five typed cue slots — global appearance, lighting, surfaces,
atmospheric effects, object-level details — plus aliases used for matching.
Retrieval is deterministic keyword and alias matching, collecting at most six
cues per slot and fifty in total.

Condition categories follow those common in adverse-condition driving datasets
(BDD100K, ACDC); those datasets were used only as conceptual references, and no
images, annotations or dataset statistics from them went into the repository.
Cue candidates were generated with a Claude Sonnet model and then reviewed by
the authors for visual plausibility, condition relevance and redundancy,
normalized into short noun phrases, and assigned to the slots. The generation
prompt is in `docs/prompts.md`.

## Citing

```bibtex
@article{TODO,
  title   = {Training-Free Structured Knowledge-Augmented Instruction Generation for Image Editing},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

## License

Licensing differs by directory:

| What | Terms |
|---|---|
| Code — `src/`, `analysis/`, `figures/`, `tools/`, `config.py` | MIT, see `LICENSE` |
| `data/emu_edit_weather_final.json` | **CC BY-NC 4.0** (Emu Edit derivative, non-commercial only) |
| Everything else in `data/`, and `docs/prompts.md` | Our own work, MIT |
| Model weights, MGIE | Not redistributed; see each project's terms |

`data/NOTICE.md` has the full attribution, the list of changes we made to the
Emu Edit annotations, and the third-party component table.

If you need commercial use, delete `data/emu_edit_weather_final.json` and obtain
the Emu Edit data under terms that permit it. Nothing else in the repository is
NonCommercial.
