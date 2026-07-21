"""
Central paths and constants.

This is the only file you should need to edit to run the code on your own
machine. The experiment scripts under `src/` are kept as they were when the
reported numbers were produced; only their path constants are read from here.

Override any path with an environment variable of the same name, e.g.

    export SK_ROOT=/data/sk-editing
    export MODEL_ROOT=/data/models
"""
import os
from pathlib import Path

# ── Repository layout ────────────────────────────────────────────────────────
ROOT      = Path(os.environ.get("SK_ROOT", Path(__file__).resolve().parent))
DATA_DIR  = ROOT / "data"
RESULTS_DIR = Path(os.environ.get("SK_RESULTS", ROOT / "results"))
FIG_DIR   = ROOT / "figures"

# ── Input data ───────────────────────────────────────────────────────────────
# Emu Edit environmental-condition subset. See README for how to obtain it:
# we distribute the sample IDs, not the source images.
EMU_PATH        = DATA_DIR / "emu_edit_weather_final.json"

# Structured knowledge repository (21 condition entries, 5 slots, 498 cues)
REPOSITORY_PATH = DATA_DIR / "structured_repository.json"

# Explicit sample selections used in the paper
FLUX_SAMPLE_IDS = DATA_DIR / "flux_sample_ids.json"   # 110 samples, Table 2
IP2P_SAMPLE_IDS = DATA_DIR / "ip2p_sample_ids.json"   # 128 samples, Table 3

# Fixed per-condition target captions used for CLIPout
CLIPOUT_CAPTIONS = DATA_DIR / "clipout_captions.json"

# ── Model checkpoints ────────────────────────────────────────────────────────
# Local directories or Hugging Face IDs.
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", ROOT / "models"))

LLAVA_PATH = os.environ.get("LLAVA_PATH", str(MODEL_ROOT / "llava-1.5-7b"))
FLUX_PATH  = os.environ.get("FLUX_PATH",  str(MODEL_ROOT / "flux-kontext-dev"))
IP2P_PATH  = os.environ.get("IP2P_PATH",  str(MODEL_ROOT / "instruct-pix2pix"))
LLM_PATH   = os.environ.get("LLM_PATH",   "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
CLIP_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-large-patch14")
DINO_MODEL = os.environ.get("DINO_MODEL", "facebook/dinov2-base")

# MGIE is third-party code and is NOT vendored in this repository.
# Clone it separately (see README) and point these at your checkout.
MGIE_CODE      = os.environ.get("MGIE_CODE",      str(MODEL_ROOT / "ml-mgie-code"))
MGIE_LLAVA     = os.environ.get("MGIE_LLAVA",     str(MODEL_ROOT / "ml-mgie-official/LLaVA-7B-v1"))
MGIE_CKPT      = os.environ.get("MGIE_CKPT",      str(MODEL_ROOT / "ml-mgie-official/mgie_7b/mllm.pt"))

# ── Output directories ───────────────────────────────────────────────────────
FLUX_OUT = RESULTS_DIR / "flux"
IP2P_OUT = RESULTS_DIR / "ip2p"

# ── Experiment constants (as used for the reported numbers) ──────────────────
SEEDS_FLUX = [42, 123, 777]   # FLUX Kontext, three seeds
SEED_IP2P  = 42               # IP2P / MGIE, single seed

FLUX_STEPS, FLUX_GUIDANCE = 28, 2.5
IP2P_STEPS, IP2P_IMAGE_GUIDANCE, IP2P_TEXT_GUIDANCE = 50, 1.5, 7.5

MAX_CUES_PER_SLOT = 6
MAX_CUES_TOTAL    = 50

DEVICE = os.environ.get("DEVICE", "cuda")   # set to "cpu" for the recompute steps
