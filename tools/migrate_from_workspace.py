"""
Move the experiment files from the original working directory into this repo.

Run this once, on the machine that holds /workspace/hojun. It copies only the
files the paper depends on, rewrites the hard-coded path constants to read from
config.py, and strips absolute paths (which contain a username) out of the
result records.

It does not touch experiment logic.

    python tools/migrate_from_workspace.py --source /workspace/hojun --dry-run
    python tools/migrate_from_workspace.py --source /workspace/hojun

Files that are deliberately left behind: the art-style experiments (not part of
the paper), superseded script versions, generated images, and model weights.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# source relative path -> destination in the repo
COPY = {
    "run_gen_data1_v6.py":            "src/run_gen_data1_v6.py",
    "run_comparison_ip2p_v4.py":      "src/run_comparison_ip2p_v4.py",
    "recompute_weather_5metrics.py":  "src/recompute_weather_5metrics.py",
    "make_fig3_final.py":             "figures/_make_fig3_original.py",
    "emu_edit_weather_final.json":    "data/emu_edit_weather_final.json",
    "weather_knowledge_graph_v2.json":"data/structured_repository.json",
    "results_emu_edit_v6/results.json":        "results/flux_results.json",
    "results_comparison_ip2p_v4/results.json": "results/ip2p_results.json",
}

# left behind on purpose, listed so the decision is visible
SKIP = [
    "run_gen_data1.py", "run_comparison_ip2p_v3.py",          # superseded
    "run_art_style_v1.py", "run_art_style_v3.py",             # art domain: not in the paper
    "run_art_style_ip2p_v1.py", "run_art_style_ip2p_v2.py",
    "more_art_cases.py", "art_style_knowledge_graph_v2.json",
    "art_style_test_dataset_v3.json",
]

# hard-coded constants -> config lookups
PATCHES = [
    (r'BASE\s*=\s*Path\("/workspace/hojun"\)',
     "from config import (  # noqa: E402\n"
     "    ROOT as BASE, EMU_PATH, REPOSITORY_PATH, FLUX_SAMPLE_IDS,\n"
     "    LLAVA_PATH, FLUX_PATH, IP2P_PATH, MGIE_CODE, MGIE_LLAVA, MGIE_CKPT,\n"
     "    FLUX_OUT, IP2P_OUT, DEVICE,\n"
     ")"),
    (r'EMU_PATH\s*=\s*BASE\s*/\s*"emu_edit_weather_final\.json"', ""),
    (r'DATA_PATH\s*=\s*BASE\s*/\s*"emu_edit_weather_final\.json"', "DATA_PATH = EMU_PATH"),
    (r'KG_PATH\s*=\s*BASE\s*/\s*"weather_knowledge_graph_v2\.json"', "KG_PATH = REPOSITORY_PATH"),
    (r'LLAVA_PATH\s*=\s*BASE\s*/\s*"models/llava-1\.5-7b"', ""),
    (r'FLUX_PATH\s*=\s*BASE\s*/\s*"models/flux-kontext-dev"', ""),
    (r'IP2P_PATH\s*=\s*BASE\s*/\s*"models/instruct-pix2pix"', ""),
    (r'OUTPUT_DIR\s*=\s*BASE\s*/\s*"results_emu_edit_v6"', "OUTPUT_DIR = FLUX_OUT"),
    (r'OUTPUT_DIR\s*=\s*BASE\s*/\s*"results_comparison_ip2p_v4"', "OUTPUT_DIR = IP2P_OUT"),
    (r'RESULTS_DIR\s*=\s*BASE\s*/\s*"results_emu_edit_v6"', "RESULTS_DIR = FLUX_OUT"),
    (r'MGIE_CODE\s*=\s*"[^"]*"', "MGIE_CODE = str(MGIE_CODE)"),
    (r'PATH_LLAVA\s*=\s*"[^"]*"', "PATH_LLAVA = str(MGIE_LLAVA)"),
    (r'PATH_MGIE_CKPT\s*=\s*"[^"]*"', "PATH_MGIE_CKPT = str(MGIE_CKPT)"),
]

# the v4 dependency: replace the implicit sample filter with the explicit ID list
V4_OLD = re.compile(
    r'V4_PATH\s*=\s*BASE\s*/\s*"results_emu_edit_v4/results\.json".*?'
    r'samples\s*=\s*\[s for s in samples if s\["hash"\] in v4_hashes\]',
    re.S)
V4_NEW = ('with open(FLUX_SAMPLE_IDS, encoding="utf-8") as f:\n'
          '    _keep = set(json.load(f)["hashes"])\n'
          'samples = [s for s in samples if s["hash"] in _keep]')


def patch_source(text: str) -> tuple[str, list[str]]:
    notes = []
    if V4_OLD.search(text):
        text = V4_OLD.sub(V4_NEW, text)
        notes.append("sample filter now reads data/flux_sample_ids.json")
    for pat, sub in PATCHES:
        new, k = re.subn(pat, sub, text)
        if k:
            text = new
            notes.append(f"patched {pat.split('=')[0].strip()}")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, notes


def scrub_results(path: Path) -> int:
    """Drop absolute image paths from result records."""
    with open(path) as f:
        rows = json.load(f)
    removed = 0
    for r in rows:
        for k in [k for k in list(r) if k.endswith("_image_path")]:
            r.pop(k)
            removed += 1
        for k, v in list(r.items()):
            if isinstance(v, str) and v.startswith("/"):
                r[k] = Path(v).name
                removed += 1
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"source: {args.source}\nrepo:   {REPO}\n")
    copied = []
    for rel, dest_rel in COPY.items():
        src, dest = args.source / rel, REPO / dest_rel
        if not src.exists():
            print(f"  MISSING  {rel}")
            continue
        print(f"  copy     {rel}  ->  {dest_rel}")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(dest)

    print("\n  left behind on purpose:")
    for rel in SKIP:
        if (args.source / rel).exists():
            print(f"    - {rel}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    print("\npatching paths:")
    for dest in copied:
        if dest.suffix != ".py":
            continue
        text = dest.read_text(encoding="utf-8")
        text, notes = patch_source(text)
        dest.write_text(text, encoding="utf-8")
        print(f"  {dest.name}: {', '.join(notes) if notes else 'no change'}")

    print("\nscrubbing absolute paths from results:")
    for name in ["results/flux_results.json", "results/ip2p_results.json"]:
        p = REPO / name
        if p.exists():
            print(f"  {name}: {scrub_results(p)} field(s)")

    print("\nnext:")
    print("  grep -rn '/workspace' src/ results/ data/     # should print nothing")
    print("  python analysis/statistics.py results/flux_results.json results/ip2p_results.json")


if __name__ == "__main__":
    main()
