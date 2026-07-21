"""
Knowledge cue retrieval — reference implementation.

Deterministic keyword/alias matching against the structured knowledge
repository. No learning, no embeddings.

Usage:
    kg = json.load(open(_P(__file__).resolve().parent.parent / "data" / "structured_repository.json"))
    cues = retrieve_candidates(kg, extract_keywords(kg, "make it rainy"))
"""
import json
from pathlib import Path as _P

SECTIONS   = ["condition", "environment", "season", "time_of_day", "weather"]
SLOT_ORDER = ["global", "lighting", "surfaces", "atmospheric_effects", "objects_details"]

MAX_PER_SLOT = 6
MAX_TOTAL    = 50


def extract_keywords(kg, text):
    """Match the user request against condition-entry names and their aliases."""
    text = text.lower()
    out = {s: [] for s in SECTIONS}
    for s in SECTIONS:
        for key, entry in kg.get(s, {}).items():
            triggers = [key.lower()] + [a.lower() for a in entry.get("aliases", [])]
            if any(t in text for t in triggers):
                out[s].append(key)
    return out


def retrieve_candidates(kg, keywords, max_total=MAX_TOTAL):
    """Collect cues from matched entries: <=6 per slot, <=50 total, de-duplicated."""
    candidates, seen = [], set()
    for section, keys in keywords.items():
        for key in keys:
            entry = kg.get(section, {}).get(key)
            if not isinstance(entry, dict):
                continue
            positives = entry.get("positives", {})
            slots = SLOT_ORDER + [s for s in positives if s not in SLOT_ORDER]
            for slot in slots:
                for item in positives.get(slot, [])[:MAX_PER_SLOT]:
                    c = item.strip() if isinstance(item, str) else str(item)
                    if c and c not in seen:
                        candidates.append(c)
                        seen.add(c)
    return candidates[:max_total]


if __name__ == "__main__":
    kg = json.load(open(_P(__file__).resolve().parent.parent / "data" / "structured_repository.json"))
    for req in ["make it rainy", "change it to night time", "make it a foggy morning"]:
        kws = extract_keywords(kg, req)
        cues = retrieve_candidates(kg, kws)
        matched = {k: v for k, v in kws.items() if v}
        print(f"\n[{req}]  matched entries: {matched}")
        print(f"  {len(cues)} cues, first 5: {cues[:5]}")
