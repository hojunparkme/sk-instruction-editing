"""Inspect deterministic cue retrieval for one or more user requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sk_editing.retrieval import StructuredRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", nargs="+", help="Editing requests to inspect")
    parser.add_argument("--repository", type=Path, default=Path("data/structured_repository.json"))
    args = parser.parse_args()
    repository = StructuredRepository.from_path(args.repository)
    for request in args.requests:
        result = repository.retrieve(request)
        print(json.dumps({"request": request, **result.as_dict()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
