"""Small, dependency-light I/O helpers used by the experiment scripts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Write JSON atomically so interrupted runs do not corrupt checkpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_emu_samples(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        raise ValueError(f"Expected a JSON object with a 'samples' list: {path}")
    return samples


def load_selected_samples(
    annotations_path: str | Path,
    sample_ids_path: str | Path,
) -> list[dict[str, Any]]:
    samples = load_emu_samples(annotations_path)
    selection = load_json(sample_ids_path)
    hashes = selection.get("hashes") if isinstance(selection, dict) else None
    if not isinstance(hashes, list):
        raise ValueError(f"Expected a JSON object with a 'hashes' list: {sample_ids_path}")
    lookup = {sample["hash"]: sample for sample in samples}
    missing = [sample_hash for sample_hash in hashes if sample_hash not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} selected hashes are absent from the annotations")
    return [lookup[sample_hash] for sample_hash in hashes]


def resolve_image_path(image_root: str | Path, relative_path: str | Path) -> Path:
    """Resolve an annotation path against an explicit image root."""
    relative_path = Path(relative_path)
    if relative_path.is_absolute():
        path = relative_path
    else:
        path = Path(image_root) / relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            "Pass --image-root pointing to the directory that contains the annotation paths."
        )
    return path


def require_files(paths: Iterable[str | Path], *, label: str = "Required file") -> None:
    missing = [str(Path(path)) for path in paths if not Path(path).exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"{label}(s) missing:\n  - {joined}")
