"""Deterministic retrieval over the released structured visual repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_json

GROUP_ORDER = ("condition", "environment", "season", "time_of_day", "weather")
SLOT_ORDER = ("global", "lighting", "surfaces", "atmospheric_effects", "objects_details")


@dataclass(frozen=True)
class RetrievalResult:
    matched_entries: tuple[tuple[str, str], ...]
    cues: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched_entries": [
                {"group": group, "entry": entry} for group, entry in self.matched_entries
            ],
            "cues": list(self.cues),
        }


class StructuredRepository:
    """Typed condition-indexed cue store with keyword and alias matching."""

    def __init__(self, payload: dict[str, Any]):
        if not isinstance(payload, dict):
            raise TypeError("Repository payload must be a JSON object")
        self.payload = payload
        self._validate()

    @classmethod
    def from_path(cls, path: str | Path) -> "StructuredRepository":
        return cls(load_json(path))

    def _validate(self) -> None:
        missing = [group for group in GROUP_ORDER if group not in self.payload]
        if missing:
            raise ValueError(f"Repository is missing group(s): {missing}")
        for group in GROUP_ORDER:
            entries = self.payload[group]
            if not isinstance(entries, dict):
                raise TypeError(f"Repository group '{group}' must be a JSON object")
            for entry_name, node in entries.items():
                if not isinstance(node, dict):
                    raise TypeError(f"Repository entry '{group}/{entry_name}' must be an object")
                if not isinstance(node.get("positives", {}), dict):
                    raise TypeError(f"Repository entry '{group}/{entry_name}' has invalid positives")

    @staticmethod
    def _triggers(entry_name: str, node: dict[str, Any]) -> tuple[str, ...]:
        aliases = node.get("aliases", [])
        values = [entry_name, *aliases]
        return tuple(str(value).strip().lower() for value in values if str(value).strip())

    def match(self, request: str) -> tuple[tuple[str, str], ...]:
        text = request.lower()
        matches: list[tuple[str, str]] = []
        for group in GROUP_ORDER:
            for entry_name, node in self.payload[group].items():
                if any(trigger in text for trigger in self._triggers(entry_name, node)):
                    matches.append((group, entry_name))
        return tuple(matches)

    def entries_for_label(self, label: str) -> tuple[tuple[str, str], ...]:
        """Resolve a normalized condition label using names and aliases.

        This helper is used by the IP2P reference implementation because the
        archived experiment stored an author-reviewed target-condition label for
        every sample. The public script also checks request-based retrieval and
        warns when it resolves to a different entry set.
        """
        text = label.strip().lower()
        matches: list[tuple[str, str]] = []
        for group in GROUP_ORDER:
            for entry_name, node in self.payload[group].items():
                if text in self._triggers(entry_name, node):
                    matches.append((group, entry_name))
        return tuple(matches)

    def collect(
        self,
        entries: tuple[tuple[str, str], ...],
        *,
        max_per_slot: int = 6,
        max_total: int = 50,
    ) -> RetrievalResult:
        cues: list[str] = []
        seen: set[str] = set()
        for group, entry_name in entries:
            node = self.payload[group][entry_name]
            positives = node.get("positives", {})
            extra_slots = tuple(slot for slot in positives if slot not in SLOT_ORDER)
            for slot in (*SLOT_ORDER, *extra_slots):
                items = positives.get(slot, [])
                if not isinstance(items, list):
                    continue
                for raw_item in items[:max_per_slot]:
                    cue = str(raw_item).strip()
                    key = cue.casefold()
                    if cue and key not in seen:
                        cues.append(cue)
                        seen.add(key)
                        if len(cues) >= max_total:
                            return RetrievalResult(entries, tuple(cues))
        return RetrievalResult(entries, tuple(cues))

    def retrieve(
        self,
        request: str,
        *,
        max_per_slot: int = 6,
        max_total: int = 50,
    ) -> RetrievalResult:
        return self.collect(
            self.match(request), max_per_slot=max_per_slot, max_total=max_total
        )

    def retrieve_label(
        self,
        label: str,
        *,
        max_per_slot: int = 6,
        max_total: int = 50,
    ) -> RetrievalResult:
        return self.collect(
            self.entries_for_label(label),
            max_per_slot=max_per_slot,
            max_total=max_total,
        )

    def statistics(self) -> dict[str, Any]:
        entry_count = 0
        cue_count = 0
        slots: set[str] = set()
        for group in GROUP_ORDER:
            for node in self.payload[group].values():
                entry_count += 1
                positives = node.get("positives", {})
                for slot, items in positives.items():
                    slots.add(slot)
                    if isinstance(items, list):
                        cue_count += len(items)
        return {
            "groups": len(GROUP_ORDER),
            "entries": entry_count,
            "cues": cue_count,
            "slots": sorted(slots),
        }
