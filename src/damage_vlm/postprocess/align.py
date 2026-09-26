"""Align description text with declared damage categories."""

from __future__ import annotations

from typing import Any, Iterable

from damage_vlm.config import alignment_templates, load_labels_config, synonyms_longest_first


def _label_mentioned(description: str, label: str, cfg: dict[str, Any]) -> bool:
    lowered = description.lower()
    phrases = synonyms_longest_first(cfg).get(label, [label])
    return any(p.lower() in lowered for p in phrases)


def align_description(
    description: str,
    categories: Iterable[str],
    cfg: dict[str, Any] | None = None,
) -> str:
    cfg = cfg or load_labels_config()
    templates = alignment_templates(cfg)
    text = description.strip()
    for label in categories:
        if _label_mentioned(text, label, cfg):
            continue
        template = templates.get(label)
        if template:
            text = f"{text} {template}".strip()
    return text
