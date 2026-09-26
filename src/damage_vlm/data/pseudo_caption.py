"""Pseudo-captioning prompt builder for external images with known labels."""

from __future__ import annotations


def build_pseudo_caption_prompt(known_labels: list[str]) -> str:
    labels = ", ".join(known_labels) if known_labels else "unknown damage"
    return (
        "Analyze this concrete structure image. We already know it contains the following "
        f"damage types: {labels}. "
        "Generate a concise, professional technical description in English describing the "
        "damage morphology, severity, and characteristics. Match the style of an official "
        "structural inspection report. Do not mention any undamaged parts."
    )
