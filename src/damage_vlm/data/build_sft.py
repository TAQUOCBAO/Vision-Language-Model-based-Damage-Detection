"""Build LLaMA-Factory-compatible multimodal SFT rows with JSON assistant targets."""

from __future__ import annotations

import json
from typing import Any, Iterable

from damage_vlm.config import closed_labels, load_prompts_config


def build_user_prompt(image_id: str, labels: list[str] | None = None) -> str:
    prompts = load_prompts_config()
    closed = ", ".join(closed_labels())
    template = str(prompts["user_template"])
    # Avoid str.format — prompt JSON examples contain curly braces.
    text = (
        template.replace("{closed_labels}", closed).replace("{image_id}", image_id)
    )
    # LLaMA-Factory multimodal templates require one <image> token per image path.
    if "<image>" not in text:
        text = "<image>\n" + text.lstrip()
    return text


def build_official_user_prompt(image_id: str) -> str:
    """User prompt that embeds the organizer Q1/Q2 template verbatim."""
    prompts = load_prompts_config()
    closed = ", ".join(closed_labels())
    q1 = str(prompts["official_q1"]).strip()
    q2 = str(prompts["official_q2"]).strip()
    template = str(prompts["official_user_template"])
    text = (
        template.replace("{official_q1}", q1)
        .replace("{official_q2}", q2)
        .replace("{closed_labels}", closed)
        .replace("{image_id}", image_id)
    )
    if "<image>" not in text:
        text = "<image>\n" + text.lstrip()
    if q1 not in text or q2 not in text:
        raise ValueError("Official prompt template lost Q1 or Q2")
    return text



def build_assistant_json(
    image_id: str,
    damage_categories: Iterable[str],
    description: str,
) -> str:
    payload = {
        "image_id": image_id,
        "damage_categories": list(damage_categories),
        "description": description,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_sft_row(
    *,
    image_id: str,
    image_path: str,
    damage_categories: list[str],
    description: str,
) -> dict[str, Any]:
    """Return one ShareGPT-style multimodal conversation record."""
    system = str(load_prompts_config()["system"]).strip()
    user = build_user_prompt(image_id)
    assistant = build_assistant_json(image_id, damage_categories, description)
    return {
        "image_id": image_id,
        "images": [image_path],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "damage_categories": list(damage_categories),
        "description": description,
    }
