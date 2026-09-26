"""Local offline paraphrase helpers (callable LLM injected for testability)."""

from __future__ import annotations

from typing import Callable

PARAPHRASE_SYSTEM = (
    "You are a professional structural forensic engineer. "
    "Paraphrase the given damage description into a new, natural, technical English sentence. "
    "Strictly preserve technical keywords, damage categories, orientations, severity, and sizes. "
    "Return ONLY the paraphrased text."
)


def paraphrase_once(
    description: str,
    generate_fn: Callable[[str, str], str],
) -> str:
    """Generate one paraphrase using an injected local generator(system, user)->text."""
    user = f'Paraphrase this damage description:\n"{description}"'
    out = generate_fn(PARAPHRASE_SYSTEM, user).strip()
    if not out:
        return description
    return out


def paraphrase_n(
    description: str,
    n: int,
    generate_fn: Callable[[str, str], str],
) -> list[str]:
    results: list[str] = []
    for _ in range(n):
        results.append(paraphrase_once(description, generate_fn))
    return results


def identity_paraphrase_fn(system: str, user: str) -> str:
    """Deterministic stub for offline unit tests without a local LLM."""
    # Extract quoted description if present
    if '"' in user:
        start = user.find('"') + 1
        end = user.rfind('"')
        base = user[start:end]
    else:
        base = user
    return f"Technically restated: {base}"
