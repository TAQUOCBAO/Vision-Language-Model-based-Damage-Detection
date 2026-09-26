"""METEOR scoring with NLTK backend (optional Java later)."""

from __future__ import annotations

from typing import Sequence

from damage_vlm.config import load_eval_config


def meteor_mean(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses length mismatch")
    if not references:
        return 0.0

    backend = load_eval_config().get("meteor", {}).get("backend", "nltk")
    if backend != "nltk":
        raise NotImplementedError(
            f"METEOR backend '{backend}' is not implemented; use 'nltk' or extend meteor_score.py"
        )

    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk import word_tokenize
    except ImportError as exc:
        raise RuntimeError(
            "nltk is required for METEOR. Install nltk>=3.9.1 and ensure WordNet data is available."
        ) from exc

    # Ensure WordNet is present
    try:
        import nltk

        nltk.data.find("corpora/wordnet")
    except LookupError:
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    scores: list[float] = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = word_tokenize(ref.lower())
        hyp_tokens = word_tokenize(hyp.lower())
        scores.append(float(meteor_score([ref_tokens], hyp_tokens)))
    return float(sum(scores) / len(scores))
