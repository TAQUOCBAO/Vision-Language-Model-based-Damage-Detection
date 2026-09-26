"""Description normalization tests (R7)."""

from __future__ import annotations

from damage_vlm.data.normalize_desc import normalize_description
from damage_vlm.data.silver_label import labels_from_text


def test_normalize_keeps_closed_set_keywords() -> None:
    cats = ["cracks", "corrosion"]
    out = normalize_description("Vertical cracks on the column.", cats)
    found = set(labels_from_text(out))
    assert "cracks" in found
    assert "corrosion" in found
    # Should not invent unrelated rare labels
    assert "efflorescence" not in found
