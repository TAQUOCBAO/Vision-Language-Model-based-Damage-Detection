"""Silver labeling tests including AE1 multi-damage case."""

from __future__ import annotations

from damage_vlm.data.silver_label import silver_labels_for_image


def test_ae1_multi_damage_spalling_rebar_corrosion() -> None:
    description = (
        "Extensive concrete spalling has occurred on the structural surface, "
        "exposing the underlying aggregate and load-bearing rebar, "
        "accompanied by severe corrosion of the rebar."
    )
    labels = silver_labels_for_image("image/lou_0001.jpg", description)
    assert "spalling" in labels
    assert "exposed_rebar" in labels
    assert "corrosion" in labels


def test_prefix_voids_plus_text_spalling() -> None:
    description = (
        "The concrete surface shows numerous fine holes, and in some areas "
        "larger voids or spalling are evident."
    )
    labels = silver_labels_for_image("image/kong_0063.jpg", description)
    assert "voids" in labels
    assert "spalling" in labels


def test_fissure_maps_to_cracks() -> None:
    labels = silver_labels_for_image(
        "image/1.jpg",
        "A fissure extends across the concrete surface.",
    )
    assert "cracks" in labels
