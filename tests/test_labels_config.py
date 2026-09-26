"""Unit tests for V1 label / synonym / prefix configuration."""

from __future__ import annotations

from damage_vlm.config import (
    closed_labels,
    load_labels_config,
    prefix_seeds,
    synonyms_longest_first,
)


def test_closed_labels_exactly_ten() -> None:
    labels = closed_labels()
    assert len(labels) == 10
    assert labels == [
        "spalling",
        "cracks",
        "corrosion",
        "voids",
        "exposed_rebar",
        "peeling",
        "potholes",
        "honeycomb",
        "looseness",
        "efflorescence",
    ]


def test_exposed_rebar_synonyms_longest_first() -> None:
    syn = synonyms_longest_first()
    phrases = syn["exposed_rebar"]
    assert phrases[0] == "exposed rebar"
    assert "exposed" in phrases
    assert phrases.index("exposed rebar") < phrases.index("exposed")


def test_prefix_map_known_and_numeric() -> None:
    seeds = prefix_seeds()
    assert "crack" in seeds
    assert seeds["crack"] == ["cracks"]
    assert seeds["kong"] == ["voids"]
    assert seeds["xiu"] == ["corrosion"]
    assert "lou" in seeds
    assert "gangf" in seeds
    # Numeric / no-prefix filenames have no seed entry
    assert "00002" not in seeds
    assert "" not in seeds


def test_labels_config_loads() -> None:
    cfg = load_labels_config()
    assert cfg["version"] == "v1"
    assert "alignment_templates" in cfg
