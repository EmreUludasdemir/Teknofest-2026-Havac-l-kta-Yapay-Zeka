from __future__ import annotations

import unicodedata

_TURKISH_ASCII = str.maketrans(
    {
        "\u00e7": "c",
        "\u011f": "g",
        "\u0131": "i",
        "\u00f6": "o",
        "\u015f": "s",
        "\u00fc": "u",
        "\u0130": "i",
    }
)

TASK1_2026_CLASS_ALIASES: dict[int, frozenset[str]] = {
    0: frozenset(
        {
            "tasit",
            "vehicle",
            "car",
            "bicycle",
            "motorcycle",
            "motorbike",
            "bus",
            "truck",
            "train",
            "boat",
            "arac",
            "otomobil",
            "minibus",
            "kamyon",
            "otobus",
        }
    ),
    1: frozenset({"insan", "person", "human", "pedestrian"}),
    2: frozenset({"uap", "uap_alani", "uap_area", "landing_uap"}),
    3: frozenset({"uai", "uai_alani", "uai_area", "landing_uai"}),
}


def normalize_task1_model_label(class_name: str) -> str:
    normalized = class_name.strip().lower().translate(_TURKISH_ASCII)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return normalized.replace("-", "_").replace(" ", "_")


def canonical_task1_class_from_model_name(class_name: str) -> int | None:
    normalized = normalize_task1_model_label(class_name)
    for class_id, aliases in TASK1_2026_CLASS_ALIASES.items():
        if normalized in aliases:
            return class_id
    return None
