from __future__ import annotations

import unittest

from src.task1.class_mapping import canonical_task1_class_from_model_name


class Task1ClassMappingTests(unittest.TestCase):
    def test_explicit_2026_class_aliases_are_mapped(self) -> None:
        self.assertEqual(canonical_task1_class_from_model_name("vehicle"), 0)
        self.assertEqual(canonical_task1_class_from_model_name("Ta\u015f\u0131t"), 0)
        self.assertEqual(canonical_task1_class_from_model_name("person"), 1)
        self.assertEqual(canonical_task1_class_from_model_name("\u0130nsan"), 1)
        self.assertEqual(canonical_task1_class_from_model_name("uap_alan\u0131"), 2)
        self.assertEqual(canonical_task1_class_from_model_name("uai area"), 3)

    def test_ambiguous_aircraft_names_are_not_mapped_to_uap_uai(self) -> None:
        self.assertIsNone(canonical_task1_class_from_model_name("airplane"))
        self.assertIsNone(canonical_task1_class_from_model_name("aeroplane"))


if __name__ == "__main__":
    unittest.main()
