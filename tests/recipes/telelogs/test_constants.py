import unittest

from recipes.telelogs.constants import normalize_root_causes, set_f1


class RootCauseNormalizationTest(unittest.TestCase):
    def test_official_ids_and_multiple_labels(self):
        self.assertEqual(normalize_root_causes("C4, C5"), ["C4", "C5"])

    def test_zero_based_choice_index(self):
        shuffled_choices = [
            "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
            "Test vehicle speed exceeds 40km/h, impacting user throughput.",
        ]
        self.assertEqual(normalize_root_causes(0, choices=shuffled_choices), ["C2"])
        self.assertEqual(normalize_root_causes(1, choices=shuffled_choices), ["C1"])

    def test_description_matching(self):
        self.assertEqual(normalize_root_causes("PCI values collide mod 30"), ["C5"])

    def test_set_f1(self):
        self.assertAlmostEqual(set_f1(["C4", "C5"], ["C4"]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
