from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from common.data import load_domain
from scripts.build_dataset import clean_data, parse_experience, split_languages


class DatasetBuilderTest(unittest.TestCase):
    def test_experience_and_language_parsing(self) -> None:
        self.assertEqual(parse_experience("Less than 1 year"), 0.5)
        self.assertEqual(parse_experience("More than 50 years"), 51.0)
        self.assertEqual(split_languages("Java;JavaScript;C++"), {"Java", "JavaScript", "C++"})

    def test_java_does_not_match_javascript(self) -> None:
        raw = pd.DataFrame(
            {
                "ResponseId": range(1, 101),
                "Country": ["South Korea"] * 100,
                "ConvertedCompYearly": range(10_000, 110_000, 1_000),
                "LanguageHaveWorkedWith": ["JavaScript"] * 50 + ["Java"] * 50,
                "YearsCodePro": ["5"] * 100,
            }
        )

        cleaned, _ = clean_data(raw, 2025, None)
        javascript_only = cleaned[
            cleaned["LanguageHaveWorkedWith"].eq("JavaScript")
        ]
        self.assertTrue(javascript_only["lang_Java"].eq(0).all())
        self.assertTrue(javascript_only["lang_JavaScript"].eq(1).all())

    def test_domain_loader_reads_the_single_shared_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "results.csv"
            metadata_path = root / "metadata.json"
            pd.DataFrame(
                {
                    "ResponseId": [1, 2, 3],
                    "lang_Java": [1, 0, 0],
                    "lang_Kotlin": [0, 1, 0],
                    "lang_Scala": [0, 0, 0],
                }
            ).to_csv(csv_path, index=False)
            metadata_path.write_text(
                json.dumps({"eligible_languages": ["Java", "Kotlin"]}),
                encoding="utf-8",
            )

            selected = load_domain(
                "domain2_jvm", csv_path, metadata_path=metadata_path
            )
            self.assertEqual(selected["ResponseId"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
