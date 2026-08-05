from __future__ import annotations

import unittest

from common.config import MIN_SAMPLE_SIZE
from common.data import load_all
from domain2_jvm.analysis import (
    LANGUAGES,
    build_summary,
    calculate_correlations,
    compare_pandas_polars,
    welch_salary_test,
)


class Domain2AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_all()

    def test_summary_contains_all_jvm_languages(self) -> None:
        summary = build_summary(self.df)
        self.assertEqual(summary["Language"].tolist(), list(LANGUAGES))
        self.assertTrue(summary["UserCount"].ge(MIN_SAMPLE_SIZE).all())
        self.assertTrue(summary["SalaryMedianUSD"].gt(0).all())

    def test_welch_test_uses_disjoint_language_groups(self) -> None:
        result = welch_salary_test(self.df)
        expected_java_only = int(
            (self.df["lang_Java"].eq(1) & self.df["lang_Kotlin"].eq(0)).sum()
        )
        expected_kotlin_only = int(
            (self.df["lang_Kotlin"].eq(1) & self.df["lang_Java"].eq(0)).sum()
        )
        self.assertEqual(result["java_only_n"], expected_java_only)
        self.assertEqual(result["kotlin_only_n"], expected_kotlin_only)
        self.assertGreaterEqual(result["p_value"], 0)
        self.assertLessEqual(result["p_value"], 1)

    def test_correlation_matrix_is_complete_and_bounded(self) -> None:
        correlation = calculate_correlations(self.df)
        self.assertEqual(correlation.shape, (5, 5))
        self.assertTrue(correlation.ge(-1).all().all())
        self.assertTrue(correlation.le(1).all().all())

    def test_pandas_polars_material_results_match(self) -> None:
        comparison = compare_pandas_polars()
        self.assertEqual(comparison["Rows"].nunique(), 1)
        self.assertEqual(comparison["Columns"].nunique(), 1)
        self.assertTrue(comparison["ColumnsMatch"].all())
        self.assertTrue(comparison["NumericSummaryMatch"].all())
        self.assertEqual(comparison["SalaryMeanUSD"].round(6).nunique(), 1)
        self.assertEqual(comparison["SalaryMedianUSD"].round(6).nunique(), 1)

if __name__ == "__main__":
    unittest.main()
