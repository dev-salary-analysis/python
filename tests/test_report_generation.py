from __future__ import annotations

import unittest

from scripts.generate_report import generate_report


class ReportGenerationTest(unittest.TestCase):
    def test_report_contains_rubric_sections_and_all_domains(self) -> None:
        report = generate_report()

        for heading in [
            "데이터 준비 및 EDA",
            "Pandas·Polars 결과 비교",
            "시각화",
            "통계 분석",
            "ML Pipeline",
            "종합 결론",
            "코드 품질 및 개선 의견",
        ]:
            self.assertIn(heading, report)

        for domain_number in range(1, 7):
            self.assertIn(f"Domain {domain_number}", report)

        self.assertIn("high_salary_model.joblib", report)
        self.assertIn("Accuracy", report)
        self.assertIn("F1-score", report)
        self.assertIn("domain1_dynamic/salary_boxplot.png", report)
        self.assertIn("domain5_platform/salary_chart.png", report)

if __name__ == "__main__":
    unittest.main()
