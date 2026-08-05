#!/usr/bin/env python3
"""C, C++, Rust 사용자의 연봉·경력·직무를 분석한다."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import EXPERIENCE_COLUMN, SALARY_COLUMN  # noqa: E402
from common.data import load_domain  # noqa: E402


DOMAIN = "domain4_system"
LANGUAGES = ("C", "C++", "Rust")
OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
CHART_PATH = OUTPUT_DIR / "salary_boxplot.png"
REPORT_PATH = OUTPUT_DIR / "result.md"


def language_frame(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Select exact language users through the shared binary flags."""
    return df.loc[pd.to_numeric(df[f"lang_{language}"], errors="coerce").eq(1)].copy()


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the descriptive statistics required by the rubric."""
    rows = []
    for language in LANGUAGES:
        subset = language_frame(df, language)
        salary = pd.to_numeric(subset[SALARY_COLUMN], errors="coerce").dropna()
        experience = pd.to_numeric(subset[EXPERIENCE_COLUMN], errors="coerce").dropna()
        dev_counts = subset["DevType"].dropna().value_counts()
        rows.append(
            {
                "Language": language,
                "Users": len(subset),
                "MeanSalary": salary.mean(),
                "MedianSalary": salary.median(),
                "SalaryStd": salary.std(),
                "SalaryQ1": salary.quantile(0.25),
                "SalaryQ3": salary.quantile(0.75),
                "MeanExperience": experience.mean(),
                "MedianExperience": experience.median(),
                "TopDevType": dev_counts.index[0] if not dev_counts.empty else "응답 없음",
            }
        )
    return pd.DataFrame(rows)


def welch_salary_test(df: pd.DataFrame) -> dict[str, float | int]:
    """Compare disjoint C++-only and Rust-only salary samples."""
    cpp_only = df["lang_C++"].eq(1) & df["lang_Rust"].eq(0)
    rust_only = df["lang_Rust"].eq(1) & df["lang_C++"].eq(0)
    cpp_salary = pd.to_numeric(df.loc[cpp_only, SALARY_COLUMN], errors="coerce").dropna()
    rust_salary = pd.to_numeric(df.loc[rust_only, SALARY_COLUMN], errors="coerce").dropna()
    result = ttest_ind(cpp_salary, rust_salary, equal_var=False, nan_policy="omit")
    return {
        "cpp_only_n": len(cpp_salary),
        "rust_only_n": len(rust_salary),
        "cpp_mean": float(cpp_salary.mean()),
        "rust_mean": float(rust_salary.mean()),
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def create_chart(df: pd.DataFrame) -> None:
    """Create a labelled static salary distribution chart."""
    data = [language_frame(df, language)[SALARY_COLUMN] for language in LANGUAGES]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.boxplot(data, tick_labels=LANGUAGES, showmeans=True, showfliers=False)
    axis.set_xlabel("System programming language")
    axis.set_ylabel("Annual salary (USD)")
    axis.set_title("Salary Distribution by System Programming Language")
    axis.yaxis.set_major_formatter(lambda value, _: f"${value / 1000:,.0f}K")
    figure.tight_layout()
    figure.savefig(CHART_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_report(summary: pd.DataFrame, test: dict[str, float | int], rows: int) -> str:
    table = [
        "| 언어 | 사용자 수 | 평균 연봉 | 중앙 연봉 | 표준편차 | Q1 | Q3 | 평균 경력 | 주요 직무 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.to_dict(orient="records"):
        table.append(
            f"| {row['Language']} | {int(row['Users']):,} | ${row['MeanSalary']:,.0f} | "
            f"${row['MedianSalary']:,.0f} | ${row['SalaryStd']:,.0f} | "
            f"${row['SalaryQ1']:,.0f} | ${row['SalaryQ3']:,.0f} | "
            f"{row['MeanExperience']:.1f}년 | {row['TopDevType']} |"
        )
    significant = float(test["p_value"]) < 0.05
    conclusion = (
        "5% 유의수준에서 C++ 전용군과 Rust 전용군의 평균 연봉 차이는 통계적으로 유의하다."
        if significant
        else "5% 유의수준에서 C++ 전용군과 Rust 전용군의 평균 연봉 차이는 통계적으로 유의하지 않다."
    )
    return f"""# 시스템 프로그래밍 언어 분석

## 1. 분석 목적

C, C++, Rust 사용자의 연봉·경력·직무 특성을 비교한다.

## 2. 대상 언어

- C, C++, Rust
- 도메인 표본: {rows:,}명

## 3. 데이터 전처리

- 공통 정제 데이터 `data/results.csv` 사용
- 정확한 언어 0/1 플래그로 사용자 선택
- 연봉과 경력은 숫자형으로 변환하고 유효값 기준으로 집계

## 4. 기술통계

{chr(10).join(table)}

## 5. 시각화 결과

![시스템 언어별 연봉 분포](salary_boxplot.png)

## 6. 통계 검정

- 검정: 양측 Welch 독립표본 t-test
- C++ 전용군: n={int(test['cpp_only_n']):,}, 평균=${float(test['cpp_mean']):,.0f}
- Rust 전용군: n={int(test['rust_only_n']):,}, 평균=${float(test['rust_mean']):,.0f}
- t-statistic: {float(test['t_statistic']):.4f}
- p-value: {float(test['p_value']):.6f}
- 해석: {conclusion}

## 7. 결과 해석

1. 시스템 언어별 연봉은 중앙값과 사분위 범위까지 함께 비교해야 한다.
2. Rust와 C++ 비교에서는 두 언어를 동시에 사용하는 응답자를 제외했다.
3. 직무·경력 구성 차이를 함께 확인해야 언어별 연봉 차이를 과대해석하지 않을 수 있다.

## 8. 분석 한계

- 언어별 그룹은 기술통계에서 서로 중복될 수 있다.
- 관찰 자료이므로 언어 사용과 연봉 사이의 인과관계를 의미하지 않는다.
"""


def main() -> None:
    df = load_domain(DOMAIN)
    summary = build_summary(df)
    test = welch_salary_test(df)
    create_chart(df)
    summary.round(3).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(create_report(summary, test, len(df)), encoding="utf-8")

    print(summary.to_string(index=False))
    print(
        f"[t-test] C++ only vs Rust only: t={test['t_statistic']:.4f}, "
        f"p={test['p_value']:.6f}"
    )
    print(f"[완료] 기술통계: {SUMMARY_PATH}")
    print(f"[완료] 차트: {CHART_PATH}")
    print(f"[완료] 결과 보고서: {REPORT_PATH}")


if __name__ == "__main__":
    main()
