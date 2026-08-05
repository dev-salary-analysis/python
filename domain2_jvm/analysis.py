#!/usr/bin/env python3
"""JVM-language EDA, statistics, and visualization workflow."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "stackoverflow-salary-matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats

from common.config import EXPERIENCE_COLUMN, MIN_SAMPLE_SIZE, SALARY_COLUMN
from common.data import PROCESSED_DATA_PATH, load_all, load_domain


MODULE_DIR = Path(__file__).resolve().parent
RESULT_DIR = MODULE_DIR / "result"
LANGUAGES = ("Java", "Kotlin", "Scala")
SUMMARY_PATH = RESULT_DIR / "summary.csv"
CHART_PATH = RESULT_DIR / "salary_chart.png"
PANDAS_POLARS_PATH = MODULE_DIR / "pandas_polars_comparison.csv"
REPORT_PATH = MODULE_DIR / "result.md"


def compare_pandas_polars() -> pd.DataFrame:
    """Load the shared CSV with both libraries and compare material results."""
    started = time.perf_counter()
    pandas_df = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False, encoding="utf-8-sig")
    pandas_seconds = time.perf_counter() - started

    started = time.perf_counter()
    polars_df = pl.read_csv(PROCESSED_DATA_PATH, encoding="utf8-lossy")
    polars_seconds = time.perf_counter() - started

    if pandas_df.shape != polars_df.shape:
        raise ValueError(
            f"Pandas와 Polars 로딩 결과가 다릅니다: "
            f"{pandas_df.shape} != {polars_df.shape}"
        )
    columns_match = list(pandas_df.columns) == list(polars_df.columns)
    if not columns_match:
        raise ValueError("Pandas와 Polars의 컬럼 이름 또는 순서가 다릅니다.")

    pandas_salary = pd.to_numeric(pandas_df[SALARY_COLUMN], errors="coerce")
    polars_salary = pd.to_numeric(
        pd.Series(polars_df[SALARY_COLUMN].to_list()), errors="coerce"
    )
    pandas_mean = float(pandas_salary.mean())
    polars_mean = float(polars_salary.mean())
    pandas_median = float(pandas_salary.median())
    polars_median = float(polars_salary.median())
    numeric_summary_match = bool(
        np.isclose(pandas_mean, polars_mean, rtol=1e-10, atol=1e-6)
        and np.isclose(pandas_median, polars_median, rtol=1e-10, atol=1e-6)
    )
    if not numeric_summary_match:
        raise ValueError("Pandas와 Polars의 연봉 요약 결과가 다릅니다.")

    pandas_duplicates = int(pandas_df["ResponseId"].duplicated().sum())
    polars_response_ids = pd.Series(polars_df["ResponseId"].to_list())
    polars_duplicates = int(polars_response_ids.duplicated().sum())
    return pd.DataFrame(
        [
            {
                "Library": "Pandas",
                "Rows": pandas_df.shape[0],
                "Columns": pandas_df.shape[1],
                "LoadSeconds": pandas_seconds,
                "SalaryMeanUSD": pandas_mean,
                "SalaryMedianUSD": pandas_median,
                "DuplicateResponseId": pandas_duplicates,
                "ColumnsMatch": columns_match,
                "NumericSummaryMatch": numeric_summary_match,
            },
            {
                "Library": "Polars",
                "Rows": polars_df.shape[0],
                "Columns": polars_df.shape[1],
                "LoadSeconds": polars_seconds,
                "SalaryMeanUSD": polars_mean,
                "SalaryMedianUSD": polars_median,
                "DuplicateResponseId": polars_duplicates,
                "ColumnsMatch": columns_match,
                "NumericSummaryMatch": numeric_summary_match,
            },
        ]
    )


def language_frame(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Select exact users of a language through its precomputed binary flag."""
    column = f"lang_{language}"
    if column not in df.columns:
        raise ValueError(f"공통 데이터에 필요한 컬럼이 없습니다: {column}")
    return df.loc[pd.to_numeric(df[column], errors="coerce").eq(1)].copy()


def top_category(series: pd.Series) -> tuple[str, float]:
    values = series.dropna().astype(str)
    if values.empty:
        return "응답 없음", np.nan
    counts = values.value_counts()
    return str(counts.index[0]), float(counts.iloc[0] / len(values) * 100)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate salary, experience, role, and organization statistics."""
    rows: list[dict[str, object]] = []
    for language in LANGUAGES:
        group = language_frame(df, language)
        if len(group) < MIN_SAMPLE_SIZE:
            continue

        salary = pd.to_numeric(group[SALARY_COLUMN], errors="coerce").dropna()
        experience = pd.to_numeric(group[EXPERIENCE_COLUMN], errors="coerce").dropna()
        top_job, top_job_rate = top_category(group["DevType"])
        top_org, top_org_rate = top_category(group["OrgSize"])
        rows.append(
            {
                "Language": language,
                "UserCount": len(group),
                "SalaryMeanUSD": salary.mean(),
                "SalaryMedianUSD": salary.median(),
                "SalaryStdUSD": salary.std(),
                "SalaryQ1USD": salary.quantile(0.25),
                "SalaryQ3USD": salary.quantile(0.75),
                "ExperienceCount": len(experience),
                "ExperienceMeanYears": experience.mean(),
                "ExperienceMedianYears": experience.median(),
                "ExperienceStdYears": experience.std(),
                "ExperienceQ1Years": experience.quantile(0.25),
                "ExperienceQ3Years": experience.quantile(0.75),
                "HighSalaryRatePct": group["HighSalary"].mean() * 100,
                "TopDevType": top_job,
                "TopDevTypePct": top_job_rate,
                "TopOrgSize": top_org,
                "TopOrgSizePct": top_org_rate,
            }
        )
    return pd.DataFrame(rows)


def build_long_language_data(df: pd.DataFrame) -> pd.DataFrame:
    """Expand multi-language users to one row for each JVM language used."""
    columns = [
        "ResponseId",
        "Country",
        "DevType",
        "OrgSize",
        SALARY_COLUMN,
        EXPERIENCE_COLUMN,
    ]
    frames: list[pd.DataFrame] = []
    for language in LANGUAGES:
        group = language_frame(df, language)[columns].copy()
        group["Language"] = language
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def welch_salary_test(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Compare disjoint Java-only and Kotlin-only users with Welch's t-test."""
    java_only = df["lang_Java"].eq(1) & df["lang_Kotlin"].eq(0)
    kotlin_only = df["lang_Kotlin"].eq(1) & df["lang_Java"].eq(0)
    java_salary = pd.to_numeric(
        df.loc[java_only, SALARY_COLUMN], errors="coerce"
    ).dropna()
    kotlin_salary = pd.to_numeric(
        df.loc[kotlin_only, SALARY_COLUMN], errors="coerce"
    ).dropna()
    result = stats.ttest_ind(
        java_salary, kotlin_salary, equal_var=False, nan_policy="omit"
    )
    return {
        "test": "Welch independent two-sample t-test",
        "java_only_n": len(java_salary),
        "kotlin_only_n": len(kotlin_salary),
        "java_only_mean_usd": float(java_salary.mean()),
        "kotlin_only_mean_usd": float(kotlin_salary.mean()),
        "mean_difference_usd": float(kotlin_salary.mean() - java_salary.mean()),
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        SALARY_COLUMN,
        EXPERIENCE_COLUMN,
        "lang_Java",
        "lang_Kotlin",
        "lang_Scala",
    ]
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="pearson")


def organization_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate within-language rates for the four most common org sizes."""
    valid = long_df.dropna(subset=["OrgSize"]).copy()
    top_sizes = valid["OrgSize"].value_counts().head(4).index.tolist()
    counts = (
        valid.groupby(["Language", "OrgSize"], observed=True)
        .size()
        .rename("Count")
        .reset_index()
    )
    totals = valid.groupby("Language", observed=True).size().rename("Total")
    counts = counts.merge(totals, on="Language")
    counts = counts[counts["OrgSize"].isin(top_sizes)].copy()
    counts["RatePct"] = counts["Count"] / counts["Total"] * 100
    short_labels = {
        "10,000 or more employees": "10,000+",
        "1,000 to 4,999 employees": "1,000–4,999",
        "100 to 499 employees": "100–499",
        "20 to 99 employees": "20–99",
        "Less than 20 employees": "<20",
    }
    counts["OrgLabel"] = counts["OrgSize"].replace(short_labels)
    counts["OrgLabel"] = pd.Categorical(
        counts["OrgLabel"],
        categories=[short_labels.get(size, size) for size in top_sizes],
        ordered=True,
    )
    return counts


def create_salary_chart(long_df: pd.DataFrame, correlation: pd.DataFrame) -> None:
    """Create one 2x2 PNG containing all JVM-domain visual results."""
    sns.set_theme(style="whitegrid", context="notebook")
    colors = {"Java": "#E76F00", "Kotlin": "#7F52FF", "Scala": "#DC322F"}
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))

    sns.boxplot(
        data=long_df,
        x="Language",
        y=SALARY_COLUMN,
        hue="Language",
        palette=colors,
        showfliers=False,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Annual Salary by JVM Language")
    axes[0, 0].set_xlabel("JVM language")
    axes[0, 0].set_ylabel("Converted compensation (USD)")
    axes[0, 0].ticklabel_format(axis="y", style="plain")

    sns.boxplot(
        data=long_df,
        x="Language",
        y=EXPERIENCE_COLUMN,
        hue="Language",
        palette=colors,
        showfliers=False,
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Professional Experience by JVM Language")
    axes[0, 1].set_xlabel("JVM language")
    axes[0, 1].set_ylabel("Years of professional experience")

    labels = ["Salary", "Experience", "Java", "Kotlin", "Scala"]
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.8},
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Pearson Correlation")

    org_data = organization_distribution(long_df)
    sns.barplot(
        data=org_data,
        x="OrgLabel",
        y="RatePct",
        hue="Language",
        palette=colors,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Top Organization Sizes by JVM Language")
    axes[1, 1].set_xlabel("Organization size (employees)")
    axes[1, 1].set_ylabel("Share within language (%)")
    axes[1, 1].legend(title="Language")

    figure.suptitle("2025 Stack Overflow JVM Developer Analysis", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(CHART_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_report(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    correlation: pd.DataFrame,
    test_result: dict[str, float | int | str],
    domain_rows: int,
) -> str:
    """Create the standard JVM-domain Markdown report."""
    comparison_rows = [
        "| 라이브러리 | 행 | 열 | 로딩 시간(초) | 평균 연봉 | 중앙 연봉 | 중복 ID |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict(orient="records"):
        comparison_rows.append(
            f"| {row['Library']} | {int(row['Rows']):,} | {int(row['Columns'])} | "
            f"{row['LoadSeconds']:.4f} | ${row['SalaryMeanUSD']:,.0f} | "
            f"${row['SalaryMedianUSD']:,.0f} | {int(row['DuplicateResponseId'])} |"
        )

    summary_rows = [
        "| 언어 | 사용자 수 | 평균 연봉 | 중앙 연봉 | 표준편차 | Q1 | Q3 | 평균 경력 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        summary_rows.append(
            f"| {row['Language']} | {int(row['UserCount']):,} | ${row['SalaryMeanUSD']:,.0f} | "
            f"${row['SalaryMedianUSD']:,.0f} | ${row['SalaryStdUSD']:,.0f} | "
            f"${row['SalaryQ1USD']:,.0f} | ${row['SalaryQ3USD']:,.0f} | "
            f"{row['ExperienceMeanYears']:.1f}년 |"
        )

    labels = ["연봉", "경력", "Java", "Kotlin", "Scala"]
    corr_rows = [
        "| 변수 | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for label, (_, row) in zip(labels, correlation.iterrows(), strict=True):
        corr_rows.append(
            f"| {label} | " + " | ".join(f"{value:.3f}" for value in row) + " |"
        )

    significant = float(test_result["p_value"]) < 0.05
    conclusion = (
        "5% 유의수준에서 Java 전용군과 Kotlin 전용군의 평균 연봉 차이는 통계적으로 유의하다."
        if significant
        else "5% 유의수준에서 Java 전용군과 Kotlin 전용군의 평균 연봉 차이는 통계적으로 유의하지 않다."
    )
    return f"""# JVM·엔터프라이즈 언어 분석

## 1. 분석 목적

Java, Kotlin, Scala 사용자의 연봉·경력·직무·조직 규모를 비교한다.

## 2. 대상 언어

- Java, Kotlin, Scala
- JVM 도메인 표본: {domain_rows:,}명

## 3. 데이터 전처리

- 공통 정제 데이터 `data/results.csv` 사용
- Pandas와 Polars로 동일 CSV를 로딩해 크기·컬럼·연봉 요약값·중복 결과 비교
- 세 라이브러리 플래그는 정확한 세미콜론 토큰 분리 결과를 사용

{chr(10).join(comparison_rows)}

## 4. 기술통계

{chr(10).join(summary_rows)}

### Pearson 상관계수

{chr(10).join(corr_rows)}

## 5. 시각화 결과

![JVM 언어 종합 차트](result/salary_chart.png)

## 6. 통계 검정

- 검정: 양측 Welch 독립표본 t-test
- Java 전용군: n={int(test_result['java_only_n']):,}
- Kotlin 전용군: n={int(test_result['kotlin_only_n']):,}
- t-statistic: {float(test_result['t_statistic']):.4f}
- p-value: {float(test_result['p_value']):.6f}
- 해석: {conclusion}

## 7. 결과 해석

1. JVM 언어별 연봉은 평균·중앙값·사분위 범위를 함께 비교해야 한다.
2. 연봉과 전문 경력의 Pearson 상관계수는 차트와 표에서 동시에 확인할 수 있다.
3. Pandas와 Polars는 동일한 데이터 크기와 핵심 연봉 요약값을 산출했다.

## 8. 분석 한계

- 복수 언어 사용자는 언어별 기술통계에 중복 포함될 수 있다.
- 국가·직무·경력 등 교란 요인을 통제하지 않은 관찰 분석이므로 인과관계로 해석할 수 없다.
"""

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = compare_pandas_polars()
    all_df = load_all()
    domain_df = load_domain("domain2_jvm")
    summary = build_summary(all_df)
    missing = sorted(set(LANGUAGES) - set(summary["Language"]))
    if missing:
        raise ValueError(f"최소 표본 수 {MIN_SAMPLE_SIZE}명 미만 언어: {missing}")

    numeric_columns = summary.select_dtypes(include="number").columns
    summary[numeric_columns] = summary[numeric_columns].round(3)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    long_df = build_long_language_data(all_df)
    correlation = calculate_correlations(domain_df)
    create_salary_chart(long_df, correlation)
    test_result = welch_salary_test(all_df)
    comparison.to_csv(PANDAS_POLARS_PATH, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(
        create_report(comparison, summary, correlation, test_result, len(domain_df)),
        encoding="utf-8",
    )

    pandas_time = comparison.loc[
        comparison["Library"].eq("Pandas"), "LoadSeconds"
    ].iloc[0]
    polars_time = comparison.loc[
        comparison["Library"].eq("Polars"), "LoadSeconds"
    ].iloc[0]
    print(
        f"[EDA] JVM 사용자 {len(domain_df):,}명, "
        f"중복 응답 {domain_df['ResponseId'].duplicated().sum():,}건"
    )
    print(
        f"[로딩 비교] Pandas {pandas_time:.4f}초, "
        f"Polars {polars_time:.4f}초"
    )
    print(
        f"[t-test] t={float(test_result['t_statistic']):.3f}, "
        f"p={float(test_result['p_value']):.6f}"
    )
    print(f"[완료] 기술통계: {SUMMARY_PATH}")
    print(f"[완료] 2×2 종합 차트: {CHART_PATH}")
    print(f"[완료] Pandas·Polars 비교: {PANDAS_POLARS_PATH}")
    print(f"[완료] 결과 보고서: {REPORT_PATH}")


if __name__ == "__main__":
    main()
