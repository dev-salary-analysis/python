"""C#, Go, Swift 플랫폼 언어 사용자의 연봉·근무·경력 분석.

저장소 루트에서 `python3 domain5_platform/analysis.py`로 실행하면
같은 디렉터리의 summary.csv, salary_chart.png, result.md를 갱신한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


LANGUAGES = ("C#", "Go", "Swift")
SALARY_COLUMN = "ConvertedCompYearly"
EXPERIENCE_COLUMN = "YearsCodePro"
REMOTE_COLUMN = "RemoteWork"
ORG_COLUMN = "OrgSize"

OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent
INPUT_CANDIDATES = (
    PROJECT_ROOT / "result.csv",
    PROJECT_ROOT / "data" / "result.csv",
    PROJECT_ROOT / "data" / "results.csv",
)

EXPERIENCE_BINS = [-np.inf, 2, 5, 10, 15, 20, np.inf]
EXPERIENCE_LABELS = ["0–2 years", "3–5 years", "6–10 years", "11–15 years", "16–20 years", "21+ years"]

ORG_ORDER = [
    "Freelancer / solo",
    "<20",
    "20–99",
    "100–499",
    "500–999",
    "1,000–4,999",
    "5,000–9,999",
    "10,000+",
]
ORG_MAP = {
    "Just me - I am a freelancer, sole proprietor, etc.": "Freelancer / solo",
    "Less than 20 employees": "<20",
    "20 to 99 employees": "20–99",
    "100 to 499 employees": "100–499",
    "500 to 999 employees": "500–999",
    "1,000 to 4,999 employees": "1,000–4,999",
    "5,000 to 9,999 employees": "5,000–9,999",
    "10,000 or more employees": "10,000+",
}


def find_input_file() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.is_file():
            return candidate
    checked = "\n".join(f"- {path}" for path in INPUT_CANDIDATES)
    raise FileNotFoundError(f"분석할 CSV를 찾을 수 없습니다. 확인한 경로:\n{checked}")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    required = {SALARY_COLUMN, EXPERIENCE_COLUMN, REMOTE_COLUMN, ORG_COLUMN}
    required.update(f"lang_{language}" for language in LANGUAGES)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    df = df.copy()
    df[SALARY_COLUMN] = pd.to_numeric(df[SALARY_COLUMN], errors="coerce")
    df[EXPERIENCE_COLUMN] = pd.to_numeric(df[EXPERIENCE_COLUMN], errors="coerce")
    for language in LANGUAGES:
        column = f"lang_{language}"
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).eq(1)
    return df


def language_frame(df: pd.DataFrame, language: str) -> pd.DataFrame:
    return df.loc[df[f"lang_{language}"]].copy()


def classify_remote(value: object) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    text = str(value)
    if text == "Remote":
        return "Remote"
    if text == "In-person":
        return "In-person"
    if text.startswith("Hybrid") or text.startswith("Your choice"):
        return "Hybrid / flexible"
    return None


def pct_table(
    df: pd.DataFrame, source_column: str, categories: list[str], mapper=None
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: dict[str, pd.Series] = {}
    valid_counts: dict[str, int] = {}
    for language in LANGUAGES:
        values = language_frame(df, language)[source_column]
        if mapper is not None:
            values = values.map(mapper)
        values = values.dropna()
        valid_counts[language] = int(values.size)
        counts = values.value_counts().reindex(categories, fill_value=0)
        rows[language] = counts.div(counts.sum()).mul(100) if counts.sum() else counts.astype(float)
    return pd.DataFrame(rows).T.reindex(columns=categories), valid_counts


def salary_and_experience_summary(
    df: pd.DataFrame, remote_pct: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for language in LANGUAGES:
        subset = language_frame(df, language)
        salary = subset[SALARY_COLUMN].dropna()
        experience = subset[EXPERIENCE_COLUMN].dropna()
        known_org = subset[ORG_COLUMN].map(ORG_MAP).dropna()
        org_mode = known_org.mode().iloc[0] if not known_org.empty else ""
        rows.append(
            {
                "language": language,
                "sample_size": int(len(subset)),
                "mean_salary_usd": salary.mean(),
                "median_salary_usd": salary.median(),
                "mean_experience_years": experience.mean(),
                "median_experience_years": experience.median(),
                "remote_pct": remote_pct.loc[language, "Remote"],
                "hybrid_flexible_pct": remote_pct.loc[language, "Hybrid / flexible"],
                "in_person_pct": remote_pct.loc[language, "In-person"],
                "most_common_org_size": org_mode,
            }
        )
    return pd.DataFrame(rows)


def experience_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: dict[str, pd.Series] = {}
    valid_counts: dict[str, int] = {}
    for language in LANGUAGES:
        experience = language_frame(df, language)[EXPERIENCE_COLUMN].dropna()
        valid_counts[language] = int(experience.size)
        groups = pd.cut(
            experience,
            bins=EXPERIENCE_BINS,
            labels=EXPERIENCE_LABELS,
            right=True,
        )
        counts = groups.value_counts(sort=False).reindex(EXPERIENCE_LABELS, fill_value=0)
        rows[language] = counts.div(counts.sum()).mul(100)
    return pd.DataFrame(rows).T, valid_counts


def welch_ttest(df: pd.DataFrame) -> dict[str, float | int]:
    # 동시 사용자를 제외해 두 표본의 독립성을 확보한다.
    csharp = df.loc[df["lang_C#"] & ~df["lang_Go"], SALARY_COLUMN].dropna().to_numpy()
    go = df.loc[df["lang_Go"] & ~df["lang_C#"], SALARY_COLUMN].dropna().to_numpy()
    test = stats.ttest_ind(csharp, go, equal_var=False, nan_policy="omit")

    mean_diff = float(csharp.mean() - go.mean())
    se = float(np.sqrt(csharp.var(ddof=1) / len(csharp) + go.var(ddof=1) / len(go)))
    numerator = (csharp.var(ddof=1) / len(csharp) + go.var(ddof=1) / len(go)) ** 2
    denominator = (
        (csharp.var(ddof=1) / len(csharp)) ** 2 / (len(csharp) - 1)
        + (go.var(ddof=1) / len(go)) ** 2 / (len(go) - 1)
    )
    degrees_freedom = float(numerator / denominator)
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    return {
        "csharp_n": int(len(csharp)),
        "go_n": int(len(go)),
        "csharp_mean": float(csharp.mean()),
        "go_mean": float(go.mean()),
        "mean_difference": mean_diff,
        "ci_low": mean_diff - critical * se,
        "ci_high": mean_diff + critical * se,
        "t_statistic": float(test.statistic),
        "degrees_freedom": degrees_freedom,
        "p_value": float(test.pvalue),
    }


def make_chart(summary: pd.DataFrame, org_pct: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [0.9, 1.5]})

    x = np.arange(len(summary))
    width = 0.35
    axes[0].bar(x - width / 2, summary["mean_salary_usd"], width, label="Mean", color="#2563EB")
    axes[0].bar(x + width / 2, summary["median_salary_usd"], width, label="Median", color="#60A5FA")
    axes[0].set_xticks(x, summary["language"])
    axes[0].set_ylabel("Annual salary (USD)")
    axes[0].set_title("Salary by language")
    axes[0].legend(frameon=False)
    axes[0].yaxis.set_major_formatter(lambda value, _: f"${value / 1000:,.0f}K")
    for container in axes[0].containers:
        axes[0].bar_label(container, labels=[f"${v / 1000:,.0f}K" for v in container.datavalues], padding=3)

    colors = ["#CBD5E1", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8", "#1E40AF", "#172554"]
    left = np.zeros(len(org_pct))
    for category, color in zip(ORG_ORDER, colors, strict=True):
        values = org_pct[category].to_numpy()
        axes[1].barh(org_pct.index, values, left=left, label=category, color=color)
        left += values
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Share of respondents with known organization size (%)")
    axes[1].set_title("Organization size distribution")
    axes[1].legend(title="Employees", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    fig.suptitle("Platform Languages: Salary and Organization Size", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, *, index_name: str = "Language", digits: int = 1) -> str:
    display = df.copy()
    display.index.name = index_name
    headers = [index_name, *map(str, display.columns)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for index, row in display.iterrows():
        values = [str(index)]
        for value in row:
            values.append(f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    input_path: Path,
    summary: pd.DataFrame,
    remote_pct: pd.DataFrame,
    remote_n: dict[str, int],
    org_pct: pd.DataFrame,
    org_n: dict[str, int],
    exp_pct: pd.DataFrame,
    exp_n: dict[str, int],
    test: dict[str, float | int],
    output_path: Path,
) -> None:
    salary_table = summary.set_index("language")[["sample_size", "mean_salary_usd", "median_salary_usd"]].copy()
    salary_table.columns = ["N", "Mean salary (USD)", "Median salary (USD)"]
    salary_table["N"] = salary_table["N"].map(lambda value: f"{int(value):,}")
    salary_table["Mean salary (USD)"] = salary_table["Mean salary (USD)"].map(lambda value: f"{value:,.0f}")
    salary_table["Median salary (USD)"] = salary_table["Median salary (USD)"].map(lambda value: f"{value:,.0f}")

    remote_display = remote_pct.copy()
    remote_display.insert(0, "Valid N", pd.Series(remote_n).map(lambda value: f"{value:,}"))
    remote_display.columns = ["Valid N", "Remote (%)", "Hybrid / flexible (%)", "In-person (%)"]
    org_display = org_pct.copy()
    org_display.insert(0, "Valid N", pd.Series(org_n).map(lambda value: f"{value:,}"))
    exp_display = exp_pct.copy()
    exp_display.insert(0, "Valid N", pd.Series(exp_n).map(lambda value: f"{value:,}"))

    significant = test["p_value"] < 0.05
    conclusion = (
        "5% 유의수준에서 두 그룹의 평균 연봉 차이는 통계적으로 유의하다."
        if significant
        else "5% 유의수준에서 두 그룹의 평균 연봉 차이는 통계적으로 유의하지 않다."
    )
    p_text = f"{test['p_value']:.3e}" if test["p_value"] < 0.001 else f"{test['p_value']:.4f}"

    report = f"""# C#, Go, Swift 사용자 분석

## 분석 개요

- 입력: `{input_path.relative_to(PROJECT_ROOT)}`
- 연봉 단위: 연간 USD (`{SALARY_COLUMN}`)
- 언어별 집계: 복수 언어 사용자는 각 언어 그룹에 중복 포함
- 비율: 해당 문항의 결측·미지 응답을 제외한 유효 응답 기준

## 1. 언어별 연봉 평균·중앙값

{markdown_table(salary_table, digits=0)}

## 2. 조직 규모 비교

셀프 고용을 별도 구간으로 두었고, 결측과 `I don’t know`는 비율 모수에서 제외했다.

{markdown_table(org_display)}

## 3. 원격·하이브리드·대면 근무 비율

설문의 두 가지 Hybrid 선택지와 `Your choice` 선택지를 `Hybrid / flexible`로 통합했다.

{markdown_table(remote_display)}

## 4. 경력 분포

전문 경력(`{EXPERIENCE_COLUMN}`)을 6개 구간으로 나눈 비율이다.

{markdown_table(exp_display)}

## 5. C# vs Go 연봉 Welch t-test

독립 표본 가정을 위해 C#과 Go를 동시에 사용한 응답자는 검정에서 제외했다. 등분산을 가정하지 않는 양측 Welch t-test를 적용했다.

- C# 전용군: n={test['csharp_n']:,}, 평균 ${test['csharp_mean']:,.0f}
- Go 전용군: n={test['go_n']:,}, 평균 ${test['go_mean']:,.0f}
- 평균 차이(C# - Go): ${test['mean_difference']:,.0f}
- 95% 신뢰구간: [${test['ci_low']:,.0f}, ${test['ci_high']:,.0f}]
- t({test['degrees_freedom']:.1f}) = {test['t_statistic']:.3f}, p = {p_text}
- 결론: {conclusion}

## 6. 차트

![언어별 연봉과 조직 규모](salary_chart.png)

## 해석 유의사항

- 이 분석은 관찰 데이터의 기술통계이며, 언어 사용이 연봉 차이의 원인임을 의미하지 않는다.
- 국가, 직무, 경력, 조직 규모 등의 교란 요인을 별도로 통제하지 않았다.
- 언어별 비율은 복수 언어 사용으로 인해 언어 간 독립적이지 않다.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    input_path = find_input_file()
    df = load_data(input_path)

    remote_pct, remote_n = pct_table(
        df,
        REMOTE_COLUMN,
        ["Remote", "Hybrid / flexible", "In-person"],
        classify_remote,
    )
    org_pct, org_n = pct_table(df, ORG_COLUMN, ORG_ORDER, lambda value: ORG_MAP.get(value))
    exp_pct, exp_n = experience_distribution(df)
    summary = salary_and_experience_summary(df, remote_pct)
    test = welch_ttest(df)

    summary_to_save = summary.copy()
    numeric_columns = summary_to_save.select_dtypes(include="number").columns
    summary_to_save[numeric_columns] = summary_to_save[numeric_columns].round(2)
    summary_to_save.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    make_chart(summary, org_pct, OUTPUT_DIR / "salary_chart.png")
    write_report(
        input_path,
        summary,
        remote_pct,
        remote_n,
        org_pct,
        org_n,
        exp_pct,
        exp_n,
        test,
        OUTPUT_DIR / "result.md",
    )
    print(f"분석 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
