# ====================================================================
# 도메인 1 - 동적·데이터 언어(Python, R, Julia) 연봉 분석
# ====================================================================
# 공용 데이터(data/results.csv)에서 Python·R 사용자를 추려
#  1) 언어별 표본 수 / 연봉 통계 / 경력 통계 / 주요 직무를 요약(summary.csv)하고
#  2) 두 언어 사용자 연봉 분포를 박스플롯으로 시각화하고(salary_boxplot.png)
#  3) Python vs R 연봉 차이를 Welch's t-test로 검정함
#
# Julia는 2025 설문 원본 표본이 0건이라 공통 규칙(MIN_SAMPLE_SIZE)에 따라
# 자동 제외됨. 자세한 배경은 result.md 참고.
#
# 변경내역:
#   2026-08-04 이송미: 최초 작성
# =============================================================

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import (  # noqa: E402
    EXPERIENCE_COLUMN,
    LANGUAGE_DOMAINS,
    MIN_SAMPLE_SIZE,
    SALARY_COLUMN,
)
from common.data import domain_languages, load_domain, load_metadata  # noqa: E402

DOMAIN = "domain1_dynamic"
OUTPUT_DIR = Path(__file__).resolve().parent


def build_summary(df: pd.DataFrame, languages: list[str]) -> pd.DataFrame:
    """언어별 표본 수·연봉 통계·경력 통계·최다 직무를 한 행씩 정리하는 함수"""
    rows = []
    for language in languages:
        # 언어 사용 여부는 lang_{language} 플래그(0/1)로 정확히 필터링됨
        # (문자열 contains() 대신 원-핫 컬럼을 쓰므로 Java ⊄ JavaScript 문제가 없음)
        subset = df.loc[df[f"lang_{language}"] == 1]
        top_devtype = subset["DevType"].value_counts()

        rows.append(
            {
                "language": language,
                "user_count": len(subset),
                "salary_mean": subset[SALARY_COLUMN].mean(),
                "salary_median": subset[SALARY_COLUMN].median(),
                "salary_std": subset[SALARY_COLUMN].std(),
                "experience_mean": subset[EXPERIENCE_COLUMN].mean(),
                "experience_median": subset[EXPERIENCE_COLUMN].median(),
                "top_devtype": top_devtype.index[0] if not top_devtype.empty else None,
                "top_devtype_share": (
                    top_devtype.iloc[0] / len(subset) if not top_devtype.empty else None
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_salary_boxplot(df: pd.DataFrame, languages: list[str], path: Path) -> None:
    """언어별 연봉 분포 박스플롯을 그리는 함수"""
    data = [df.loc[df[f"lang_{language}"] == 1, SALARY_COLUMN] for language in languages]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=languages, showmeans=True)

    ax.set_ylabel("Annual salary (USD)")
    ax.set_title("Salary distribution by dynamic/data language")
    ax.yaxis.set_major_formatter(lambda value, _: f"${value / 1000:,.0f}K")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    # 1) 어떤 언어가 최소 표본 기준(MIN_SAMPLE_SIZE)을 통과했는지 확인
    #    2025 원본은 Julia 응답이 0건이라 항상 제외 목록에 잡힘
    metadata = load_metadata()
    configured_languages = list(LANGUAGE_DOMAINS[DOMAIN])
    eligible_languages = domain_languages(DOMAIN)
    excluded_languages = [
        language for language in configured_languages if language not in eligible_languages
    ]

    # 2) 공용 정제 데이터에서 이 도메인(Python 또는 R 사용자)만 추출
    df = load_domain(DOMAIN)

    # 3) 산출물 생성: summary.csv, salary_boxplot.png
    summary = build_summary(df, eligible_languages)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    plot_salary_boxplot(df, eligible_languages, OUTPUT_DIR / "salary_boxplot.png")

    # 4) 통계 검정: Python vs R 연봉 평균 차이가 유의미한지 Welch's t-test로 확인
    #    두 표본 크기 차이가 크므로(12,457 vs 984) 등분산 가정을 두지 않음
    python_salary = df.loc[df["lang_Python"] == 1, SALARY_COLUMN]
    r_salary = df.loc[df["lang_R"] == 1, SALARY_COLUMN]
    ttest = stats.ttest_ind(python_salary, r_salary, equal_var=False)

    print(f"대상 언어: {eligible_languages} (제외: {excluded_languages}, "
          f"최소 표본 {MIN_SAMPLE_SIZE}명 미만)")
    for language in configured_languages:
        print(f"  {language}: 원본 표본 수 = {metadata['language_counts'][language]}")
    print(f"도메인 표본 수 (Python 또는 R 사용자, 중복 포함): {len(df)}")
    print()
    print(summary.to_string(index=False))
    print()
    print(
        "Python vs R salary t-test (Welch): "
        f"t={ttest.statistic:.4f}, p={ttest.pvalue:.6f}"
    )


if __name__ == "__main__":
    main()
