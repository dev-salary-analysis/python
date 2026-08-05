#!/usr/bin/env python3
"""Combine the six independent domain reports into the rubric report.md."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "report.md"
METADATA_PATH = PROJECT_ROOT / "data" / "preprocessing_metadata.json"
COMPARISON_PATH = PROJECT_ROOT / "domain2_jvm" / "pandas_polars_comparison.csv"
MODEL_METRICS_PATH = PROJECT_ROOT / "domain6_ml" / "model_metrics.json"
DOMAIN_REPORTS = [
    ("Domain 1 - 동적·데이터 언어", PROJECT_ROOT / "domain1_dynamic" / "result.md"),
    ("Domain 2 - JVM·엔터프라이즈 언어", PROJECT_ROOT / "domain2_jvm" / "result.md"),
    ("Domain 3 - 웹 개발 언어", PROJECT_ROOT / "domain3_web" / "result.md"),
    ("Domain 4 - 시스템 프로그래밍 언어", PROJECT_ROOT / "domain4_system" / "result.md"),
    ("Domain 5 - 플랫폼·백엔드 언어", PROJECT_ROOT / "domain5_platform" / "result.md"),
    ("Domain 6 - 통합 고연봉 예측", PROJECT_ROOT / "domain6_ml" / "result.md"),
]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"필수 산출물이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_comparison() -> list[dict[str, str]]:
    if not COMPARISON_PATH.is_file():
        raise FileNotFoundError(f"Pandas·Polars 비교 파일이 없습니다: {COMPARISON_PATH}")
    with COMPARISON_PATH.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def strip_first_heading(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def prefix_relative_links(text: str, directory_name: str) -> str:
    """Make links copied from a domain report valid from the repository root."""
    pattern = re.compile(r"(\]\()([^)]*)(\))")

    def replace(match: re.Match[str]) -> str:
        target = match.group(2)
        if not target or target.startswith(("http://", "https://", "#", "/", "mailto:")):
            return match.group(0)
        return f"{match.group(1)}{directory_name}/{target}{match.group(3)}"

    return pattern.sub(replace, text)


def generate_report() -> str:
    metadata = load_json(METADATA_PATH)
    metrics = load_json(MODEL_METRICS_PATH)
    comparison = load_comparison()
    missing_reports = [str(path) for _, path in DOMAIN_REPORTS if not path.is_file()]
    if missing_reports:
        raise FileNotFoundError("도메인 보고서가 누락되었습니다: " + ", ".join(missing_reports))

    pandas_row = next(row for row in comparison if row["Library"] == "Pandas")
    polars_row = next(row for row in comparison if row["Library"] == "Polars")
    full_model = metrics["full_model"]
    excluded = metadata.get("excluded_languages", [])
    duplicate_count = int(float(pandas_row.get("DuplicateResponseId", 0)))

    lines = [
        "# Stack Overflow 개발자 연봉 분석 및 고연봉 예측",
        "",
        "> 이 문서는 여섯 개 독립 도메인의 분석 결과를 자동으로 통합한 최종 보고서입니다.",
        "",
        "## Executive Summary",
        "",
        f"- 데이터: Stack Overflow Developer Survey {metadata['survey_year']}",
        f"- 공통 정제 표본: {metadata['final_rows']:,}명",
        f"- 분석 언어: {len(metadata['eligible_languages'])}개",
        f"- 표본 기준 미달 언어: {', '.join(excluded) if excluded else '없음'}",
        f"- 전체 연봉 중앙값: ${metadata['median_salary_usd']:,.0f}",
        f"- 최종 ML: Accuracy {full_model['accuracy']:.3f}, F1 {full_model['f1']:.3f}, ROC-AUC {full_model['roc_auc']:.3f}",
        "",
        "## 1. 데이터 준비 및 EDA",
        "",
        "공통 전처리 스크립트가 연봉·언어 결측, 0 이하 연봉, 중복 ResponseId, 상·하위 1% 연봉 이상치를 처리한다. 전문 경력은 숫자로 변환하고 언어는 세미콜론 토큰으로 정확히 분리한다.",
        "",
        f"- 원본 행 수: {metadata['initial_rows']:,}",
        f"- 필수값 처리 후: {metadata['rows_after_required_value_filter']:,}",
        f"- 최종 행 수: {metadata['final_rows']:,}",
        f"- 공통 정제 데이터의 중복 ResponseId: {duplicate_count:,}건",
        f"- 연봉 이상치 범위: ${metadata['salary_outlier_rule']['lower_usd']:,.0f} - ${metadata['salary_outlier_rule']['upper_usd']:,.0f}",
        "",
        "### Pandas·Polars 결과 비교",
        "",
        "| 라이브러리 | 행 | 열 | 로딩 시간(초) | 평균 연봉 | 중앙 연봉 | 컬럼 일치 | 요약값 일치 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
        f"| Pandas | {int(pandas_row['Rows']):,} | {pandas_row['Columns']} | {float(pandas_row['LoadSeconds']):.4f} | ${float(pandas_row['SalaryMeanUSD']):,.0f} | ${float(pandas_row['SalaryMedianUSD']):,.0f} | {pandas_row['ColumnsMatch']} | {pandas_row['NumericSummaryMatch']} |",
        f"| Polars | {int(polars_row['Rows']):,} | {polars_row['Columns']} | {float(polars_row['LoadSeconds']):.4f} | ${float(polars_row['SalaryMeanUSD']):,.0f} | ${float(polars_row['SalaryMedianUSD']):,.0f} | {polars_row['ColumnsMatch']} | {polars_row['NumericSummaryMatch']} |",
        "",
        "## 2. 시각화",
        "",
        "- Seaborn 정적 차트: [JVM 언어 종합 차트](domain2_jvm/result/salary_chart.png)",
        "- Plotly 인터랙티브 차트: [웹 언어 인터랙티브 분석](domain3_web/salary_interactive.html)",
        "- Plotly 통합 보고서: [전체 언어 연봉 분석](domain6_ml/language_salary_report.html)",
        "",
        "모든 주요 차트에는 제목과 축 레이블을 포함했다.",
        "",
        "## 3. 통계 분석",
        "",
        "각 도메인은 평균·표준편차·사분위수를 산출하고, Domain 2는 Pearson 상관계수를 표와 히트맵으로 출력한다. 언어 간 평균 연봉 비교에는 `scipy.stats.ttest_ind` 기반 양측 Welch t-test를 사용하며 동시 사용자는 비교 표본에서 제외한다.",
        "",
        "## 4. ML Pipeline",
        "",
        f"- 목표: 학습 데이터 연봉 중앙값(${metrics['salary_threshold']:,.0f}) 이상 여부",
        "- 전처리: 수치형 중앙값 대치·표준화, 범주형 최빈값 대치·One-Hot Encoding",
        "- 모델: LogisticRegression(class_weight='balanced')",
        f"- Accuracy: {full_model['accuracy']:.3f}",
        f"- Precision: {full_model['precision']:.3f}",
        f"- Recall: {full_model['recall']:.3f}",
        f"- F1-score: {full_model['f1']:.3f}",
        f"- ROC-AUC: {full_model['roc_auc']:.3f}",
        "- 저장 모델: [high_salary_model.joblib](domain6_ml/high_salary_model.joblib)",
        "- 혼동행렬: [confusion_matrix.png](domain6_ml/confusion_matrix.png)",
        "",
        "## 5. 도메인별 결과",
        "",
    ]

    for title, path in DOMAIN_REPORTS:
        domain_text = strip_first_heading(path.read_text(encoding="utf-8"))
        domain_text = prefix_relative_links(domain_text, path.parent.name)
        lines.extend([f"### {title}", "", domain_text, ""])

    lines.extend(
        [
            "## 6. 종합 결론",
            "",
            "1. 언어별 연봉 차이는 존재하지만 국가·경력·직무·조직 규모가 함께 반영된 관찰 결과다.",
            "2. 전체 모델은 언어 정보만 사용한 모델보다 높은 성능을 보였고, 프로필 특성이 예측의 큰 부분을 설명했다.",
            "3. 결과를 언어의 직접적인 인과 효과나 개인의 연봉 책정 근거로 사용해서는 안 된다.",
            "",
            "## 7. 코드 품질 및 개선 의견",
            "",
            "- 공통 전처리로 여섯 도메인의 데이터 기준을 통일하고 도메인 분석은 독립 실행 가능하게 구성했다.",
            "- 독립표본 검정에서는 복수 언어 동시 사용자를 제외해 검정 가정을 보완했다.",
            "- 후속 개선으로 교차검증, 하이퍼파라미터 탐색, 국가별 생활비 보정, 신뢰구간 시각화를 적용할 수 있다.",
            "- 데이터 연도와 스키마가 바뀌면 공통 전처리 검증 테스트를 먼저 실행해야 한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = generate_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[완료] 통합 보고서: {REPORT_PATH}")


if __name__ == "__main__":
    main()
