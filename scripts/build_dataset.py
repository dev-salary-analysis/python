#!/usr/bin/env python3
"""Download and clean the Stack Overflow Developer Survey into one shared CSV."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import (  # noqa: E402
    ALL_LANGUAGES,
    COUNTRIES,
    DATA_DIR,
    EXPERIENCE_COLUMN,
    LANGUAGE_COLUMN,
    LANGUAGE_DOMAINS,
    MIN_SAMPLE_SIZE,
    OFFICIAL_DATA_URLS,
    SALARY_COLUMN,
    SALARY_LOWER_QUANTILE,
    SALARY_UPPER_QUANTILE,
)


STANDARD_COLUMNS = (
    "ResponseId",
    "Country",
    SALARY_COLUMN,
    LANGUAGE_COLUMN,
    EXPERIENCE_COLUMN,
    "DevType",
    "EdLevel",
    "RemoteWork",
    "OrgSize",
    "Employment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack Overflow 설문 원본을 공통 분석 데이터로 만듭니다."
    )
    parser.add_argument("--year", type=int, default=2025, choices=OFFICIAL_DATA_URLS)
    parser.add_argument(
        "--source",
        help="공식 URL 대신 사용할 로컬 CSV 경로 또는 CSV URL",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        help="포함할 국가명. 생략하면 common/config.py의 COUNTRIES를 사용",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="이미 내려받은 원본 CSV가 있어도 다시 다운로드",
    )
    parser.add_argument(
        "--write-domain-files",
        action="store_true",
        help="필요한 경우에만 도메인별 data.csv도 추가 생성",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    """Stream a remote CSV to disk without loading the whole file into memory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        with requests.get(url, stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def obtain_source(source: str, raw_path: Path, force: bool) -> Path:
    """Resolve a local/remote input into a stable raw CSV path."""
    if raw_path.exists() and not force and source.startswith(("http://", "https://")):
        print(f"[재사용] {raw_path}")
        return raw_path

    if source.startswith(("http://", "https://")):
        print(f"[다운로드] {source}")
        download_file(source, raw_path)
        return raw_path

    local_path = Path(source).expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {local_path}")
    if local_path != raw_path.resolve():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, raw_path)
    return raw_path


def parse_experience(value: object) -> float:
    """Convert survey year values such as 'Less than 1 year' to numbers."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().lower()
    if text in {"less than 1 year", "less than 1 years"}:
        return 0.5
    if text.startswith("more than "):
        number = pd.to_numeric(text.split()[2], errors="coerce")
        return float(number + 1) if pd.notna(number) else np.nan
    return float(pd.to_numeric(text, errors="coerce"))


def split_languages(value: object) -> set[str]:
    """Split exact language tokens so Java never matches JavaScript."""
    if not isinstance(value, str):
        return set()
    return {token.strip() for token in value.split(";") if token.strip()}


def uses_language(value: object, language: str) -> bool:
    return language in split_languages(value)


def normalize_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Map year-specific columns to the shared analysis schema."""
    df = raw.copy()

    # 2025 survey renamed professional experience to WorkExp.
    if EXPERIENCE_COLUMN not in df.columns and "WorkExp" in df.columns:
        df[EXPERIENCE_COLUMN] = df["WorkExp"]

    required = {SALARY_COLUMN, LANGUAGE_COLUMN, EXPERIENCE_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"원본 데이터에 필수 컬럼이 없습니다: {missing}")

    for column in STANDARD_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df.loc[:, STANDARD_COLUMNS].copy()


def clean_data(
    raw: pd.DataFrame,
    year: int,
    countries: Iterable[str] | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = normalize_schema(raw)
    initial_rows = len(df)

    # 응답 ID가 있는 행만 중복 검사한다. 원본에 ResponseId 컬럼이 없는 경우
    # normalize_schema()가 결측 컬럼을 추가하므로, 결측 ID까지 중복으로 제거하면
    # 정상 행이 사라질 수 있다.
    valid_response_id = df["ResponseId"].notna()
    duplicate_mask = valid_response_id & df["ResponseId"].duplicated(keep="first")
    duplicate_rows_removed = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()
    rows_after_duplicate_filter = len(df)

    df[SALARY_COLUMN] = pd.to_numeric(df[SALARY_COLUMN], errors="coerce")
    df[EXPERIENCE_COLUMN] = df[EXPERIENCE_COLUMN].map(parse_experience)
    df = df.dropna(subset=[SALARY_COLUMN, LANGUAGE_COLUMN]).copy()
    df = df[df[SALARY_COLUMN] > 0].copy()
    after_required_filter = len(df)

    country_list = tuple(countries) if countries else None
    if country_list:
        df = df[df["Country"].isin(country_list)].copy()
    after_country_filter = len(df)

    if df.empty:
        country_text = ", ".join(country_list) if country_list else "ALL"
        raise ValueError(
            "연봉·언어·국가 조건을 통과한 응답이 없습니다. "
            f"국가명과 원본 스키마를 확인하세요: {country_text}"
        )

    lower = float(df[SALARY_COLUMN].quantile(SALARY_LOWER_QUANTILE))
    upper = float(df[SALARY_COLUMN].quantile(SALARY_UPPER_QUANTILE))
    df = df[df[SALARY_COLUMN].between(lower, upper, inclusive="both")].copy()

    language_sets = df[LANGUAGE_COLUMN].map(split_languages)
    for language in ALL_LANGUAGES:
        df[f"lang_{language}"] = language_sets.map(lambda values, x=language: int(x in values))

    # 목표 변수 기준값은 모든 도메인이 공유하는 정제 데이터의 중앙값입니다.
    median_salary = float(df[SALARY_COLUMN].median())
    df["HighSalary"] = (df[SALARY_COLUMN] >= median_salary).astype("int8")
    df["SurveyYear"] = year

    language_counts = {
        language: int(df[f"lang_{language}"].sum()) for language in ALL_LANGUAGES
    }
    eligible = [
        language for language, count in language_counts.items() if count >= MIN_SAMPLE_SIZE
    ]

    metadata: dict[str, object] = {
        "survey_year": year,
        "initial_rows": initial_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_after_duplicate_filter": rows_after_duplicate_filter,
        "rows_after_required_value_filter": after_required_filter,
        "rows_after_country_filter": after_country_filter,
        "final_rows": len(df),
        "countries": list(country_list) if country_list else "ALL",
        "salary_outlier_rule": {
            "lower_quantile": SALARY_LOWER_QUANTILE,
            "upper_quantile": SALARY_UPPER_QUANTILE,
            "lower_usd": lower,
            "upper_usd": upper,
        },
        "median_salary_usd": median_salary,
        "minimum_language_sample_size": MIN_SAMPLE_SIZE,
        "language_counts": language_counts,
        "eligible_languages": eligible,
        "excluded_languages": sorted(set(ALL_LANGUAGES) - set(eligible)),
    }
    return df.reset_index(drop=True), metadata


def write_outputs(
    df: pd.DataFrame,
    metadata: dict[str, object],
    create_domain_files: bool,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    common_path = DATA_DIR / "results.csv"
    df.to_csv(common_path, index=False, encoding="utf-8-sig")

    metadata_path = DATA_DIR / "preprocessing_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if create_domain_files:
        eligible = set(metadata["eligible_languages"])
        for domain, configured_languages in LANGUAGE_DOMAINS.items():
            languages = [lang for lang in configured_languages if lang in eligible]
            domain_dir = PROJECT_ROOT / domain
            domain_dir.mkdir(parents=True, exist_ok=True)

            if languages:
                mask = df[[f"lang_{lang}" for lang in languages]].any(axis=1)
                domain_df = df.loc[mask].copy()
            else:
                domain_df = df.iloc[0:0].copy()
            domain_df.to_csv(domain_dir / "data.csv", index=False, encoding="utf-8-sig")

        ml_dir = PROJECT_ROOT / "domain6_ml"
        ml_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(ml_dir / "data.csv", index=False, encoding="utf-8-sig")

    print(f"[완료] 공통 데이터: {common_path} ({len(df):,}행)")
    print(f"[완료] 전처리 기록: {metadata_path}")


def main() -> None:
    args = parse_args()
    source = args.source or OFFICIAL_DATA_URLS[args.year]
    raw_path = DATA_DIR / f"raw_results_{args.year}.csv"
    source_path = obtain_source(source, raw_path, args.force_download)

    print(f"[읽기] {source_path}")
    raw = pd.read_csv(source_path, low_memory=False, encoding="utf-8-sig")
    selected_countries = args.countries if args.countries is not None else COUNTRIES
    cleaned, metadata = clean_data(raw, args.year, selected_countries)
    write_outputs(cleaned, metadata, args.write_domain_files)


if __name__ == "__main__":
    main()
