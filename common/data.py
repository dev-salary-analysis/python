"""Shared data loader used by all six analysis domains."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.config import DATA_DIR, LANGUAGE_DOMAINS


PROCESSED_DATA_PATH = DATA_DIR / "results.csv"
METADATA_PATH = DATA_DIR / "preprocessing_metadata.json"


def load_all(path: str | Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the single cleaned dataset. Domain 6 should use this function."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"공통 데이터가 없습니다: {data_path}\n"
            "먼저 `python scripts/build_dataset.py --year 2025`를 실행하세요."
        )
    return pd.read_csv(data_path, low_memory=False, encoding="utf-8-sig")


def load_metadata(path: str | Path = METADATA_PATH) -> dict[str, object]:
    """Load sample counts and thresholds recorded during preprocessing."""
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"전처리 메타데이터가 없습니다: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def domain_languages(
    domain: str,
    *,
    eligible_only: bool = True,
    metadata_path: str | Path = METADATA_PATH,
) -> list[str]:
    """Return configured languages, optionally excluding samples below the minimum."""
    if domain not in LANGUAGE_DOMAINS:
        valid = ", ".join(LANGUAGE_DOMAINS)
        raise ValueError(f"알 수 없는 도메인입니다: {domain}. 사용 가능: {valid}")

    languages = list(LANGUAGE_DOMAINS[domain])
    if not eligible_only:
        return languages

    eligible = set(load_metadata(metadata_path)["eligible_languages"])
    return [language for language in languages if language in eligible]


def load_domain(
    domain: str,
    path: str | Path = PROCESSED_DATA_PATH,
    *,
    eligible_only: bool = True,
    metadata_path: str | Path = METADATA_PATH,
) -> pd.DataFrame:
    """Select one language domain from the shared CSV without substring matching.

    A respondent who uses multiple language families can legitimately appear in
    more than one domain. This is intentional and must be noted in the report.
    """
    df = load_all(path)
    languages = domain_languages(
        domain, eligible_only=eligible_only, metadata_path=metadata_path
    )
    if not languages:
        return df.iloc[0:0].copy()

    flag_columns = [f"lang_{language}" for language in languages]
    missing = [column for column in flag_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"공통 데이터에 언어 플래그가 없습니다: {missing}. 데이터를 다시 생성하세요."
        )

    flags = df[flag_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df.loc[flags.eq(1).any(axis=1)].copy()
