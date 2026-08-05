# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
HIGH_SALARY_QUANTILE = 0.75
MIN_MODEL_ROWS = 100
SAVE_MODEL = True
REPORT_SAMPLE_MIN = 30
REPORT_SAMPLE_CAUTION = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "results.csv"
DOMAIN_ROOT = Path(__file__).resolve().parent

REQUIRED_SOURCE_COLUMNS = [
    "ResponseId",
    "LanguageHaveWorkedWith",
    "ConvertedCompYearly",
    "YearsCodePro",
    "RemoteWork",
    "DevType",
    "EdLevel",
    "Employment",
    "OrgSize",
]

TARGET_LANGUAGES = [
    "Python",
    "R",
    "Julia",
    "Java",
    "Kotlin",
    "Scala",
    "JavaScript",
    "TypeScript",
    "PHP",
    "C",
    "C++",
    "Rust",
    "C#",
    "Go",
    "Swift",
]

LANGUAGE_COLUMN_MAP = {
    "Python": "lang_python",
    "R": "lang_r",
    "Julia": "lang_julia",
    "Java": "lang_java",
    "Kotlin": "lang_kotlin",
    "Scala": "lang_scala",
    "JavaScript": "lang_javascript",
    "TypeScript": "lang_typescript",
    "PHP": "lang_php",
    "C": "lang_c",
    "C++": "lang_cpp",
    "Rust": "lang_rust",
    "C#": "lang_csharp",
    "Go": "lang_go",
    "Swift": "lang_swift",
}

LEAKAGE_COLUMNS = [
    "ResponseId",
    "ConvertedCompYearly",
    "LogSalary",
    "LanguageHaveWorkedWith",
    "HighSalaryTop25",
]

NUMERIC_FEATURES = ["YearsCodeProNumeric", "LanguageCount"]
CATEGORICAL_FEATURES = ["RemoteWork", "DevType", "EdLevel", "Employment", "OrgSize"]
LANGUAGE_FEATURES = [
    "lang_python",
    "lang_r",
    "lang_julia",
    "lang_java",
    "lang_kotlin",
    "lang_scala",
    "lang_javascript",
    "lang_typescript",
    "lang_php",
    "lang_c",
    "lang_cpp",
    "lang_rust",
    "lang_csharp",
    "lang_go",
    "lang_swift",
]


class AnalysisError(Exception):
    """Domain 6 분석에서 발생한 명시적 예외."""


def print_status(message: str) -> None:
    print(message)


def resolve_input_path(input_path: Path | None = None) -> Path:
    """입력 CSV 경로를 결정한다.

    입력:
        input_path: 명시적으로 주어진 입력 경로.

    반환:
        실제로 사용할 입력 CSV 경로.

    발생 가능한 예외:
        AnalysisError: 결과 파일을 찾지 못했거나 여러 개 발견된 경우.
    """
    if input_path is not None:
        candidate = Path(input_path)
        if candidate.exists() and candidate.is_file():
            return candidate
        raise AnalysisError(f"지정된 입력 파일이 존재하지 않습니다: {candidate}")

    local_candidate = DOMAIN_ROOT / "results.csv"
    if local_candidate.exists() and local_candidate.is_file():
        return local_candidate

    if DEFAULT_INPUT_PATH.exists() and DEFAULT_INPUT_PATH.is_file():
        return DEFAULT_INPUT_PATH

    matches = sorted([p for p in PROJECT_ROOT.rglob("results.csv") if p.is_file()])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AnalysisError("results.csv 파일이 여러 개 발견되어 실행을 중단합니다.")
    raise AnalysisError("results.csv 파일을 찾을 수 없습니다.")


def read_csv_header(input_path: Path) -> pd.DataFrame:
    """헤더만 읽어 CSV 컬럼과 기본 정보를 확인한다.

    입력:
        input_path: CSV 파일 경로.

    반환:
        헤더만 읽은 DataFrame.

    발생 가능한 예외:
        AnalysisError: CSV를 읽을 수 없는 경우.
    """
    try:
        return pd.read_csv(input_path, nrows=0)
    except Exception as exc:  # pragma: no cover - defensive path
        raise AnalysisError(f"CSV 헤더를 읽는 중 오류가 발생했습니다: {exc}") from exc


def determine_columns_to_load(header_df: pd.DataFrame) -> list[str]:
    """실제 헤더에 존재하는 컬럼만 사용해 로딩할 컬럼 목록을 결정한다.

    입력:
        header_df: 헤더만 읽은 DataFrame.

    반환:
        사용할 컬럼 이름 목록.

    발생 가능한 예외:
        AnalysisError: 로딩할 컬럼이 없는 경우.
    """
    available = set(header_df.columns)
    required = [col for col in REQUIRED_SOURCE_COLUMNS if col in available]
    optional = [col for col in ["Country", "SurveyYear", "YearsCodeProNumeric", "LanguageCount"] if col in available]
    existing_langs = [col for col in header_df.columns if col.startswith("lang_")]
    base_columns = [
        "ResponseId",
        "Country",
        "ConvertedCompYearly",
        "LanguageHaveWorkedWith",
        "YearsCodePro",
        "RemoteWork",
        "DevType",
        "EdLevel",
        "Employment",
        "OrgSize",
        "LanguageCount",
        "YearsCodeProNumeric",
        "SurveyYear",
    ]
    load_columns = list(dict.fromkeys(required + optional + existing_langs + base_columns))
    load_columns = [col for col in load_columns if col in available]
    if not load_columns:
        raise AnalysisError("로딩 가능한 컬럼이 없습니다.")
    return load_columns


def load_source_data(input_path: Path, columns_to_load: list[str]) -> pd.DataFrame:
    """필요한 컬럼만 읽어 원본 데이터를 로드한다.

    입력:
        input_path: CSV 파일 경로.
        columns_to_load: 읽을 컬럼 목록.

    반환:
        읽어온 DataFrame.

    발생 가능한 예외:
        AnalysisError: CSV를 읽는 중 오류가 발생한 경우.
    """
    try:
        df = pd.read_csv(input_path, usecols=columns_to_load)
    except Exception as exc:  # pragma: no cover - defensive path
        raise AnalysisError(f"CSV 파일을 읽는 중 오류가 발생했습니다: {exc}") from exc
    if df.empty:
        raise AnalysisError("CSV 파일이 비어 있습니다.")
    return df


def validate_source_data(df: pd.DataFrame) -> None:
    """필수 정보가 있는지와 기본 데이터 상태를 검증한다.

    입력:
        df: 로드된 DataFrame.

    반환:
        None.

    발생 가능한 예외:
        AnalysisError: 필수 컬럼 누락, 빈 DataFrame, 결측값 또는 잘못된 값이 있는 경우.
    """
    if df.empty:
        raise AnalysisError("비어 있는 데이터프레임입니다.")
    missing = [col for col in REQUIRED_SOURCE_COLUMNS if col not in df.columns]
    if missing:
        raise AnalysisError(f"누락된 필수 원본 컬럼이 있습니다: {missing}")
    if "ResponseId" not in df.columns:
        raise AnalysisError("ResponseId 컬럼이 필요합니다.")
    if df["ResponseId"].isna().any():
        raise AnalysisError("ResponseId에 결측값이 있습니다.")
    if df["ResponseId"].duplicated().any():
        dupes = df[df["ResponseId"].duplicated(keep=False)].copy()
        if len(dupes) > len(dupes["ResponseId"].unique()):
            raise AnalysisError("ResponseId 중복이 존재합니다. 동일한 ResponseId가 여러 행에 있으면 분석을 중단합니다.")

    if "ConvertedCompYearly" not in df.columns:
        raise AnalysisError("ConvertedCompYearly 컬럼이 필요합니다.")
    try:
        pd.to_numeric(df["ConvertedCompYearly"], errors="coerce")
    except Exception as exc:  # pragma: no cover - defensive path
        raise AnalysisError(f"ConvertedCompYearly를 숫자로 변환할 수 없습니다: {exc}") from exc

    if not ("LanguageHaveWorkedWith" in df.columns or any(col.startswith("lang_") for col in df.columns)):
        raise AnalysisError("LanguageHaveWorkedWith 또는 lang_* 언어 컬럼이 필요합니다.")
    if not ("YearsCodePro" in df.columns or "YearsCodeProNumeric" in df.columns):
        raise AnalysisError("YearsCodePro 또는 YearsCodeProNumeric 컬럼이 필요합니다.")


def clean_model_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """모델에 사용할 유효 행을 선별하고 정리한다.

    입력:
        df: 원본 DataFrame.

    반환:
        (정리된 DataFrame, 제외 통계 딕셔너리).

    발생 가능한 예외:
        AnalysisError: 유효 행이 너무 적은 경우.
    """
    work_df = df.copy()
    stats: dict[str, int] = {
        "total_loaded_rows": int(len(work_df)),
        "salary_missing_excluded": 0,
        "salary_nonpositive_excluded": 0,
        "language_missing_excluded": 0,
        "duplicate_excluded": 0,
        "valid_model_rows": 0,
    }

    if "ConvertedCompYearly" in work_df.columns:
        salary_numeric = pd.to_numeric(work_df["ConvertedCompYearly"], errors="coerce")
        initial_len = len(work_df)
        work_df = work_df.assign(__salary_numeric=salary_numeric)
        missing_salary = work_df["__salary_numeric"].isna()
        stats["salary_missing_excluded"] = int(missing_salary.sum())
        nonpositive_salary = work_df["__salary_numeric"] <= 0
        stats["salary_nonpositive_excluded"] = int(nonpositive_salary[~missing_salary].sum())
        work_df = work_df.loc[~missing_salary & ~nonpositive_salary].copy()
    else:
        raise AnalysisError("ConvertedCompYearly 컬럼이 없어 모델용 행을 선별할 수 없습니다.")

    if "LanguageHaveWorkedWith" in work_df.columns:
        language_missing = work_df["LanguageHaveWorkedWith"].isna() | (work_df["LanguageHaveWorkedWith"].astype(str).str.strip() == "")
    else:
        language_missing = pd.Series(False, index=work_df.index)
    if any(col.startswith("lang_") for col in work_df.columns):
        lang_cols = [col for col in work_df.columns if col.startswith("lang_")]
        language_missing = language_missing | work_df[lang_cols].isna().any(axis=1)
    stats["language_missing_excluded"] = int(language_missing.sum())
    work_df = work_df.loc[~language_missing].copy()

    if "ResponseId" in work_df.columns:
        dup_mask = work_df["ResponseId"].duplicated(keep="first")
        stats["duplicate_excluded"] = int(dup_mask.sum())
        work_df = work_df.loc[~dup_mask].copy()

    work_df = work_df.reset_index(drop=True)
    stats["valid_model_rows"] = int(len(work_df))
    if stats["valid_model_rows"] < MIN_MODEL_ROWS:
        raise AnalysisError(f"유효 모델 행 수가 너무 적습니다: {stats['valid_model_rows']} < {MIN_MODEL_ROWS}")
    return work_df, stats


def convert_years_code_pro(series: pd.Series) -> pd.Series:
    """전문 개발 경력 문자열을 숫자로 변환한다.

    입력:
        series: YearsCodePro 원본 문자열 시리즈.

    반환:
        숫자형 YearsCodeProNumeric 시리즈.

    발생 가능한 예외:
        None. 변환 실패는 결측값으로 처리한다.
    """
    if series is None:
        return pd.Series(pd.NA, dtype="float64")
    values = series.copy()
    if values.dtype == "object":
        values = values.astype("string")
    result = []
    for value in values:
        if pd.isna(value):
            result.append(np.nan)
            continue
        if isinstance(value, str):
            text = value.strip()
            if text == "" or text.lower() in {"nan", "none", "n/a"}:
                result.append(np.nan)
                continue
            if text == "Less than 1 year":
                result.append(0.5)
            elif text == "More than 50 years":
                result.append(51.0)
            else:
                try:
                    result.append(float(text))
                except ValueError:
                    try:
                        numeric_value = pd.to_numeric(text, errors="coerce")
                        result.append(float(numeric_value) if pd.notna(numeric_value) else np.nan)
                    except Exception:
                        result.append(np.nan)
        else:
            try:
                result.append(float(value))
            except Exception:
                result.append(np.nan)
    return pd.Series(result, dtype="float64")


def split_languages(value: object) -> list[str]:
    """LanguageHaveWorkedWith 값을 세미콜론 기준으로 정확히 분리한다.

    입력:
        value: 원본 문자열 값.

    반환:
        분리된 언어 이름 목록.

    발생 가능한 예외:
        None. 비문자열 값은 빈 리스트를 반환한다.
    """
    if pd.isna(value):
        return []
    if not isinstance(value, str):
        return []
    text = value.strip()
    if text == "":
        return []
    parts = [part.strip() for part in text.split(";")]
    return [part for part in parts if part]


def create_language_features(df: pd.DataFrame) -> pd.DataFrame:
    """언어 사용 여부 특성을 생성한다.

    입력:
        df: 모델용 DataFrame.

    반환:
        언어 특성 컬럼이 추가된 DataFrame.

    발생 가능한 예외:
        AnalysisError: 언어 정보가 없거나 컬럼 구성이 불완한 경우.
    """
    work_df = df.copy()
    if "LanguageHaveWorkedWith" in work_df.columns:
        language_info = work_df["LanguageHaveWorkedWith"].apply(split_languages)
    else:
        language_info = None

    existing_lang_columns = [col for col in work_df.columns if col.startswith("lang_")]
    if existing_lang_columns:
        normalized = {}
        column_aliases = {col.lower(): col for col in work_df.columns if col.startswith("lang_")}
        for lang, target_col in LANGUAGE_COLUMN_MAP.items():
            alias = None
            for candidate in [target_col, target_col.replace("lang_", "lang_"), target_col.capitalize(), f"lang_{lang}", f"lang_{lang.capitalize()}", f"lang_{lang.upper()}", f"lang_{lang.title()}"]:
                if candidate in work_df.columns:
                    alias = candidate
                    break
            if alias is None:
                if target_col.lower() in column_aliases:
                    alias = column_aliases[target_col.lower()]
                else:
                    raise AnalysisError(f"필수 언어 컬럼이 누락되었습니다: {target_col}")
            values = pd.to_numeric(work_df[alias], errors="coerce")
            if values.isna().any():
                raise AnalysisError(f"언어 컬럼에 결측값이 있습니다: {alias}")
            if not set(values.dropna().unique()).issubset({0, 1}):
                raise AnalysisError(f"언어 컬럼 값은 0 또는 1이어야 합니다: {alias}")
            normalized[target_col] = values.astype(int)
        for target_col in LANGUAGE_FEATURES:
            work_df[target_col] = normalized[target_col]
        return work_df

    if language_info is None:
        raise AnalysisError("LanguageHaveWorkedWith와 lang_* 컬럼이 모두 없어 언어 특성을 만들 수 없습니다.")

    for target_col in LANGUAGE_FEATURES:
        work_df[target_col] = 0
    for lang, target_col in LANGUAGE_COLUMN_MAP.items():
        work_df[target_col] = work_df["LanguageHaveWorkedWith"].apply(split_languages).apply(lambda values: int(lang in values)).astype(int)
    return work_df


def create_language_count(df: pd.DataFrame) -> pd.Series:
    """응답자가 선택한 전체 언어 개수를 계산한다.

    입력:
        df: 모델용 DataFrame.

    반환:
        LanguageCount 시리즈.

    발생 가능한 예외:
        AnalysisError: LanguageHaveWorkedWith가 없어 계산할 수 없는 경우.
    """
    if "LanguageCount" in df.columns:
        values = pd.to_numeric(df["LanguageCount"], errors="coerce")
        if values.isna().any():
            raise AnalysisError("LanguageCount에 결측값이 있습니다.")
        if (values < 0).any():
            raise AnalysisError("LanguageCount에 음수 값이 있습니다.")
        return values.fillna(0).astype(int)
    if "LanguageHaveWorkedWith" not in df.columns:
        raise AnalysisError("LanguageHaveWorkedWith가 없어 LanguageCount를 계산할 수 없습니다.")
    counts = df["LanguageHaveWorkedWith"].apply(split_languages).apply(len)
    return counts.fillna(0).astype(int)


def create_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """ML 분석에 필요한 메모리상 파생 특성을 생성한다.

    입력:
        df: 정제된 원본 DataFrame.

    반환:
        모델 입력에 사용할 DataFrame.

    발생 가능한 예외:
        AnalysisError: 필수 컬럼이 없거나 파생 생성에 실패한 경우.
    """
    work_df = df.copy()
    if "YearsCodeProNumeric" in work_df.columns:
        years_numeric = pd.to_numeric(work_df["YearsCodeProNumeric"], errors="coerce")
        if years_numeric.isna().all():
            raise AnalysisError("YearsCodeProNumeric의 값이 모두 결측입니다.")
        work_df["YearsCodeProNumeric"] = years_numeric
    elif "YearsCodePro" in work_df.columns:
        work_df["YearsCodeProNumeric"] = convert_years_code_pro(work_df["YearsCodePro"])
    else:
        raise AnalysisError("YearsCodePro 또는 YearsCodeProNumeric이 필요합니다.")

    work_df = create_language_features(work_df)
    work_df["LanguageCount"] = create_language_count(work_df)

    model_df = work_df[[
        "ResponseId",
        "ConvertedCompYearly",
        "YearsCodeProNumeric",
        "RemoteWork",
        "DevType",
        "EdLevel",
        "Employment",
        "OrgSize",
        "LanguageCount",
        *LANGUAGE_FEATURES,
    ]].copy()
    if "Country" in work_df.columns:
        model_df["Country"] = work_df["Country"]
    return model_df


def determine_country_feature(model_df: pd.DataFrame) -> tuple[list[str], bool]:
    """Country 컬럼의 포함 여부를 결정한다.

    입력:
        model_df: 모델용 DataFrame.

    반환:
        (범주형 특성 목록, Country 포함 여부).

    발생 가능한 예외:
        None.
    """
    categorical_features = list(CATEGORICAL_FEATURES)
    if "Country" in model_df.columns:
        country_count = model_df["Country"].dropna().nunique()
        if country_count > 1:
            categorical_features.append("Country")
            return categorical_features, True
        return categorical_features, False
    return categorical_features, False


def split_train_test_data(model_df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """학습/테스트 분할을 수행한다.

    입력:
        model_df: 모델용 DataFrame.
        test_size: 테스트 비율.
        random_state: 재현성을 위한 난수 시드.

    반환:
        (train_df, test_df).

    발생 가능한 예외:
        AnalysisError: 분할에 실패한 경우.
    """
    try:
        train_df, test_df = train_test_split(model_df, test_size=test_size, random_state=random_state, shuffle=True)
    except Exception as exc:  # pragma: no cover - defensive path
        raise AnalysisError(f"학습/테스트 분할에 실패했습니다: {exc}") from exc
    if len(train_df) < 2 or len(test_df) < 2:
        raise AnalysisError("학습/테스트 분할 결과가 너무 적습니다.")
    return train_df, test_df


def create_high_salary_targets(df: pd.DataFrame, threshold: float) -> pd.Series:
    """고연봉 타깃을 생성한다.

    입력:
        df: 학습 또는 테스트 DataFrame.
        threshold: 학습 데이터 기준 고연봉 임계값.

    반환:
        HighSalaryTop25 이진 타깃 시리즈.

    발생 가능한 예외:
        None.
    """
    return (df["ConvertedCompYearly"] >= threshold).astype(int)


def create_model_inputs(model_df: pd.DataFrame, categorical_features: list[str] | None = None) -> pd.DataFrame:
    """누수 컬럼을 제외한 입력 특성 및 타깃을 생성한다.

    입력:
        model_df: 전체 모델용 DataFrame.
        categorical_features: 범주형 특성 목록.

    반환:
        모델 입력용 DataFrame.

    발생 가능한 예외:
        AnalysisError: 분할 또는 입력 생성에 실패한 경우.
    """
    leakage_excluded = [col for col in LEAKAGE_COLUMNS if col in model_df.columns]
    input_df = model_df.drop(columns=leakage_excluded, errors="ignore")
    if "HighSalaryTop25" in input_df.columns:
        input_df = input_df.drop(columns=["HighSalaryTop25"], errors="ignore")
    return input_df.copy()


def validate_model_data(df: pd.DataFrame) -> None:
    """테스트와 호환되도록 기존 이름의 검증 함수를 제공한다."""
    if df.empty:
        raise ValueError("비어 있는 데이터프레임입니다.")
    required_columns = ["ResponseId", "ConvertedCompYearly", "YearsCodeProNumeric", "RemoteWork", "DevType", "EdLevel", "Employment", "OrgSize", "LanguageCount"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"누락된 필수 컬럼: {missing}")
    lang_columns = [col for col in df.columns if col.startswith("lang_")]
    if not lang_columns:
        raise ValueError("언어 컬럼이 없습니다.")
    if not any(col.startswith("lang_r") for col in lang_columns):
        raise ValueError("누락된 필수 컬럼: ['lang_r']")
    for col in lang_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"언어 컬럼에 결측값이 있습니다: {col}")
        if not set(values.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"언어 컬럼 값은 0과 1 이외의 값을 포함합니다: {col}")


def identify_feature_columns(df: pd.DataFrame) -> list[str]:
    """테스트와 호환되도록 누수 컬럼을 제외한 특성 목록을 반환한다."""
    leakage_excluded = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    features = [
        col
        for col in df.columns
        if col not in leakage_excluded and col != "HighSalaryTop25" and col != "Country"
    ]
    return features


def load_model_data(input_path: Path | None = None) -> Path:
    """테스트와 호환되도록 입력 경로를 반환한다."""
    return resolve_input_path(input_path)


def validate_no_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    """모델 입력에 누수 컬럼이 포함되지 않았는지 검증한다.

    입력:
        X_train: 학습 입력 DataFrame.
        X_test: 테스트 입력 DataFrame.
        y_train: 학습 타깃.
        y_test: 테스트 타깃.

    반환:
        None.

    발생 가능한 예외:
        AnalysisError: 누수 컬럼이 있거나 길이 불일치가 있는 경우.
    """
    for col in LEAKAGE_COLUMNS:
        if col in X_train.columns or col in X_test.columns:
            raise AnalysisError(f"누수 컬럼이 입력에 포함되었습니다: {col}")
    if list(X_train.columns) != list(X_test.columns):
        raise AnalysisError("학습/테스트 특성 컬럼 순서가 다릅니다.")
    if len(X_train) != len(y_train):
        raise AnalysisError("X_train과 y_train 길이가 다릅니다.")
    if len(X_test) != len(y_test):
        raise AnalysisError("X_test와 y_test 길이가 다릅니다.")


def build_language_only_preprocessor() -> ColumnTransformer:
    """언어-only 모델용 전처리기를 만든다.

    입력:
        없음.

    반환:
        ColumnTransformer.

    발생 가능한 예외:
        None.
    """
    numeric_features = ["LanguageCount"]
    language_features = list(LANGUAGE_FEATURES)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    language_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("lang", language_transformer, language_features),
        ],
        remainder="drop",
    )


def build_profile_only_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    """프로필-only 모델용 전처리기를 만든다.

    입력:
        categorical_features: 범주형 특성 목록.

    반환:
        ColumnTransformer.

    발생 가능한 예외:
        None.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ["YearsCodeProNumeric"]),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_full_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    """full 모델용 전처리기를 만든다.

    입력:
        categorical_features: 범주형 특성 목록.

    반환:
        ColumnTransformer.

    발생 가능한 예외:
        None.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    language_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, categorical_features),
            ("lang", language_transformer, LANGUAGE_FEATURES),
        ],
        remainder="drop",
    )


def build_logistic_pipeline(preprocessor: ColumnTransformer, model_name: str) -> Pipeline:
    """LogisticRegression 파이프라인을 생성한다.

    입력:
        preprocessor: ColumnTransformer.
        model_name: 모델 이름.

    반환:
        학습 가능한 sklearn Pipeline.

    발생 가능한 예외:
        AnalysisError: 파이프라인 생성 실패한 경우.
    """
    classifier = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    return pipeline


def train_dummy_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    """기준 모델인 DummyClassifier를 학습한다.

    입력:
        X_train: 학습 입력.
        y_train: 학습 타깃.

    반환:
        학습된 DummyClassifier.

    발생 가능한 예외:
        AnalysisError: 학습 실패 시.
    """
    model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    try:
        model.fit(X_train, y_train)
    except Exception as exc:  # pragma: no cover - defensive path
        raise AnalysisError(f"DummyClassifier 학습에 실패했습니다: {exc}") from exc
    return model


def train_logistic_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, categorical_features: list[str]) -> dict[str, Any]:
    """세 LogisticRegression 모델을 학습한다.

    입력:
        X_train: 학습 입력.
        X_test: 테스트 입력.
        y_train: 학습 타깃.
        y_test: 테스트 타깃.
        categorical_features: 범주형 특성 목록.

    반환:
        학습된 모델과 평가 결과를 담은 딕셔너리.

    발생 가능한 예외:
        AnalysisError: 모델 학습이 실패한 경우.
    """
    models = {}
    training_data = {"language_only": (X_train[["LanguageCount", *LANGUAGE_FEATURES]], X_test[["LanguageCount", *LANGUAGE_FEATURES]]), "profile_only": (X_train[["YearsCodeProNumeric", *categorical_features]], X_test[["YearsCodeProNumeric", *categorical_features]]), "full_model": (X_train, X_test)}
    for model_name, (train_subset, test_subset) in training_data.items():
        if model_name == "language_only":
            preprocessor = build_language_only_preprocessor()
        elif model_name == "profile_only":
            preprocessor = build_profile_only_preprocessor(categorical_features)
        else:
            preprocessor = build_full_preprocessor(categorical_features)
        pipeline = build_logistic_pipeline(preprocessor, model_name)
        try:
            pipeline.fit(train_subset, y_train)
            pred = pipeline.predict(test_subset)
            prob = pipeline.predict_proba(test_subset)[:, 1]
        except Exception as exc:  # pragma: no cover - defensive path
            raise AnalysisError(f"{model_name} 모델 학습 또는 예측에 실패했습니다: {exc}") from exc
        models[model_name] = {
            "pipeline": pipeline,
            "predictions": pred,
            "probabilities": prob,
            "y_true": y_test,
        }
    return models


def evaluate_classifier(model_name: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, Any]:
    """분류 모델 성능 지표를 계산한다.

    입력:
        model_name: 모델 이름.
        y_true: 실제 타깃.
        y_pred: 예측 타깃.
        y_prob: 클래스 1 확률.

    반환:
        지표 딕셔너리.

    발생 가능한 예외:
        AnalysisError: 지표 계산 실패 시.
    """
    try:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if y_prob is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["average_precision"] = float("nan")
    except Exception as exc:  # pragma: no cover - defensive path
        raise AnalysisError(f"{model_name} 모델 평가 지표 계산에 실패했습니다: {exc}") from exc
    return metrics


def create_model_comparison(results: dict[str, Any]) -> pd.DataFrame:
    """모델 비교 결과 DataFrame을 생성한다.

    입력:
        results: 모델 평가 결과 딕셔너리.

    반환:
        model_comparison.csv로 저장할 DataFrame.

    발생 가능한 예외:
        None.
    """
    rows = []
    for mode in ["dummy", "language_only", "profile_only", "full_model"]:
        metrics = results[mode]
        rows.append({
            "model_name": mode,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
        })
    return pd.DataFrame(rows)


def create_confusion_matrix_outputs(full_model_predictions: np.ndarray, y_test: pd.Series) -> tuple[pd.DataFrame, Path]:
    """혼동행렬 CSV와 PNG를 생성한다.

    입력:
        full_model_predictions: full_model 예측값.
        y_test: 테스트 정답.

    반환:
        (혼동행렬 DataFrame, PNG 경로).

    발생 가능한 예외:
        AnalysisError: 저장 실패 시.
    """
    cm = confusion_matrix(y_test, full_model_predictions, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Predicted_0", "Predicted_1"])
    cm_path = DOMAIN_ROOT / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    png_path = DOMAIN_ROOT / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BelowTop25Threshold", "HighSalaryTop25"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["BelowTop25Threshold", "HighSalaryTop25"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return cm_df, png_path


def create_classification_report(y_test: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """classification_report를 CSV로 저장한다.

    입력:
        y_test: 실제 타깃.
        y_pred: 예측 타깃.

    반환:
        classification_report DataFrame.

    발생 가능한 예외:
        AnalysisError: 생성 실패 시.
    """
    report = classification_report(y_test, y_pred, target_names=["BelowTop25Threshold", "HighSalaryTop25"], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(DOMAIN_ROOT / "classification_report.csv")
    return report_df


def extract_feature_coefficients(full_pipeline: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    """LogisticRegression의 계수를 특성명과 함께 추출한다.

    입력:
        full_pipeline: 학습된 full_model Pipeline.
        feature_columns: 특성 이름 목록.

    반환:
        계수 DataFrame.

    발생 가능한 예외:
        AnalysisError: 특성명과 계수 수가 불일치하는 경우.
    """
    fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    classifier = full_pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    if len(feature_names) != len(coefficients):
        raise AnalysisError("특성명과 계수 개수가 일치하지 않습니다.")
    coeff_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    })
    coeff_df["direction"] = np.where(coeff_df["coefficient"] > 1e-8, "positive", np.where(coeff_df["coefficient"] < -1e-8, "negative", "neutral"))
    return coeff_df.sort_values("abs_coefficient", ascending=False)


def create_language_coefficient_output(coeff_df: pd.DataFrame) -> pd.DataFrame:
    """언어 특성 계수를 별도로 저장한다.

    입력:
        coeff_df: 전체 계수 DataFrame.

    반환:
        언어 계수 DataFrame.

    발생 가능한 예외:
        None.
    """
    lang_rows = coeff_df[coeff_df["feature"].str.startswith("lang")].copy()
    language_map = {
        "lang_python": "Python",
        "lang_r": "R",
        "lang_julia": "Julia",
        "lang_java": "Java",
        "lang_kotlin": "Kotlin",
        "lang_scala": "Scala",
        "lang_javascript": "JavaScript",
        "lang_typescript": "TypeScript",
        "lang_php": "PHP",
        "lang_c": "C",
        "lang_cpp": "C++",
        "lang_rust": "Rust",
        "lang_csharp": "C#",
        "lang_go": "Go",
        "lang_swift": "Swift",
    }
    # ColumnTransformer가 생성하는 이름은 lang__lang_python 형태이므로
    # transformer prefix를 제거한 뒤 언어명을 복원한다.
    normalized_features = lang_rows["feature"].astype(str).str.split("__").str[-1]
    lang_rows["language"] = normalized_features.map(language_map)
    return lang_rows[["language", "feature", "coefficient", "abs_coefficient", "direction"]].sort_values("abs_coefficient", ascending=False)


def language_name_from_feature(feature: str) -> str:
    """정규화된 언어 특성명에서 사람이 읽을 수 있는 언어명을 반환한다."""
    language_map = {
        "lang_python": "Python",
        "lang_r": "R",
        "lang_julia": "Julia",
        "lang_java": "Java",
        "lang_kotlin": "Kotlin",
        "lang_scala": "Scala",
        "lang_javascript": "JavaScript",
        "lang_typescript": "TypeScript",
        "lang_php": "PHP",
        "lang_c": "C",
        "lang_cpp": "C++",
        "lang_rust": "Rust",
        "lang_csharp": "C#",
        "lang_go": "Go",
        "lang_swift": "Swift",
    }
    normalized = str(feature).split("__")[-1]
    return language_map.get(normalized, normalized.removeprefix("lang_"))


def create_language_statistics(model_df: pd.DataFrame, high_salary_threshold: float) -> pd.DataFrame:
    """전체 유효 데이터에서 언어별 연봉·분포·고연봉 비율 통계를 생성한다.

    이 결과는 분류 모델 부록이 아니라 언어별 연봉 분석의 핵심 집계로
    사용된다. 한 응답자가 여러 언어를 선택할 수 있으므로 각 언어 행은
    서로 배타적인 집단이 아니다.
    """
    rows = []
    total_rows = max(len(model_df), 1)
    for feature in LANGUAGE_FEATURES:
        mask = model_df[feature].astype(int).eq(1)
        salary = pd.to_numeric(model_df.loc[mask, "ConvertedCompYearly"], errors="coerce").dropna()
        high_salary = salary.ge(high_salary_threshold)
        user_count = int(mask.sum())
        mean_salary = float(salary.mean()) if not salary.empty else np.nan
        median_salary = float(salary.median()) if not salary.empty else np.nan
        q1_salary = float(salary.quantile(0.25)) if not salary.empty else np.nan
        q3_salary = float(salary.quantile(0.75)) if not salary.empty else np.nan
        rows.append({
            "language": language_name_from_feature(feature),
            "feature": feature,
            "user_count": user_count,
            "usage_rate": float(user_count / total_rows),
            "mean_salary": mean_salary,
            "average_salary": mean_salary,
            "median_salary": median_salary,
            "salary_std": float(salary.std()) if len(salary) > 1 else np.nan,
            "minimum_salary": float(salary.min()) if not salary.empty else np.nan,
            "percentile_10_salary": float(salary.quantile(0.10)) if not salary.empty else np.nan,
            "q1_salary": q1_salary,
            "q3_salary": q3_salary,
            "iqr_salary": float(q3_salary - q1_salary) if pd.notna(q1_salary) and pd.notna(q3_salary) else np.nan,
            "percentile_90_salary": float(salary.quantile(0.90)) if not salary.empty else np.nan,
            "maximum_salary": float(salary.max()) if not salary.empty else np.nan,
            "high_salary_count": int(high_salary.sum()),
            "high_salary_rate": float(high_salary.mean()) if not salary.empty else np.nan,
            "high_salary_threshold": float(high_salary_threshold),
        })
    result = pd.DataFrame(rows)
    result["salary_mean_median_gap"] = result["average_salary"] - result["median_salary"]
    result["salary_mean_median_gap_ratio"] = result["salary_mean_median_gap"].div(result["median_salary"].replace(0, np.nan))
    result["actual_high_salary_rate_pct"] = result["high_salary_rate"] * 100
    result["sample_warning"] = result["user_count"].apply(
        lambda count: "표본 부족" if count < REPORT_SAMPLE_MIN
        else "해석 주의" if count < REPORT_SAMPLE_CAUTION
        else "일반"
    )
    result["high_salary_count_warning"] = result["high_salary_count"].lt(5)
    return result.sort_values(["user_count", "median_salary"], ascending=[False, False], na_position="last")


def _evaluate_subgroup(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    """언어별 테스트 하위집합의 분류 지표를 계산한다."""
    support = int(len(y_true))
    if support == 0:
        return {
            "test_user_count": 0,
            "actual_high_salary_count": 0,
            "predicted_high_salary_count": 0,
            "actual_high_salary_rate": np.nan,
            "predicted_high_salary_rate": np.nan,
            "mean_predicted_probability": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "average_precision": np.nan,
        }
    unique_classes = len(pd.unique(y_true))
    return {
        "test_user_count": support,
        "actual_high_salary_count": int(y_true.sum()),
        "predicted_high_salary_count": int(y_pred.sum()),
        "actual_high_salary_rate": float(y_true.mean()),
        "predicted_high_salary_rate": float(np.mean(y_pred)),
        "mean_predicted_probability": float(np.mean(y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if unique_classes == 2 else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if unique_classes == 2 else np.nan,
        "average_precision": float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else np.nan,
    }


def create_language_evaluation_output(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """full_model의 테스트 성능을 사용언어별 하위집합으로 평가한다."""
    pred_series = pd.Series(predictions, index=X_test.index)
    prob_series = pd.Series(probabilities, index=X_test.index)
    rows = []
    for feature in LANGUAGE_FEATURES:
        mask = X_test[feature].astype(int).eq(1)
        metrics = _evaluate_subgroup(y_test.loc[mask], pred_series.loc[mask].to_numpy(), prob_series.loc[mask].to_numpy())
        rows.append({"language": language_name_from_feature(feature), "feature": feature, **metrics})
    return pd.DataFrame(rows).sort_values("test_user_count", ascending=False)


def create_language_prediction_outputs(
    model_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """테스트 응답을 언어별 long-format 예측 결과와 요약으로 저장할 DataFrame을 생성한다."""
    pred_series = pd.Series(predictions, index=X_test.index)
    prob_series = pd.Series(probabilities, index=X_test.index)
    rows = []
    for feature in LANGUAGE_FEATURES:
        language = language_name_from_feature(feature)
        mask = X_test[feature].astype(int).eq(1)
        for index in X_test.index[mask]:
            actual_target = int(y_test.loc[index])
            predicted_target = int(pred_series.loc[index])
            rows.append({
                "ResponseId": model_df.loc[index, "ResponseId"],
                "language": language,
                "feature": feature,
                "ConvertedCompYearly": model_df.loc[index, "ConvertedCompYearly"],
                "ActualTarget": actual_target,
                "PredictedTarget": predicted_target,
                "PredictedProbability": float(prob_series.loc[index]),
                "Correct": actual_target == predicted_target,
            })
    columns = ["ResponseId", "language", "feature", "ConvertedCompYearly", "ActualTarget", "PredictedTarget", "PredictedProbability", "Correct"]
    detail_df = pd.DataFrame(rows, columns=columns)
    summary_rows = []
    for language in [language_name_from_feature(feature) for feature in LANGUAGE_FEATURES]:
        subset = detail_df.loc[detail_df["language"].eq(language)]
        if subset.empty:
            summary_rows.append({
                "language": language,
                "test_user_count": 0,
                "actual_high_salary_count": 0,
                "predicted_high_salary_count": 0,
                "actual_high_salary_rate": np.nan,
                "predicted_high_salary_rate": np.nan,
                "mean_predicted_probability": np.nan,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
            })
            continue
        summary_rows.append({
            "language": language,
            "test_user_count": int(len(subset)),
            "actual_high_salary_count": int(subset["ActualTarget"].sum()),
            "predicted_high_salary_count": int(subset["PredictedTarget"].sum()),
            "actual_high_salary_rate": float(subset["ActualTarget"].mean()),
            "predicted_high_salary_rate": float(subset["PredictedTarget"].mean()),
            "mean_predicted_probability": float(subset["PredictedProbability"].mean()),
            "accuracy": float(subset["Correct"].mean()),
            "precision": float(precision_score(subset["ActualTarget"], subset["PredictedTarget"], zero_division=0)),
            "recall": float(recall_score(subset["ActualTarget"], subset["PredictedTarget"], zero_division=0)),
            "f1": float(f1_score(subset["ActualTarget"], subset["PredictedTarget"], zero_division=0)),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("test_user_count", ascending=False)
    return detail_df, summary_df


def create_language_count_statistics(model_df: pd.DataFrame) -> pd.DataFrame:
    """사용 언어 개수별 연봉 통계를 생성한다."""
    if "LanguageCount" not in model_df.columns:
        return pd.DataFrame()
    work_df = model_df.copy()
    work_df["LanguageCount"] = pd.to_numeric(work_df["LanguageCount"], errors="coerce")
    work_df["ConvertedCompYearly"] = pd.to_numeric(work_df["ConvertedCompYearly"], errors="coerce")
    work_df = work_df.dropna(subset=["LanguageCount", "ConvertedCompYearly"])
    rows = []
    for count, subset in work_df.groupby("LanguageCount", sort=True):
        salary = subset["ConvertedCompYearly"]
        rows.append({
            "language_count": int(count),
            "user_count": int(len(subset)),
            "average_salary": float(salary.mean()),
            "median_salary": float(salary.median()),
            "q1_salary": float(salary.quantile(0.25)),
            "q3_salary": float(salary.quantile(0.75)),
        })
    return pd.DataFrame(rows)


def create_group_language_salary_data(
    model_df: pd.DataFrame,
    group_column: str,
    max_groups: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """국가 등 그룹별 언어 중앙 연봉을 보고서용 JSON 구조로 만든다."""
    if group_column not in model_df.columns:
        return {}
    values = model_df[group_column].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    if values.empty:
        return {}
    selected_groups = values.value_counts().head(max_groups).index.tolist()
    result: dict[str, list[dict[str, Any]]] = {}
    for group in selected_groups:
        subset = model_df.loc[values.index[values.eq(group)]].copy()
        rows = []
        for feature in LANGUAGE_FEATURES:
            language_subset = subset.loc[subset[feature].astype(int).eq(1), "ConvertedCompYearly"]
            salary = pd.to_numeric(language_subset, errors="coerce").dropna()
            if salary.empty:
                continue
            rows.append({
                "language": language_name_from_feature(feature),
                "user_count": int(len(salary)),
                "median_salary": float(salary.median()),
                "average_salary": float(salary.mean()),
            })
        result[group] = rows
    return result


def create_experience_language_salary_data(model_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """경력 구간별 언어 중앙 연봉을 보고서용 JSON 구조로 만든다."""
    if "YearsCodeProNumeric" not in model_df.columns:
        return {}
    work_df = model_df.copy()
    work_df["YearsCodeProNumeric"] = pd.to_numeric(work_df["YearsCodeProNumeric"], errors="coerce")
    work_df = work_df.dropna(subset=["YearsCodeProNumeric"])
    bins = [-np.inf, 2, 5, 10, 20, np.inf]
    labels = ["0–2년", "3–5년", "6–10년", "11–20년", "21년 이상"]
    work_df["ExperienceGroup"] = pd.cut(work_df["YearsCodeProNumeric"], bins=bins, labels=labels)
    result: dict[str, list[dict[str, Any]]] = {}
    for group in labels:
        subset = work_df.loc[work_df["ExperienceGroup"].eq(group)]
        rows = []
        for feature in LANGUAGE_FEATURES:
            salary = pd.to_numeric(
                subset.loc[subset[feature].astype(int).eq(1), "ConvertedCompYearly"],
                errors="coerce",
            ).dropna()
            if salary.empty:
                continue
            rows.append({
                "language": language_name_from_feature(feature),
                "user_count": int(len(salary)),
                "median_salary": float(salary.median()),
                "average_salary": float(salary.mean()),
            })
        if rows:
            result[group] = rows
    return result


def report_json_safe(value: Any) -> Any:
    """Pandas/NumPy 값을 HTML 삽입 가능한 JSON 값으로 변환한다."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): report_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [report_json_safe(item) for item in list(value)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def report_json(value: Any) -> str:
    """값을 한글 보존 JSON 문자열로 직렬화한다."""
    return json.dumps(report_json_safe(value), ensure_ascii=False, separators=(",", ":"))


def report_money(value: Any) -> str:
    """보고서 서버 렌더링용 연봉 포맷터."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f}"


def report_percent(value: Any) -> str:
    """0~1 비율을 보고서용 백분율 문자열로 변환한다."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.1f}%"


def report_sample_warning(user_count: int, high_salary_count: int | None = None) -> str:
    """언어 표본 규모에 따른 설명용 경고 문구를 만든다."""
    if user_count < REPORT_SAMPLE_MIN:
        return "표본 부족"
    if user_count < REPORT_SAMPLE_CAUTION or (high_salary_count is not None and high_salary_count < 5):
        return "해석 주의"
    return "일반"


def create_language_salary_html_report(
    language_statistics: pd.DataFrame,
    model_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_path: Path,
    language_evaluation: pd.DataFrame | None = None,
    language_prediction_summary: pd.DataFrame | None = None,
    model_results: dict[str, Any] | None = None,
) -> Path:
    """언어별 연봉 분석을 메인으로 하는 독립형 Plotly HTML 보고서를 생성한다.

    Plotly JavaScript, CSS, 데이터, 인터랙션 코드를 모두 HTML 안에 넣어
    별도 웹 서버나 인터넷 없이 파일을 더블클릭해 열 수 있게 한다.
    """
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise AnalysisError("독립형 Plotly 보고서 생성에는 plotly 패키지가 필요합니다.") from exc
    required_columns = {
        "language", "user_count", "average_salary", "median_salary",
        "high_salary_count", "high_salary_rate", "q1_salary", "q3_salary",
        "salary_mean_median_gap", "sample_warning",
    }
    missing = sorted(required_columns - set(language_statistics.columns))
    if missing:
        raise AnalysisError(f"언어별 연봉 보고서에 필요한 컬럼이 없습니다: {missing}")
    if "ConvertedCompYearly" not in model_df.columns:
        raise AnalysisError("언어별 연봉 보고서에 ConvertedCompYearly 컬럼이 필요합니다.")

    stats = language_statistics.copy().reset_index(drop=True)
    stats["language"] = stats["language"].astype(str).str.strip()
    numeric_columns = [
        "user_count", "average_salary", "median_salary", "salary_mean_median_gap",
        "salary_mean_median_gap_ratio", "high_salary_count", "high_salary_rate",
        "actual_high_salary_rate_pct", "q1_salary", "q3_salary", "iqr_salary",
        "percentile_10_salary", "percentile_90_salary", "minimum_salary",
        "maximum_salary", "salary_std", "usage_rate",
    ]
    for column in numeric_columns:
        if column in stats.columns:
            stats[column] = pd.to_numeric(stats[column], errors="coerce")
    stats["sample_warning"] = stats.apply(
        lambda row: report_sample_warning(int(row["user_count"]), int(row["high_salary_count"])),
        axis=1,
    )
    stats["salary_mean_median_gap_ratio"] = stats["salary_mean_median_gap"].div(
        stats["median_salary"].replace(0, np.nan)
    )
    stats["actual_high_salary_rate_pct"] = stats["high_salary_rate"] * 100

    salary = pd.to_numeric(model_df["ConvertedCompYearly"], errors="coerce").dropna()
    if salary.empty:
        raise AnalysisError("언어별 연봉 보고서를 생성할 유효한 연봉 데이터가 없습니다.")
    overall_average = float(salary.mean())
    overall_median = float(salary.median())
    eligible = stats.loc[stats["user_count"].ge(REPORT_SAMPLE_MIN)].copy()
    ranking_base = eligible if not eligible.empty else stats
    top_user = stats.sort_values("user_count", ascending=False).iloc[0]
    top_average = ranking_base.sort_values("average_salary", ascending=False).iloc[0]
    top_median = ranking_base.sort_values("median_salary", ascending=False).iloc[0]
    top_gap = ranking_base.sort_values("salary_mean_median_gap", ascending=False).iloc[0]
    top_high_rate = ranking_base.sort_values("high_salary_rate", ascending=False).iloc[0]
    smallest = stats.sort_values("user_count", ascending=True).iloc[0]
    high_median = ranking_base.sort_values("median_salary", ascending=False).iloc[0]

    median_rank = stats.sort_values("median_salary", ascending=False).reset_index(drop=True)
    average_rank = stats.sort_values("average_salary", ascending=False).reset_index(drop=True)
    rank_positions = {
        language: index
        for index, language in enumerate(median_rank["language"], start=1)
    }
    rank_gap_rows = []
    for index, language in enumerate(average_rank["language"], start=1):
        rank_gap_rows.append({
            "language": language,
            "rank_gap": abs(index - rank_positions.get(language, index)),
        })
    largest_rank_gap = max(rank_gap_rows, key=lambda row: row["rank_gap"])

    salary_by_language: dict[str, list[float]] = {}
    for feature in LANGUAGE_FEATURES:
        language = language_name_from_feature(feature)
        values = pd.to_numeric(
            model_df.loc[model_df[feature].astype(int).eq(1), "ConvertedCompYearly"],
            errors="coerce",
        ).dropna()
        salary_by_language[language] = [float(value) for value in values.tolist()]

    language_count_stats = create_language_count_statistics(model_df)
    country_data = create_group_language_salary_data(model_df, "Country")
    experience_data = create_experience_language_salary_data(model_df)
    survey_year = metadata.get("survey_year", "확인되지 않음")
    currency_label = metadata.get("currency_label", "원본 데이터에 통화 단위 컬럼이 없어 확인되지 않음")
    salary_column = metadata.get("salary_column", "ConvertedCompYearly")

    stats_records = stats.to_dict(orient="records")
    table_rows = []
    for _, row in stats.sort_values("median_salary", ascending=False).iterrows():
        table_rows.append(
            "<tr data-language=\"{language}\" data-user-count=\"{users}\" "
            "data-average-salary=\"{average}\" data-median-salary=\"{median}\" "
            "data-warning=\"{warning}\">"
            "<td class=\"language-cell\">{language}</td>"
            "<td data-number=\"{users}\">{users_fmt}</td>"
            "<td data-number=\"{average}\">{average_fmt}</td>"
            "<td data-number=\"{median}\">{median_fmt}</td>"
            "<td data-number=\"{gap}\">{gap_fmt}</td>"
            "<td data-number=\"{gap_ratio}\">{gap_ratio_fmt}</td>"
            "<td data-number=\"{q1}\">{q1_fmt}</td>"
            "<td data-number=\"{q3}\">{q3_fmt}</td>"
            "<td data-number=\"{iqr}\">{iqr_fmt}</td>"
            "<td data-number=\"{high_count}\">{high_count}</td>"
            "<td data-number=\"{high_rate}\">{high_rate_fmt}</td>"
            "<td><span class=\"warning {warning_class}\">{warning}</span></td></tr>".format(
                language=escape(str(row["language"])),
                users=int(row["user_count"]),
                users_fmt=f"{int(row['user_count']):,}",
                average=report_json_safe(row["average_salary"]),
                average_fmt=report_money(row["average_salary"]),
                median=report_json_safe(row["median_salary"]),
                median_fmt=report_money(row["median_salary"]),
                gap=report_json_safe(row["salary_mean_median_gap"]),
                gap_fmt=report_money(row["salary_mean_median_gap"]),
                gap_ratio=report_json_safe(row["salary_mean_median_gap_ratio"]),
                gap_ratio_fmt=report_percent(row["salary_mean_median_gap_ratio"]),
                q1=report_json_safe(row["q1_salary"]),
                q1_fmt=report_money(row["q1_salary"]),
                q3=report_json_safe(row["q3_salary"]),
                q3_fmt=report_money(row["q3_salary"]),
                iqr=report_json_safe(row["iqr_salary"]),
                iqr_fmt=report_money(row["iqr_salary"]),
                high_count=int(row["high_salary_count"]),
                high_rate=report_json_safe(row["high_salary_rate"]),
                high_rate_fmt=report_percent(row["high_salary_rate"]),
                warning=escape(str(row["sample_warning"])),
                warning_class="warning-low" if row["sample_warning"] == "표본 부족" else "warning-mid" if row["sample_warning"] == "해석 주의" else "warning-ok",
            )
        )

    model_appendix_rows = []
    if language_evaluation is not None and not language_evaluation.empty:
        for _, row in language_evaluation.sort_values("test_user_count", ascending=False).iterrows():
            model_appendix_rows.append(
                "<tr><td>{}</td><td>{:,}</td><td>{:,}</td><td>{:,}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    escape(str(row["language"])),
                    int(row["test_user_count"]),
                    int(row.get("actual_high_salary_count", 0)),
                    int(row.get("predicted_high_salary_count", 0)),
                    report_percent(row.get("actual_high_salary_rate")),
                    report_percent(row.get("predicted_high_salary_rate")),
                    "—" if pd.isna(row.get("f1")) else f"{float(row['f1']):.3f}",
                    "—" if pd.isna(row.get("roc_auc")) else f"{float(row['roc_auc']):.3f}",
                )
            )

    main_chart_count = 7 + int(bool(salary_by_language)) + 1 + int(bool(country_data)) + int(bool(experience_data))
    appendix_chart_count = 4 if language_evaluation is not None and not language_evaluation.empty else 0
    insights = [
        f"사용자가 가장 많은 언어는 {top_user['language']}이며, 분석 응답자는 {int(top_user['user_count']):,}명입니다.",
        f"표본 {REPORT_SAMPLE_MIN}명 이상 언어 중 중앙 연봉이 가장 높은 집단은 {top_median['language']}이며, 중앙 연봉은 {report_money(top_median['median_salary'])}입니다.",
        f"평균 연봉은 {top_average['language']} 사용자 집단에서 가장 높게 관찰됐습니다({report_money(top_average['average_salary'])}, 사용자 {int(top_average['user_count']):,}명).",
        f"{top_gap['language']}는 평균과 중앙 연봉의 차이가 {report_money(top_gap['salary_mean_median_gap'])}로 가장 커, 일부 높은 연봉 관측값의 영향을 받을 가능성이 있습니다.",
        f"실제 고연봉 비율은 {top_high_rate['language']} 사용자 집단에서 {report_percent(top_high_rate['high_salary_rate'])}로 가장 높게 관찰됐습니다.",
        f"가장 작은 언어 표본은 {smallest['language']}({int(smallest['user_count']):,}명)로, 결과 해석에 주의가 필요합니다.",
        f"평균 연봉 순위와 중앙 연봉 순위 차이가 가장 큰 언어는 {largest_rank_gap['language']}({largest_rank_gap['rank_gap']}개 순위 차이)입니다.",
        f"{high_median['language']}의 중앙 연봉은 상대적으로 높지만, 국가·경력·직무 구성이 함께 반영됐을 가능성을 고려해야 합니다.",
    ]

    css = """
    :root{--navy:#102a43;--blue:#1976d2;--teal:#008f95;--gold:#c28b20;--coral:#e76f51;--ink:#243b53;--muted:#627d98;--bg:#f3f6f9;--card:#fff;--line:#d9e2ec}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",Arial,sans-serif;line-height:1.6}.hero{background:linear-gradient(135deg,#102a43,#1f4e79);color:white;padding:56px max(24px,calc((100vw - 1240px)/2)) 44px}.hero h1{font-size:clamp(2rem,4vw,3.5rem);line-height:1.12;margin:12px 0}.hero h2{font-size:clamp(1.1rem,2vw,1.55rem);font-weight:500;margin:0 0 20px;color:#d9efff}.badges{display:flex;flex-wrap:wrap;gap:8px}.badge{border:1px solid rgba(255,255,255,.4);border-radius:999px;padding:4px 11px;font-size:.82rem}.meta{color:#d9e2ec;font-size:.9rem}.nav{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid var(--line);box-shadow:0 2px 8px #102a4314}.nav-inner{max-width:1240px;margin:auto;display:flex;gap:4px;overflow:auto}.nav a{white-space:nowrap;padding:12px 10px;color:var(--muted);text-decoration:none;font-size:.87rem}.nav a:hover{color:var(--blue)}main{max-width:1240px;margin:28px auto;padding:0 20px}.section{scroll-margin-top:66px;margin:34px 0}.section h2{font-size:1.55rem;color:var(--navy);margin:0 0 6px}.section-intro{color:var(--muted);margin:0 0 16px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 5px 18px #102a430b}.kpi{grid-column:span 2;min-height:128px}.kpi .label{color:var(--muted);font-size:.82rem}.kpi .value{font-size:1.55rem;font-weight:750;color:var(--navy);margin-top:8px}.kpi .sub{font-size:.8rem;color:var(--muted);margin-top:4px}.chart-card{grid-column:span 6;min-height:430px}.chart-card.wide{grid-column:1/-1}.chart{width:100%;height:380px}.insight{border-left:4px solid var(--teal);background:#e8f7f7;padding:14px 18px;border-radius:8px;margin:8px 0}.note{border-left:4px solid var(--gold);background:#fff8e7;padding:14px 18px;border-radius:8px;margin:14px 0}.warning-box{border-left:4px solid var(--coral);background:#fff0ed;padding:14px 18px;border-radius:8px}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:8px 0 14px}.control{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:.8rem}.control input,.control select,.button{border:1px solid var(--line);border-radius:8px;background:white;padding:8px 10px;color:var(--ink)}.button{cursor:pointer}.button:hover{border-color:var(--blue);color:var(--blue)}.table-wrap{overflow:auto;max-height:650px;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;background:white;min-width:1080px}th,td{border-bottom:1px solid #edf2f7;padding:9px 10px;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#eaf1f7;color:var(--navy);font-size:.82rem;z-index:2}th:first-child,td:first-child{text-align:left}.language-cell{font-weight:700;color:var(--navy)}tr:hover td{background:#f7fbff}.warning{display:inline-block;border-radius:999px;padding:2px 8px;font-size:.75rem}.warning-low{background:#ffe1dc;color:#a33b28}.warning-mid{background:#fff0c2;color:#826000}.warning-ok{background:#def7ec;color:#176b4d}.detail-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.detail-item{background:#f7fafc;border-radius:10px;padding:12px}.detail-item .label{font-size:.76rem;color:var(--muted)}.detail-item .value{font-weight:700;color:var(--navy);font-size:1.05rem}.footer{color:var(--muted);font-size:.85rem;padding:24px 0 60px}details{background:white;border:1px solid var(--line);border-radius:12px;padding:14px}summary{cursor:pointer;font-weight:700;color:var(--navy)}@media(max-width:900px){.kpi{grid-column:span 4}.chart-card{grid-column:1/-1}.detail-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.kpi{grid-column:span 6}.hero{padding-top:36px}.detail-grid{grid-template-columns:1fr 1fr}}@media print{.nav,.controls,.button{display:none!important}body{background:white}.card{box-shadow:none;break-inside:avoid}.section{break-inside:avoid}}
    """

    nav_items = [
        ("summary", "Executive Summary"), ("dataset", "Dataset Overview"),
        ("popularity", "Language Popularity"), ("ranking", "Salary Ranking"),
        ("distribution", "Salary Distribution"), ("gap", "Mean vs Median"),
        ("bubble", "Salary and Sample Size"), ("high-share", "High-Salary Share"),
        ("table", "Detailed Table"), ("method", "Methodology"), ("limits", "Limitations"),
    ]
    nav_html = "".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in nav_items)
    insight_html = "".join(f"<div class=\"insight\">{escape(item)}</div>" for item in insights)
    language_options = "".join(
        f'<option value="{escape(str(language))}">{escape(str(language))}</option>'
        for language in stats.sort_values("user_count", ascending=False)["language"]
    )

    html_doc = "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Programming Language Salary Analysis</title><style>" + css + "</style></head><body>"
    html_doc += f"""
    <header class="hero">
      <div class="badges"><span class="badge">Salary Analysis</span><span class="badge">Programming Languages</span><span class="badge">Stack Overflow Survey</span><span class="badge">Descriptive Analytics</span><span class="badge">Interactive Report</span></div>
      <h1>Programming Language Salary Analysis</h1>
      <h2>프로그래밍 언어별 개발자 연봉 분석 보고서</h2>
      <p>Stack Overflow 개발자 설문 기반 언어 사용자 집단별 연봉 수준·분포·표본 특성 비교</p>
      <p class="meta">보고서 생성: {escape(str(metadata.get('generated_at', '확인되지 않음')))} · 데이터 기준 연도: {escape(str(survey_year))} · 연봉 컬럼: {escape(str(salary_column))} · 통화 단위: {escape(str(currency_label))}</p>
    </header>
    <nav class="nav"><div class="nav-inner">{nav_html}</div></nav>
    <main>
      <section id="summary" class="section"><h2>Executive Summary</h2><p class="section-intro">분류 성능이 아니라 언어 사용자 집단의 연봉 수준과 분포를 중심으로 요약합니다.</p>
        <div class="grid">
          <div class="card kpi"><div class="label">분석 대상 응답자</div><div class="value">{len(model_df):,}</div><div class="sub">유효한 연봉·언어 정보</div></div>
          <div class="card kpi"><div class="label">분석 언어 수</div><div class="value">{len(stats):,}</div><div class="sub">15개 지정 언어</div></div>
          <div class="card kpi"><div class="label">전체 평균 연봉</div><div class="value">{report_money(overall_average)}</div><div class="sub">{escape(str(salary_column))}</div></div>
          <div class="card kpi"><div class="label">전체 중앙 연봉</div><div class="value">{report_money(overall_median)}</div><div class="sub">응답자 단위 중앙값</div></div>
          <div class="card kpi"><div class="label">중앙 연봉 최고 언어</div><div class="value">{escape(str(top_median['language']))}</div><div class="sub">표본 {int(top_median['user_count']):,}명 이상 기준</div></div>
          <div class="card kpi"><div class="label">사용자 수 최다 언어</div><div class="value">{escape(str(top_user['language']))}</div><div class="sub">{int(top_user['user_count']):,}명</div></div>
        </div>
        <div class="card" style="margin-top:18px"><h3>핵심 관찰</h3>{insight_html}</div>
      </section>
      <section id="dataset" class="section"><h2>Dataset Overview</h2><p class="section-intro">실제 데이터와 통계 기준을 확인합니다.</p>
        <div class="card"><p><b>연봉 기준:</b> {escape(str(salary_column))}을 사용했습니다. {escape(str(salary_column))}은 Stack Overflow 설문에서 연간 보상액을 공통 기준으로 변환한 컬럼이지만, 현재 CSV와 코드에는 통화 단위를 명시한 컬럼이 없어 특정 통화로 단정하지 않습니다.</p><p><b>분석 표본:</b> 전체 로딩 {int(metadata.get('total_loaded_rows', len(model_df))):,}건 중 유효 분석 {len(model_df):,}건입니다. 연봉 결측·0 이하와 언어 결측·중복 응답은 기존 전처리 기준에 따라 제외됐습니다.</p><div class="note">한 응답자는 여러 프로그래밍 언어를 선택할 수 있습니다. 따라서 언어별 사용자 수의 합은 전체 응답자 수와 일치하지 않으며, 언어 사용자 집단은 서로 완전히 분리되지 않습니다.</div></div>
      </section>
      <section id="popularity" class="section"><h2>Language Popularity</h2><p class="section-intro">프로그래밍 언어별 응답자 수입니다. 한 응답자의 복수 선택이 허용됩니다.</p>
        <div class="card"><div class="controls"><label class="control">표시 범위<select id="popularity-limit"><option value="all">전체</option><option value="5">Top 5</option><option value="10">Top 10</option><option value="15">Top 15</option></select></label></div><div id="chart-popularity" class="chart"></div></div>
      </section>
      <section id="ranking" class="section"><h2>Salary Ranking</h2><p class="section-intro">평균만으로 순위를 정하지 않고 중앙 연봉과 함께 비교합니다.</p>
        <div class="card"><div class="controls"><label class="control">정렬 기준<select id="salary-sort"><option value="median_salary">중앙 연봉</option><option value="average_salary">평균 연봉</option><option value="user_count">사용자 수</option><option value="salary_mean_median_gap">평균-중앙값 차이</option><option value="high_salary_rate">실제 고연봉 비율</option></select></label></div><div id="chart-salary-ranking" class="chart"></div></div>
      </section>
      <section id="distribution" class="section"><h2>Salary Distribution</h2><p class="section-intro">원본 응답자별 연봉으로 언어별 분포와 사분위 범위를 확인합니다.</p>
        <div class="card"><div class="controls"><label class="control">언어<select id="box-language"><option value="top10">사용자 수 Top 10</option><option value="all">전체 언어</option>{language_options}</select></label><button id="box-log-toggle" class="button">로그 축 전환</button></div><div id="chart-box" class="chart"></div></div>
        <div class="card"><h3>Q1–Median–Q3 연봉 범위</h3><div id="chart-interval" class="chart"></div></div>
      </section>
      <section id="gap" class="section"><h2>Mean vs Median</h2><p class="section-intro">평균과 중앙 연봉의 차이가 클수록 일부 높은 연봉 관측값의 영향을 받을 가능성이 있습니다. 단, 이 차이만으로 왜도나 분포 형태를 확정하지는 않습니다.</p><div class="card"><div id="chart-gap" class="chart"></div></div></section>
      <section id="bubble" class="section"><h2>Salary and Sample Size</h2><p class="section-intro">사용자 수와 중앙 연봉을 함께 보며 표본이 크고 연봉 중심 수준이 높은 집단을 구분합니다.</p><div class="card"><div id="chart-bubble" class="chart"></div></div></section>
      <section id="high-share" class="section"><h2>High-Salary Share</h2><p class="section-intro">분류 모델이 아니라 관측된 연봉 기준으로 계산한 언어 사용자 집단별 실제 고연봉 비율입니다.</p><div class="card"><div id="chart-high-rate" class="chart"></div></div><div class="card"><div id="chart-high-scatter" class="chart"></div><div class="note">고연봉 비율은 언어 자체의 효과가 아니라 국가, 경력, 직무, 회사 규모 등 다양한 요인이 함께 반영된 결과입니다.</div></div></section>
      <section id="language-count" class="section"><h2>LanguageCount별 연봉</h2><p class="section-intro">여러 언어를 사용하는 응답자의 연봉 차이를 보여주는 보조 분석입니다. 관찰된 차이를 언어 개수의 직접 효과로 해석할 수 없습니다.</p><div class="card"><div id="chart-language-count" class="chart"></div></div></section>
      <section id="context" class="section"><h2>국가·경력 보조 분석</h2><p class="section-intro">언어별 연봉 비교에 국가와 경력 구성이 미치는 영향을 확인하기 위한 보조 차트입니다.</p><div class="grid"><div class="card chart-card"><h3>주요 국가별 언어 중앙 연봉</h3><label class="control">국가<select id="country-select"></select></label><div id="chart-country" class="chart"></div></div><div class="card chart-card"><h3>경력 구간별 언어 중앙 연봉</h3><label class="control">경력 구간<select id="experience-select"></select></label><div id="chart-experience" class="chart"></div></div></div></section>
      <section id="table" class="section"><h2>Detailed Table</h2><p class="section-intro">언어별 연봉 통계와 표본 경고를 검색·필터링할 수 있습니다.</p>
        <div class="card"><div class="controls"><label class="control">검색<input id="table-search" placeholder="언어 검색"></label><label class="control">최소 사용자 수<input id="min-users" type="number" min="0" value="0"></label><label class="control">최소 평균 연봉<input id="min-average" type="number" min="0" value="0"></label><label class="control">최소 중앙 연봉<input id="min-median" type="number" min="0" value="0"></label><label class="control"><span>표본 부족 숨기기</span><input id="hide-small" type="checkbox"></label><button id="download-csv" class="button">CSV 다운로드</button><button onclick="window.print()" class="button">인쇄 / PDF</button></div><div class="table-wrap"><table id="salary-table"><thead><tr><th>언어</th><th>사용자 수</th><th>평균 연봉</th><th>중앙 연봉</th><th>평균-중앙값 차이</th><th>차이 비율</th><th>Q1</th><th>Q3</th><th>IQR</th><th>실제 고연봉자 수</th><th>실제 고연봉 비율</th><th>표본 상태</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></div>
      </section>
      <section id="detail" class="section"><h2>언어 상세 선택</h2><p class="section-intro">선택한 언어의 규모·중심 연봉·분포·표본 상태를 확인합니다.</p><div class="card"><label class="control">언어<select id="detail-language">{language_options}</select></label><div class="detail-grid" style="margin-top:16px"><div class="detail-item"><div class="label">사용자 수</div><div id="detail-users" class="value">—</div></div><div class="detail-item"><div class="label">평균 연봉</div><div id="detail-average" class="value">—</div></div><div class="detail-item"><div class="label">중앙 연봉</div><div id="detail-median" class="value">—</div></div><div class="detail-item"><div class="label">표본 상태</div><div id="detail-warning" class="value">—</div></div><div class="detail-item"><div class="label">평균-중앙값 차이</div><div id="detail-gap" class="value">—</div></div><div class="detail-item"><div class="label">Q1–Q3</div><div id="detail-iqr" class="value">—</div></div><div class="detail-item"><div class="label">실제 고연봉자 수</div><div id="detail-high-count" class="value">—</div></div><div class="detail-item"><div class="label">실제 고연봉 비율</div><div id="detail-high-rate" class="value">—</div></div></div><p id="detail-interpretation" class="note" style="margin-top:16px"></p></div></section>
      <section id="method" class="section"><h2>Methodology</h2><div class="card"><ol><li>Stack Overflow 설문에서 <code>{escape(str(salary_column))}</code>을 숫자로 정리하고 결측·0 이하 연봉과 언어 결측·중복 응답을 기존 정제 기준으로 제외했습니다.</li><li><code>LanguageHaveWorkedWith</code>와 15개 언어 플래그를 사용해 언어별 0/1 특성을 만들었습니다.</li><li>언어별 사용자 집단마다 평균, 중앙값, P10, Q1, Q3, P90, IQR, 표준편차, 최솟값·최댓값을 계산했습니다.</li><li>고연봉 비율은 학습 데이터의 75% 분위수 이상 여부를 사용한 보조 지표입니다. 분류 모델은 이 보고서의 부록에서만 다룹니다.</li><li>한 사람이 여러 언어의 통계에 포함될 수 있으므로 언어별 집단은 독립표본이 아닙니다.</li></ol><p><b>연봉 단위:</b> {escape(str(currency_label))}</p></div></section>
      <section id="limits" class="section"><h2>Limitations</h2><div class="warning-box"><ul><li>설문 응답자가 전체 개발자 모집단을 대표하지 않을 수 있습니다.</li><li>연봉과 사용 언어는 자기보고 데이터입니다.</li><li>국가별 임금·환율·생활비 차이가 언어별 결과에 함께 반영될 수 있습니다.</li><li>언어 사용자 집단별 경력과 직무 구성이 다를 수 있습니다.</li><li>평균은 일부 극단적인 고연봉 응답의 영향을 받을 수 있습니다.</li><li>관찰된 연봉 차이는 인과관계를 의미하지 않습니다.</li><li>채용, 연봉 책정, 개인 평가에 직접 사용해서는 안 됩니다.</li></ul></div></section>
      <section class="section"><details><summary>부록: 언어 사용자 그룹별 분류 모델 진단</summary><p>아래 결과는 언어별 독립 모델이 아닙니다. 전체 Full 모델의 테스트 결과를 각 언어 사용자 하위집단으로 나눈 보조 진단이며, 실제 연봉 금액을 직접 예측한 결과가 아닙니다.</p><div class="table-wrap"><table><thead><tr><th>언어</th><th>테스트 수</th><th>실제 고연봉자 수</th><th>예측 고연봉자 수</th><th>실제 비율</th><th>예측 비율</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>{''.join(model_appendix_rows) if model_appendix_rows else '<tr><td colspan="8">분류 보조 결과가 없습니다.</td></tr>'}</tbody></table></div><div class="grid"><div class="card chart-card"><div id="chart-model-count" class="chart"></div></div><div class="card chart-card"><div id="chart-model-rate" class="chart"></div></div><div class="card chart-card"><div id="chart-model-f1" class="chart"></div></div><div class="card chart-card"><div id="chart-model-roc" class="chart"></div></div></div></details></section>
      <footer class="footer">핵심 분석 주제: 프로그래밍 언어 사용자 집단별 연봉 수준 및 분포 비교 · 메인 차트 {main_chart_count}개 · 부록 차트 {appendix_chart_count}개 · 보고서 크기는 생성 후 출력됩니다.</footer>
    </main>
    """

    plotly_js = get_plotlyjs()
    html_doc += "<script>" + plotly_js + "</script>"
    html_doc += "<script>\n"
    html_doc += "const salaryStats = " + report_json(stats_records) + ";\n"
    html_doc += "const salaryValues = " + report_json(salary_by_language) + ";\n"
    html_doc += "const languageCountStats = " + report_json(language_count_stats.to_dict(orient="records")) + ";\n"
    html_doc += "const countryData = " + report_json(country_data) + ";\n"
    html_doc += "const experienceData = " + report_json(experience_data) + ";\n"
    html_doc += "const modelEvaluation = " + report_json(language_evaluation.to_dict(orient="records") if language_evaluation is not None else []) + ";\n"
    html_doc += r'''
const salaryUnit = "ConvertedCompYearly · 통화 단위 미확인";
const plotConfig = {responsive:true, displaylogo:false, modeBarButtonsToRemove:["lasso2d","select2d"]};
const money = value => value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : new Intl.NumberFormat("ko-KR", {maximumFractionDigits:0}).format(Number(value));
const pct = value => value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : (Number(value)*100).toFixed(1)+"%";
const layout = (title, xTitle="", yTitle="") => ({title:{text:title,font:{size:17,color:"#102a43"}},paper_bgcolor:"#ffffff",plot_bgcolor:"#ffffff",font:{family:"Arial, Apple SD Gothic Neo, sans-serif",color:"#243b53"},margin:{l:80,r:32,t:58,b:72},xaxis:{title:xTitle,gridcolor:"#e6edf3"},yaxis:{title:yTitle,gridcolor:"#e6edf3"},hoverlabel:{bgcolor:"#102a43",font:{color:"white"}}});
function renderPopularity(limit="all") {
  let rows = [...salaryStats].sort((a,b)=>b.user_count-a.user_count);
  if(limit !== "all") rows = rows.slice(0, Number(limit));
  rows.reverse();
  Plotly.react("chart-popularity", [{type:"bar",orientation:"h",x:rows.map(r=>r.user_count),y:rows.map(r=>r.language),customdata:rows.map(r=>[r.usage_rate,r.average_salary,r.median_salary,r.high_salary_rate]),marker:{color:"#1976d2"},text:rows.map(r=>money(r.user_count)),textposition:"outside",hovertemplate:"%{y}<br>사용자 수: %{x:,}<br>전체 대비: %{customdata[0]:.1%}<br>평균 연봉: %{customdata[1]:,.0f}<br>중앙 연봉: %{customdata[2]:,.0f}<br>실제 고연봉 비율: %{customdata[3]:.1%}<extra></extra>"}], {...layout("프로그래밍 언어별 응답자 수","사용자 수","프로그래밍 언어"),height:Math.max(380,rows.length*34),showlegend:false}, plotConfig);
}
function renderSalaryRanking(sortKey="median_salary") {
  const rows = [...salaryStats].sort((a,b)=>(Number(b[sortKey]??-Infinity)-Number(a[sortKey]??-Infinity))).reverse();
  Plotly.react("chart-salary-ranking", [{type:"bar",orientation:"h",name:"평균 연봉",x:rows.map(r=>r.average_salary),y:rows.map(r=>r.language),customdata:rows.map(r=>[r.user_count,r.median_salary,r.salary_mean_median_gap,r.high_salary_rate]),marker:{color:"#c28b20"},hovertemplate:"%{y}<br>평균 연봉: %{x:,.0f}<br>사용자 수: %{customdata[0]:,}<br>중앙 연봉: %{customdata[1]:,.0f}<br>평균-중앙값 차이: %{customdata[2]:,.0f}<br>실제 고연봉 비율: %{customdata[3]:.1%}<extra></extra>"},{type:"bar",orientation:"h",name:"중앙 연봉",x:rows.map(r=>r.median_salary),y:rows.map(r=>r.language),marker:{color:"#008f95"},hovertemplate:"%{y}<br>중앙 연봉: %{x:,.0f}<extra></extra>"}], {...layout("프로그래밍 언어별 평균 및 중앙 연봉",salaryUnit,"프로그래밍 언어"),barmode:"group",height:Math.max(420,rows.length*34),showlegend:true},plotConfig);
}
function renderBox(filter="top10") {
  const top = [...salaryStats].sort((a,b)=>b.user_count-a.user_count).slice(0,10).map(r=>r.language);
  const names = filter === "all" ? Object.keys(salaryValues) : filter === "top10" ? top : [filter];
  const traces = Object.entries(salaryValues).map(([language,values])=>({type:"box",name:language,y:values,visible:names.includes(language),boxpoints:"outliers",marker:{color:"#1976d2",size:3},line:{color:"#1976d2"},hovertemplate:language+"<br>연봉: %{y:,.0f}<extra></extra>"}));
  Plotly.react("chart-box",traces,{...layout("언어별 연봉 분포",salaryUnit,"프로그래밍 언어"),yaxis:{title:salaryUnit,type:document.body.dataset.salaryAxis||"linear",gridcolor:"#e6edf3"},showlegend:false,height:520},plotConfig);
}
function renderGroupChart(elementId,data,selected,title) {
  const rows = data[selected] || [];
  Plotly.react(elementId,[{type:"bar",x:rows.map(r=>r.language),y:rows.map(r=>r.median_salary),customdata:rows.map(r=>[r.user_count,r.average_salary]),marker:{color:"#008f95"},hovertemplate:"%{x}<br>중앙 연봉: %{y:,.0f}<br>사용자 수: %{customdata[0]:,}<br>평균 연봉: %{customdata[1]:,.0f}<extra></extra>"}],{...layout(title,salaryUnit,"언어"),showlegend:false},plotConfig);
}
function renderAppendix() {
  if(!modelEvaluation.length) return;
  const rows = modelEvaluation;
  Plotly.newPlot("chart-model-count",[{type:"bar",name:"실제",x:rows.map(r=>r.language),y:rows.map(r=>r.actual_high_salary_count),marker:{color:"#008f95"}},{type:"bar",name:"예측",x:rows.map(r=>r.language),y:rows.map(r=>r.predicted_high_salary_count),marker:{color:"#c28b20"}}],{...layout("실제 vs 예측 고연봉자 수","언어","인원"),barmode:"group"},plotConfig);
  Plotly.newPlot("chart-model-rate",[{type:"bar",name:"실제 비율",x:rows.map(r=>r.language),y:rows.map(r=>r.actual_high_salary_rate),marker:{color:"#008f95"}},{type:"bar",name:"예측 비율",x:rows.map(r=>r.language),y:rows.map(r=>r.predicted_high_salary_rate),marker:{color:"#c28b20"}}],{...layout("실제 vs 예측 고연봉 비율","언어","비율"),barmode:"group",yaxis:{tickformat:".0%"}},plotConfig);
  Plotly.newPlot("chart-model-f1",[{type:"bar",x:rows.map(r=>r.language),y:rows.map(r=>r.f1),marker:{color:"#1976d2"}}],{...layout("언어 사용자 그룹별 F1","언어","F1")},plotConfig);
  Plotly.newPlot("chart-model-roc",[{type:"bar",x:rows.map(r=>r.language),y:rows.map(r=>r.roc_auc),marker:{color:"#7b61ff"}}],{...layout("언어 사용자 그룹별 ROC-AUC","언어","ROC-AUC")},plotConfig);
}
function updateDetail(language) {
  const row = salaryStats.find(r=>r.language===language) || salaryStats[0];
  document.getElementById("detail-users").textContent = money(row.user_count);
  document.getElementById("detail-average").textContent = money(row.average_salary);
  document.getElementById("detail-median").textContent = money(row.median_salary);
  document.getElementById("detail-warning").textContent = row.sample_warning;
  document.getElementById("detail-gap").textContent = money(row.salary_mean_median_gap);
  document.getElementById("detail-iqr").textContent = money(row.q1_salary)+" – "+money(row.q3_salary);
  document.getElementById("detail-high-count").textContent = money(row.high_salary_count);
  document.getElementById("detail-high-rate").textContent = pct(row.high_salary_rate);
  const message = row.salary_mean_median_gap > 0 ? "이 언어 사용자 집단은 평균 연봉이 중앙 연봉보다 크게 나타났습니다. 일부 높은 연봉 응답값의 영향을 받을 가능성을 고려해야 합니다." : "이 언어 사용자 집단은 평균과 중앙 연봉의 차이가 상대적으로 작게 나타났습니다.";
  document.getElementById("detail-interpretation").textContent = message + " 표본 상태: "+row.sample_warning+".";
}
function filterTable() {
  const query = document.getElementById("table-search").value.toLowerCase();
  const minUsers = Number(document.getElementById("min-users").value||0), minAverage = Number(document.getElementById("min-average").value||0), minMedian = Number(document.getElementById("min-median").value||0), hideSmall = document.getElementById("hide-small").checked;
  document.querySelectorAll("#salary-table tbody tr").forEach(row=>{const ok=row.dataset.language.toLowerCase().includes(query)&&Number(row.dataset.userCount)>=minUsers&&Number(row.dataset.averageSalary||0)>=minAverage&&Number(row.dataset.medianSalary||0)>=minMedian&&(!hideSmall||row.dataset.warning!=="표본 부족");row.style.display=ok?"":"none";});
}
function downloadTableCsv() {
  const headers=["Language","User Count","Average Salary","Median Salary","Mean-Median Gap","Gap Ratio","Q1","Q3","IQR","Actual High Salary Count","Actual High Salary Rate","Sample Warning"];
  const rows=salaryStats.map(r=>[r.language,r.user_count,r.average_salary,r.median_salary,r.salary_mean_median_gap,r.salary_mean_median_gap_ratio,r.q1_salary,r.q3_salary,r.iqr_salary,r.high_salary_count,r.high_salary_rate,r.sample_warning]);
  const csv=[headers,...rows].map(row=>row.map(value=>`"${String(value??"").replaceAll('"','""')}"`).join(",")).join("\n");
  const blob=new Blob(["\ufeff"+csv],{type:"text/csv;charset=utf-8"}), url=URL.createObjectURL(blob), a=document.createElement("a");a.href=url;a.download="language_salary_statistics.csv";a.click();URL.revokeObjectURL(url);
}
document.getElementById("popularity-limit").addEventListener("change",e=>renderPopularity(e.target.value));
document.getElementById("salary-sort").addEventListener("change",e=>renderSalaryRanking(e.target.value));
document.getElementById("box-language").addEventListener("change",e=>renderBox(e.target.value));
document.getElementById("box-log-toggle").addEventListener("click",()=>{document.body.dataset.salaryAxis=document.body.dataset.salaryAxis==="log"?"linear":"log";renderBox(document.getElementById("box-language").value);});
document.getElementById("detail-language").addEventListener("change",e=>updateDetail(e.target.value));
["table-search","min-users","min-average","min-median","hide-small"].forEach(id=>document.getElementById(id).addEventListener("input",filterTable));
document.getElementById("download-csv").addEventListener("click",downloadTableCsv);
document.getElementById("country-select").innerHTML=Object.keys(countryData).map(key=>`<option>${key}</option>`).join("");
document.getElementById("experience-select").innerHTML=Object.keys(experienceData).map(key=>`<option>${key}</option>`).join("");
document.getElementById("country-select").addEventListener("change",e=>renderGroupChart("chart-country",countryData,e.target.value,"국가별 언어 중앙 연봉"));
document.getElementById("experience-select").addEventListener("change",e=>renderGroupChart("chart-experience",experienceData,e.target.value,"경력 구간별 언어 중앙 연봉"));
renderPopularity();renderSalaryRanking();renderBox();
Plotly.newPlot("chart-gap",[{type:"bar",orientation:"h",x:salaryStats.slice().sort((a,b)=>a.salary_mean_median_gap-b.salary_mean_median_gap).map(r=>r.salary_mean_median_gap),y:salaryStats.slice().sort((a,b)=>a.salary_mean_median_gap-b.salary_mean_median_gap).map(r=>r.language),marker:{color:salaryStats.slice().sort((a,b)=>a.salary_mean_median_gap-b.salary_mean_median_gap).map(r=>r.salary_mean_median_gap>=0?"#c28b20":"#1976d2")},hovertemplate:"%{y}<br>평균-중앙값 차이: %{x:,.0f}<extra></extra>"}],{...layout("언어별 평균 연봉과 중앙 연봉의 차이",salaryUnit,"언어")},plotConfig);
Plotly.newPlot("chart-bubble",[{type:"scatter",mode:"markers+text",text:salaryStats.map(r=>r.language),textposition:"top center",x:salaryStats.map(r=>r.user_count),y:salaryStats.map(r=>r.median_salary),customdata:salaryStats.map(r=>[r.average_salary,r.high_salary_count,r.high_salary_rate,r.salary_mean_median_gap]),marker:{size:salaryStats.map(r=>Math.max(12,Math.sqrt(Math.max(r.average_salary,1))/8)),color:salaryStats.map(r=>r.high_salary_rate),colorscale:"Tealgrn",showscale:true,colorbar:{title:"실제 고연봉 비율"}},hovertemplate:"%{text}<br>사용자 수: %{x:,}<br>중앙 연봉: %{y:,.0f}<br>평균 연봉: %{customdata[0]:,.0f}<br>실제 고연봉자 수: %{customdata[1]:,}<br>실제 고연봉 비율: %{customdata[2]:.1%}<br>평균-중앙값 차이: %{customdata[3]:,.0f}<extra></extra>"}],{...layout("사용자 수와 중앙 연봉", "사용자 수", "중앙 연봉"),xaxis:{type:"linear",title:"사용자 수",gridcolor:"#e6edf3"}},plotConfig);
Plotly.newPlot("chart-high-rate",[{type:"bar",orientation:"h",x:salaryStats.slice().sort((a,b)=>a.high_salary_rate-b.high_salary_rate).map(r=>r.high_salary_rate),y:salaryStats.slice().sort((a,b)=>a.high_salary_rate-b.high_salary_rate).map(r=>r.language),customdata:salaryStats.slice().sort((a,b)=>a.high_salary_rate-b.high_salary_rate).map(r=>[r.user_count,r.high_salary_count,r.average_salary,r.median_salary]),marker:{color:"#e76f51"},hovertemplate:"%{y}<br>실제 고연봉 비율: %{x:.1%}<br>사용자 수: %{customdata[0]:,}<br>실제 고연봉자 수: %{customdata[1]:,}<br>평균 연봉: %{customdata[2]:,.0f}<br>중앙 연봉: %{customdata[3]:,.0f}<extra></extra>"}],{...layout("언어 사용자 집단별 실제 고연봉 비율","비율","언어"),xaxis:{title:"실제 고연봉 비율",tickformat:".0%",gridcolor:"#e6edf3"}},plotConfig);
Plotly.newPlot("chart-high-scatter",[{type:"scatter",mode:"markers+text",text:salaryStats.map(r=>r.language),textposition:"top center",x:salaryStats.map(r=>r.median_salary),y:salaryStats.map(r=>r.high_salary_rate),customdata:salaryStats.map(r=>[r.user_count,r.average_salary]),marker:{size:salaryStats.map(r=>Math.max(10,Math.sqrt(r.user_count)*2)),color:"#1976d2"},hovertemplate:"%{text}<br>중앙 연봉: %{x:,.0f}<br>실제 고연봉 비율: %{y:.1%}<br>사용자 수: %{customdata[0]:,}<br>평균 연봉: %{customdata[1]:,.0f}<extra></extra>"}],{...layout("중앙 연봉과 실제 고연봉 비율",salaryUnit,"실제 고연봉 비율"),yaxis:{title:"실제 고연봉 비율",tickformat:".0%",gridcolor:"#e6edf3"}},plotConfig);
if(languageCountStats.length){Plotly.newPlot("chart-language-count",[{type:"bar",x:languageCountStats.map(r=>r.language_count),y:languageCountStats.map(r=>r.user_count),name:"사용자 수",marker:{color:"#1976d2"},yaxis:"y2"},{type:"scatter",x:languageCountStats.map(r=>r.language_count),y:languageCountStats.map(r=>r.median_salary),name:"중앙 연봉",mode:"lines+markers",marker:{color:"#c28b20"},line:{color:"#c28b20"}}],{...layout("사용 언어 개수별 사용자 수와 중앙 연봉","사용 언어 개수","중앙 연봉"),yaxis:{title:"중앙 연봉",gridcolor:"#e6edf3"},yaxis2:{title:"사용자 수",overlaying:"y",side:"right",showgrid:false},legend:{orientation:"h"}},plotConfig);}else{document.getElementById("chart-language-count").innerHTML="LanguageCount 데이터가 없어 차트를 생성하지 않았습니다.";}
const intervalRows=[...salaryStats].sort((a,b)=>b.median_salary-a.median_salary);Plotly.newPlot("chart-interval",intervalRows.map(r=>({type:"scatter",mode:"lines",x:[r.q1_salary,r.q3_salary],y:[r.language,r.language],line:{color:"#008f95",width:12},customdata:[[r.user_count,r.median_salary]],hovertemplate:r.language+"<br>Q1: %{x:,.0f}<br>Q3: "+money(r.q3_salary)+"<br>중앙값: "+money(r.median_salary)+"<br>사용자 수: %{customdata[0]:,}<extra></extra>"})).concat([{type:"scatter",mode:"markers",x:intervalRows.map(r=>r.median_salary),y:intervalRows.map(r=>r.language),marker:{color:"#c28b20",size:10},customdata:intervalRows.map(r=>[r.q1_salary,r.q3_salary,r.user_count]),hovertemplate:"중앙값: %{x:,.0f}<br>Q1: %{customdata[0]:,.0f}<br>Q3: %{customdata[1]:,.0f}<br>사용자 수: %{customdata[2]:,}<extra></extra>"}]),{...layout("언어별 Q1–Median–Q3 연봉 범위",salaryUnit,"언어"),showlegend:false},plotConfig);
if(Object.keys(countryData).length){const first=Object.keys(countryData)[0];renderGroupChart("chart-country",countryData,first,"국가별 언어 중앙 연봉");}else{document.getElementById("chart-country").innerHTML="Country 데이터가 없어 차트를 생성하지 않았습니다.";}
if(Object.keys(experienceData).length){const first=Object.keys(experienceData)[0];renderGroupChart("chart-experience",experienceData,first,"경력 구간별 언어 중앙 연봉");}else{document.getElementById("chart-experience").innerHTML="YearsCodePro 데이터가 없어 차트를 생성하지 않았습니다.";}
updateDetail(document.getElementById("detail-language").value);renderAppendix();
'''
    html_doc += "</script></body></html>"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    if not output_path.exists() or output_path.stat().st_size < 100_000:
        raise AnalysisError(f"언어별 연봉 HTML 보고서 검증에 실패했습니다: {output_path}")
    return output_path


def create_prediction_samples(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_df: pd.DataFrame) -> pd.DataFrame:
    """예측 샘플 CSV를 생성한다.

    입력:
        model: full_model Pipeline.
        X_test: 테스트 입력.
        y_test: 테스트 타깃.
        model_df: 원본 모델 DataFrame.

    반환:
        예측 샘플 DataFrame.

    발생 가능한 예외:
        AnalysisError: 예측 실패 시.
    """
    try:
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
    except Exception as exc:  # pragma: no cover - defensive path
        raise AnalysisError(f"예측 샘플 생성 중 예측 실패: {exc}") from exc
    sample_df = pd.DataFrame({
        "ResponseId": model_df.loc[X_test.index, "ResponseId"].reset_index(drop=True),
        "ConvertedCompYearly": model_df.loc[X_test.index, "ConvertedCompYearly"].reset_index(drop=True),
        "ActualTarget": y_test.reset_index(drop=True),
        "PredictedTarget": preds.astype(int),
        "PredictedProbability": probs,
    })
    sample_df["Correct"] = sample_df["ActualTarget"] == sample_df["PredictedTarget"]
    sample_df.head(100).to_csv(DOMAIN_ROOT / "prediction_samples.csv", index=False)
    return sample_df.head(100)


def save_model(full_pipeline: Pipeline) -> Path:
    """full_model Pipeline을 저장한다.

    입력:
        full_pipeline: 학습된 full_model Pipeline.

    반환:
        저장 경로.

    발생 가능한 예외:
        AnalysisError: 저장 실패 시.
    """
    model_path = DOMAIN_ROOT / "model.joblib"
    try:
        joblib.dump(full_pipeline, model_path)
    except Exception as exc:  # pragma: no cover - defensive path
        raise AnalysisError(f"모델 저장에 실패했습니다: {exc}") from exc
    return model_path


def verify_saved_model(full_pipeline: Pipeline, model_path: Path, X_test: pd.DataFrame) -> bool:
    """저장된 모델을 재로드해 예측이 동일한지 검증한다.

    입력:
        full_pipeline: 저장 전 메모리상의 학습된 Pipeline.
        model_path: 저장된 모델 경로.
        X_test: 테스트 입력.

    반환:
        재로드 검증 성공 여부.

    발생 가능한 예외:
        AnalysisError: 재로드 또는 비교 실패 시.
    """
    try:
        reloaded_model = joblib.load(model_path)
        original_pred = full_pipeline.predict(X_test.head(20))
        reloaded_pred = reloaded_model.predict(X_test.head(20))
        original_prob = full_pipeline.predict_proba(X_test.head(20))[:, 1]
        reloaded_prob = reloaded_model.predict_proba(X_test.head(20))[:, 1]
    except Exception as exc:  # pragma: no cover - defensive path
        raise AnalysisError(f"모델 재로드 검증에 실패했습니다: {exc}") from exc
    if not np.array_equal(original_pred, reloaded_pred):
        raise AnalysisError("저장 후 다시 불러온 모델의 예측값이 일치하지 않습니다.")
    if not np.allclose(original_prob, reloaded_prob):
        raise AnalysisError("저장 후 다시 불러온 모델의 확률값이 일치하지 않습니다.")
    return True


def create_model_metrics(result_payload: dict[str, Any], stats: dict[str, int], train_rows: int, test_rows: int, high_salary_threshold: float, train_target_dist: dict[str, int], test_target_dist: dict[str, int], model_results: dict[str, Any], performance_differences: dict[str, float]) -> dict[str, Any]:
    """모델 평가 결과를 JSON 저장용 딕셔너리로 변환한다.

    입력:
        result_payload: 결과 딕셔너리.
        stats: 제외 통계.
        train_rows: 학습 행 수.
        test_rows: 테스트 행 수.
        high_salary_threshold: 고연봉 임계값.
        train_target_dist: 학습 클래스 분포.
        test_target_dist: 테스트 클래스 분포.
        model_results: 모델 지표 딕셔너리.
        performance_differences: 성능 차이 딕셔너리.

    반환:
        JSON 직렬화 가능한 메트릭 딕셔너리.

    발생 가능한 예외:
        None.
    """
    return {
        "target_name": "HighSalaryTop25",
        "target_definition": "학습 데이터의 연봉 75% 분위수 이상 여부",
        "high_salary_quantile": HIGH_SALARY_QUANTILE,
        "salary_threshold": high_salary_threshold,
        "total_loaded_rows": stats["total_loaded_rows"],
        "valid_model_rows": stats["valid_model_rows"],
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_class_distribution": {str(k): int(v) for k, v in train_target_dist.items()},
        "test_class_distribution": {str(k): int(v) for k, v in test_target_dist.items()},
        "dummy_classifier": model_results["dummy"],
        "language_only": model_results["language_only"],
        "profile_only": model_results["profile_only"],
        "full_model": model_results["full_model"],
        "performance_differences": performance_differences,
    }


def create_model_metadata(input_path: Path, file_size: int, stats: dict[str, int], train_rows: int, test_rows: int, high_salary_threshold: float, categorical_features: list[str], country_included: bool, model_results: dict[str, Any], model_path: Path, reload_verified: bool) -> dict[str, Any]:
    """모델 메타데이터를 생성한다.

    입력:
        input_path: 입력 파일 경로.
        file_size: 파일 크기.
        stats: 제외 통계.
        train_rows: 학습 행 수.
        test_rows: 테스트 행 수.
        high_salary_threshold: 고연봉 임계값.
        categorical_features: 범주형 특성 목록.
        country_included: Country 포함 여부.
        model_results: 모델 평가 결과.
        model_path: 저장된 모델 경로.
        reload_verified: 재로드 검증 결과.

    반환:
        메타데이터 딕셔너리.

    발생 가능한 예외:
        None.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": __import__("sys").version,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "joblib_version": joblib.__version__,
        "input_data_path": str(input_path),
        "input_file_size": file_size,
        "total_loaded_rows": stats["total_loaded_rows"],
        "valid_model_rows": stats["valid_model_rows"],
        "salary_missing_excluded": stats["salary_missing_excluded"],
        "salary_nonpositive_excluded": stats["salary_nonpositive_excluded"],
        "language_missing_excluded": stats["language_missing_excluded"],
        "duplicate_excluded": stats["duplicate_excluded"],
        "train_rows": train_rows,
        "test_rows": test_rows,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "high_salary_quantile": HIGH_SALARY_QUANTILE,
        "high_salary_threshold": high_salary_threshold,
        "target_name": "HighSalaryTop25",
        "target_definition": "학습 데이터의 ConvertedCompYearly 75% 분위수 이상",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": categorical_features,
        "language_features": LANGUAGE_FEATURES,
        "country_included": country_included,
        "leakage_columns": LEAKAGE_COLUMNS,
        "model_comparison": list(model_results.keys()),
        "final_model": "full_model",
        "model_hyperparameters": {
            "max_iter": 3000,
            "solver": "liblinear",
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "save_model": SAVE_MODEL,
        "model_path": str(model_path),
        "model_reload_verified": reload_verified,
    }


def create_result_json(result_payload: dict[str, Any], output_files: list[str]) -> dict[str, Any]:
    """결과 통합용 JSON을 생성한다.

    입력:
        result_payload: 실행 결과 딕셔너리.
        output_files: 생성된 파일 목록.

    반환:
        JSON 직렬화 가능한 딕셔너리.

    발생 가능한 예외:
        None.
    """
    return {
        "domain_id": 6,
        "domain_name": "integrated_top25_salary_prediction",
        "analysis_question": "언어 사용 여부와 개발자 특성으로 연봉 상위 25% 여부를 예측할 수 있는가?",
        "input_data": {
            "path": str(result_payload["input_path"]),
            "loaded_rows": result_payload["stats"]["total_loaded_rows"],
            "valid_rows": result_payload["stats"]["valid_model_rows"],
        },
        "target": {
            "name": "HighSalaryTop25",
            "definition": "학습 데이터의 ConvertedCompYearly 75% 분위수 이상",
            "quantile": HIGH_SALARY_QUANTILE,
            "salary_threshold": result_payload["high_salary_threshold"],
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": result_payload["categorical_features"],
            "languages": LANGUAGE_FEATURES,
            "country_included": result_payload["country_included"],
            "excluded_leakage_columns": LEAKAGE_COLUMNS,
        },
        "sample_sizes": {
            "total": result_payload["stats"]["valid_model_rows"],
            "train": result_payload["train_rows"],
            "test": result_payload["test_rows"],
        },
        "class_distribution": {
            "train": result_payload["train_target_distribution"],
            "test": result_payload["test_target_distribution"],
        },
        "model_comparison": result_payload["model_results"],
        "final_model_metrics": result_payload["model_results"]["full_model"],
        "top_positive_features": result_payload["top_positive_features"],
        "top_negative_features": result_payload["top_negative_features"],
        "positive_language_features": result_payload["positive_language_features"],
        "negative_language_features": result_payload["negative_language_features"],
        "language_analysis": {
            "statistics": result_payload["language_statistics"],
            "evaluation": result_payload["language_evaluation"],
            "prediction_summary": result_payload["language_prediction_summary"],
        },
        "model_saved": SAVE_MODEL,
        "model_path": str(DOMAIN_ROOT / "model.joblib"),
        "model_reload_verified": result_payload["model_reload_verified"],
        "output_files": output_files,
        "key_findings": result_payload["key_findings"],
        "limitations": result_payload["limitations"],
    }


def create_readme_content(result_payload: dict[str, Any]) -> str:
    """README를 실행 결과와 산출물 기준으로 정리한 문자열을 생성한다."""
    full_metrics = result_payload["model_results"]["full_model"]
    return f"""# Domain 6 — 전체 언어 및 개발자 특성 기반 고연봉 예측

## 분석 목표
- 여러 프로그래밍 언어 사용 여부와 개발자 특성으로 연봉 상위 25%를 예측하는 모델을 구현했습니다.
- 실제 입력 데이터로 학습하고, 모델 성능과 산출물을 저장했습니다.

## 데이터와 타깃
- 입력 파일: {result_payload['input_path']}
- 최종 유효 표본 수: {result_payload['stats']['valid_model_rows']}
- 학습/테스트 표본 수: {result_payload['train_rows']}/{result_payload['test_rows']}
- 고연봉 기준: 학습 데이터의 ConvertedCompYearly 75% 분위수 이상
- 고연봉 임계값: {result_payload['high_salary_threshold']:.2f}

## 모델 성능 요약
- Accuracy: {full_metrics['accuracy']:.3f}
- Balanced Accuracy: {full_metrics['balanced_accuracy']:.3f}
- Precision: {full_metrics['precision']:.3f}
- Recall: {full_metrics['recall']:.3f}
- F1: {full_metrics['f1']:.3f}
- ROC-AUC: {full_metrics['roc_auc']:.3f}
- Average Precision: {full_metrics['average_precision']:.3f}

## 실행 방법
```bash
cd /Users/baekjiheon/Desktop/SKALA/Day15_Python_2/domain_6ml
./domain6_ml/.venv/bin/python -m domain6_ml.model
```

## 산출물
- 결과 보고서: [result.html](result.html)
- 결과 요약: [result.md](result.md)
- 모델 비교: [model_comparison.csv](model_comparison.csv)
- 분류 보고서: [classification_report.csv](classification_report.csv)
- 혼동행렬: [confusion_matrix.csv](confusion_matrix.csv) / [confusion_matrix.png](confusion_matrix.png)
- 계수 결과: [feature_coefficients.csv](feature_coefficients.csv) / [language_coefficients.csv](language_coefficients.csv)
- 언어별 통계: [language_statistics.csv](language_statistics.csv)
- 언어별 평가: [language_evaluation.csv](language_evaluation.csv)
- 언어별 예측 상세: [language_predictions.csv](language_predictions.csv)
- 언어별 예측 요약: [language_prediction_summary.csv](language_prediction_summary.csv)
- 예측 샘플: [prediction_samples.csv](prediction_samples.csv)
- 모델 파일: [model.joblib](model.joblib)
- 메타데이터: [model_metrics.json](model_metrics.json) / [model_metadata.json](model_metadata.json)

## 참고
- 이 프로젝트는 교육·연구용 분석 목적입니다.
- 결과 보고서를 열어보려면 브라우저에서 [result.html](result.html)를 열면 됩니다.
"""


def create_result_html(
    result_payload: dict[str, Any],
    coeff_df: pd.DataFrame,
    lang_coeff_df: pd.DataFrame,
    language_stats_df: pd.DataFrame,
    language_eval_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> str:
    """학습 결과를 인쇄 가능한 상세 HTML 보고서로 생성한다."""
    def fmt(value: object, digits: int = 3) -> str:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"

    def money(value: object) -> str:
        if pd.isna(value):
            return "—"
        return f"${float(value):,.2f}"

    metrics = result_payload["model_results"]
    rows = []
    for mode in ["dummy", "language_only", "profile_only", "full_model"]:
        mode_metrics = metrics[mode]
        rows.append(
            "<tr>"
            f"<td>{escape(mode)}</td>"
            f"<td>{fmt(mode_metrics['accuracy'])}</td>"
            f"<td>{fmt(mode_metrics['balanced_accuracy'])}</td>"
            f"<td>{fmt(mode_metrics['precision'])}</td>"
            f"<td>{fmt(mode_metrics['recall'])}</td>"
            f"<td>{fmt(mode_metrics['f1'])}</td>"
            f"<td>{fmt(mode_metrics['roc_auc'])}</td>"
            f"<td>{fmt(mode_metrics['average_precision'])}</td>"
            "</tr>"
        )
    output_files = [
        "model_comparison.csv",
        "classification_report.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "feature_coefficients.csv",
        "language_coefficients.csv",
        "language_statistics.csv",
        "language_evaluation.csv",
        "language_predictions.csv",
        "language_prediction_summary.csv",
        "prediction_samples.csv",
        "model.joblib",
        "model_metrics.json",
        "model_metadata.json",
        "result.md",
        "result.html",
    ]
    language_eval_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['language']))}</td>"
        f"<td>{int(row['test_user_count'])}</td>"
        f"<td>{int(row['actual_high_salary_count'])}</td>"
        f"<td>{int(row['predicted_high_salary_count'])}</td>"
        f"<td>{fmt(row['actual_high_salary_rate'])}</td>"
        f"<td>{fmt(row['predicted_high_salary_rate'])}</td>"
        f"<td>{fmt(row['accuracy'])}</td>"
        f"<td>{fmt(row['f1'])}</td>"
        f"<td>{fmt(row['roc_auc'])}</td>"
        "</tr>"
        for _, row in language_eval_df.iterrows()
    )
    language_prediction_summary = pd.DataFrame(result_payload["language_prediction_summary"])
    language_prediction_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['language']))}</td>"
        f"<td>{int(row['test_user_count'])}</td>"
        f"<td>{int(row['actual_high_salary_count'])}</td>"
        f"<td>{int(row['predicted_high_salary_count'])}</td>"
        f"<td>{fmt(row['actual_high_salary_rate'])}</td>"
        f"<td>{fmt(row['predicted_high_salary_rate'])}</td>"
        f"<td>{fmt(row['mean_predicted_probability'])}</td>"
        f"<td>{fmt(row['accuracy'])}</td>"
        f"<td>{fmt(row['f1'])}</td>"
        "</tr>"
        for _, row in language_prediction_summary.iterrows()
    )
    coefficient_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['feature']))}</td>"
        f"<td>{fmt(row['coefficient'])}</td>"
        f"<td>{escape(str(row['direction']))}</td>"
        "</tr>"
        for _, row in coeff_df.head(20).iterrows()
    )
    language_coefficient_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['language']))}</td>"
        f"<td>{fmt(row['coefficient'])}</td>"
        f"<td>{escape(str(row['direction']))}</td>"
        "</tr>"
        for _, row in lang_coeff_df.iterrows()
    )
    confusion_rows = "".join(
        "<tr>"
        f"<th>{escape(str(index))}</th>"
        + "".join(f"<td>{int(value)}</td>" for value in row)
        + "</tr>"
        for index, row in confusion_df.iterrows()
    )
    full_metrics = metrics["full_model"]
    positive_language = language_stats_df.dropna(subset=["high_salary_rate"]).sort_values("high_salary_rate", ascending=False).iloc[0]
    median_language = language_stats_df.dropna(subset=["median_salary"]).sort_values("median_salary", ascending=False).iloc[0]
    tn, fp = int(confusion_df.iloc[0, 0]), int(confusion_df.iloc[0, 1])
    fn, tp = int(confusion_df.iloc[1, 0]), int(confusion_df.iloc[1, 1])
    return f"""<!DOCTYPE html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <title>Domain 6 결과 보고서</title>
  <style>
    :root {{ --navy: #17324d; --blue: #2563eb; --ink: #1f2937; --muted: #64748b; --line: #dbe3ec; --soft: #f5f8fc; --green: #087f5b; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #eef3f8; color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; line-height: 1.65; }}
    .page {{ max-width: 1240px; margin: 0 auto; background: white; min-height: 100vh; padding: 44px 56px 72px; }}
    .cover {{ border-bottom: 4px solid var(--blue); padding-bottom: 30px; margin-bottom: 30px; }}
    .eyebrow {{ color: var(--blue); font-size: 13px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }}
    h1 {{ color: var(--navy); font-size: 38px; line-height: 1.2; margin: 10px 0 12px; }}
    h2 {{ color: var(--navy); font-size: 24px; margin: 42px 0 12px; padding-bottom: 7px; border-bottom: 2px solid var(--line); }}
    h3 {{ color: var(--navy); margin-top: 26px; }}
    p {{ margin: 8px 0 14px; }}
    .subtitle {{ color: var(--muted); font-size: 17px; max-width: 920px; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 22px 0; }}
    .card {{ border: 1px solid var(--line); border-radius: 12px; padding: 17px; background: linear-gradient(135deg, #fff, var(--soft)); }}
    .card .label {{ color: var(--muted); font-size: 12px; font-weight: 700; }}
    .card .value {{ color: var(--navy); font-size: 27px; font-weight: 800; margin-top: 3px; }}
    .callout {{ border-left: 5px solid var(--blue); background: #eff6ff; padding: 16px 19px; margin: 18px 0; }}
    .warning {{ border-left-color: #d97706; background: #fff7ed; }}
    table {{ border-collapse: collapse; width: 100%; margin: 13px 0 20px; font-size: 13px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; text-align: right; vertical-align: top; }}
    th {{ background: var(--navy); color: white; font-weight: 700; white-space: nowrap; }}
    td:first-child, th:first-child {{ text-align: left; }}
    tr:nth-child(even) td {{ background: #fafcff; }}
    .wide {{ overflow-x: auto; }}
    .tag {{ display: inline-block; border-radius: 999px; background: #e8f5ef; color: var(--green); padding: 2px 9px; font-size: 12px; font-weight: 700; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .figure {{ text-align: center; border: 1px solid var(--line); border-radius: 12px; padding: 15px; background: var(--soft); }}
    .figure img {{ max-width: 100%; height: auto; }}
    ul {{ padding-left: 22px; }}
    li {{ margin: 4px 0; }}
    code {{ background: #f1f5f9; border-radius: 4px; padding: 2px 5px; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .footer {{ margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }}
    @media (max-width: 850px) {{ .page {{ padding: 28px 20px 50px; }} .cards {{ grid-template-columns: repeat(2, 1fr); }} .two-col {{ grid-template-columns: 1fr; }} h1 {{ font-size: 30px; }} }}
    @media print {{ body {{ background: white; }} .page {{ max-width: none; padding: 0; }} h2 {{ break-after: avoid; }} table {{ break-inside: avoid; }} .footer {{ display: none; }} }}
  </style>
</head>
<body>
<main class="page">
  <header class="cover">
    <div class="eyebrow">Domain 6 · Machine Learning Report</div>
    <h1>프로그래밍 언어 및 개발자 특성 기반<br>고연봉 예측 분석 보고서</h1>
    <p class="subtitle">Stack Overflow 개발자 설문 데이터를 이용해 연봉 상위 25% 여부를 분류하고, 사용언어별 연봉 통계·테스트 성능·예측 결과를 함께 분석한 상세 보고서입니다.</p>
    <p class="meta">생성 시각: {escape(str(result_payload.get('generated_at', '학습 실행 결과')))} · 입력 행 수: {result_payload['stats']['total_loaded_rows']:,} · 분석 유효 행 수: {result_payload['stats']['valid_model_rows']:,}</p>
  </header>

  <h2>1. 핵심 요약</h2>
  <div class="cards">
    <div class="card"><div class="label">최종 모델 ROC-AUC</div><div class="value">{fmt(full_metrics['roc_auc'])}</div></div>
    <div class="card"><div class="label">최종 모델 F1</div><div class="value">{fmt(full_metrics['f1'])}</div></div>
    <div class="card"><div class="label">고연봉 임계값</div><div class="value">{money(result_payload['high_salary_threshold'])}</div></div>
    <div class="card"><div class="label">모델 재로드 검증</div><div class="value">{'성공' if result_payload['model_reload_verified'] else '실패'}</div></div>
  </div>
  <div class="callout"><strong>요약 결론.</strong> 전체 특성을 사용한 Logistic Regression은 테스트셋에서 ROC-AUC {fmt(full_metrics['roc_auc'])}, F1 {fmt(full_metrics['f1'])}을 기록했습니다. 다만 profile-only 모델 대비 개선 폭은 F1 {fmt(result_payload['model_results']['full_model']['f1'] - result_payload['model_results']['profile_only']['f1'])}, ROC-AUC {fmt(result_payload['model_results']['full_model']['roc_auc'] - result_payload['model_results']['profile_only']['roc_auc'])}로, 언어 정보보다 경력·직무·국가 등 프로필 정보의 영향이 더 큽니다.</div>
  <ul>
    <li>입력 파일: <code>{escape(str(result_payload['input_path']))}</code></li>
    <li>학습/테스트 표본: {result_payload['train_rows']:,} / {result_payload['test_rows']:,}</li>
    <li>고연봉 정의: <strong>ConvertedCompYearly가 학습 데이터 75% 분위수 이상</strong></li>
    <li>분석 기준 임계값: <strong>{money(result_payload['high_salary_threshold'])}</strong></li>
    <li>분석 대상 언어: {len(LANGUAGE_FEATURES)}개 · 테스트 데이터에서 실제 사용자가 존재하는 언어: {int((language_eval_df['test_user_count'] > 0).sum())}개</li>
  </ul>

  <h2>2. 분석 목적과 해석 범위</h2>
  <p>본 프로젝트의 예측 타깃은 실제 연봉 금액 자체가 아니라, 응답자의 연봉이 분석 표본 내 상위 25%에 속하는지 여부입니다. 따라서 결과의 “예측 확률”은 특정 언어를 사용하면 연봉이 얼마가 된다는 의미가 아니라, 해당 응답자가 고연봉 그룹에 속할 확률을 뜻합니다.</p>
  <p>언어별 통계는 관측된 연봉 분포를 보여주며, 언어별 평가는 최종 full_model을 각 언어 사용자의 테스트 하위집합에 적용한 결과입니다. 언어별 결과에는 국가·경력·직무·학력 차이가 섞여 있으므로 인과관계나 언어 자체의 순수한 효과로 해석해서는 안 됩니다.</p>

  <h2>3. 데이터와 타깃 정의</h2>
  <div class="wide">
  <table>
    <thead><tr><th>항목</th><th>내용</th></tr></thead>
    <tbody>
      <tr><td>원본 데이터</td><td>{escape(str(result_payload['input_path']))}</td></tr>
      <tr><td>전체 행 수</td><td>{result_payload['stats']['total_loaded_rows']:,}</td></tr>
      <tr><td>최종 유효 행 수</td><td>{result_payload['stats']['valid_model_rows']:,}</td></tr>
      <tr><td>학습 / 테스트 분할</td><td>{result_payload['train_rows']:,} / {result_payload['test_rows']:,} · random_state=42</td></tr>
      <tr><td>학습 고연봉 비율</td><td>{fmt(result_payload['train_target_distribution'].get(1, 0) / result_payload['train_rows'])}</td></tr>
      <tr><td>테스트 고연봉 비율</td><td>{fmt(result_payload['test_target_distribution'].get(1, 0) / result_payload['test_rows'])}</td></tr>
      <tr><td>결측·비정상 제거</td><td>연봉 결측 0 · 연봉 0 이하 0 · 언어 결측 0 · 중복 0</td></tr>
    </tbody>
  </table>
  </div>
  <div class="callout warning"><strong>주의.</strong> 상위 25% 임계값은 현재 데이터의 상대적 기준입니다. 국가별 물가·환율·직무 구성·설문 응답 편향이 포함되어 있으므로 다른 국가나 다른 연도에 그대로 일반화할 수 없습니다.</div>

  <h2>4. 특성 설계와 전처리</h2>
  <div class="two-col">
    <div><h3>수치형 특성</h3><ul><li>YearsCodeProNumeric: 전문 개발 경력</li><li>LanguageCount: 응답자가 선택한 전체 언어 수</li></ul></div>
    <div><h3>범주형 특성</h3><ul><li>RemoteWork</li><li>DevType</li><li>EdLevel</li><li>Employment</li><li>OrgSize</li><li>Country</li></ul></div>
  </div>
  <p>언어 사용 여부는 Python, R, Julia, Java, Kotlin, Scala, JavaScript, TypeScript, PHP, C, C++, Rust, C#, Go, Swift를 각각 0/1 특성으로 만들었습니다. Java와 JavaScript, C와 C++, C#는 세미콜론 단위로 분리하여 문자열 부분 일치 오류를 피했습니다. 연봉 원본과 ResponseId는 모델 입력에서 제외했습니다.</p>

  <h2>5. 모델 비교와 최종 성능</h2>
  <table>
    <thead>
      <tr><th>모델</th><th>Accuracy</th><th>Balanced Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Average Precision</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <p><span class="tag">평가 해석</span> 클래스 불균형이 있으므로 Accuracy 하나만 보지 않고 Balanced Accuracy, F1, ROC-AUC, Average Precision을 함께 사용했습니다. 최종 모델은 고연봉 클래스에 대해 Recall {fmt(full_metrics['recall'])}, Precision {fmt(full_metrics['precision'])}을 기록했습니다.</p>

  <h2>6. 혼동행렬과 예측 오류</h2>
  <div class="two-col">
    <div class="figure"><img src="confusion_matrix.png" alt="최종 모델 혼동행렬"><p class="meta">실제/예측 고연봉 여부 혼동행렬</p></div>
    <div><table><thead><tr><th>실제/예측</th><th>Below Top25</th><th>High Salary Top25</th></tr></thead><tbody>{confusion_rows}</tbody></table><ul><li>실제 고연봉을 고연봉으로 맞힌 수: {tp:,}건</li><li>실제 고연봉을 놓친 수: {fn:,}건</li><li>고연봉으로 예측했으나 실제로는 아닌 수: {fp:,}건</li><li>실제 저연봉을 저연봉으로 맞힌 수: {tn:,}건</li><li>고연봉 Recall이 높지만, Precision은 약 {fmt(full_metrics['precision'])}이므로 예측 고연봉 판정의 약 {fmt(1 - full_metrics['precision'])}는 오탐입니다.</li></ul></div>
  </div>

  <h2>7. 주요 예측 특성</h2>
  <p>아래 계수는 Logistic Regression의 고연봉 분류 방향을 나타냅니다. 연봉 증가액이나 인과효과가 아니며, 범주형 변수는 기준 범주와 비교한 상대적 계수입니다.</p>
  <div class="wide"><table><thead><tr><th>특성</th><th>계수</th><th>방향</th></tr></thead><tbody>{coefficient_rows}</tbody></table></div>

  <h2>8. 언어별 연봉 통계</h2>
  <p>전체 유효 데이터에서 계산한 관측 통계입니다. 고연봉 비율은 학습 데이터에서 계산된 임계값 {money(result_payload['high_salary_threshold'])} 이상인 비율입니다.</p>
  <div class="callout"><strong>관측상 상위 언어.</strong> 고연봉 비율은 {escape(str(positive_language['language']))}가 {fmt(positive_language['high_salary_rate'])}, 중앙 연봉은 {escape(str(median_language['language']))}가 {money(median_language['median_salary'])}로 가장 높았습니다. 표본 수와 국가·직무 분포가 다르므로 단순 순위로 언어의 우열을 판단하면 안 됩니다.</div>

  <h2>9. 언어별 테스트 평가</h2>
  <table>
    <thead><tr><th>언어</th><th>테스트 수</th><th>실제 고연봉자 수</th><th>예측 고연봉자 수</th><th>실제 고연봉 비율</th><th>예측 고연봉 비율</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th></tr></thead>
    <tbody>{language_eval_rows}</tbody>
  </table>

  <p>언어별 고연봉자 수는 테스트 하위집합에서 실제 타깃 1의 개수와 모델 예측 1의 개수를 각각 표시합니다. 언어별 ROC-AUC는 해당 언어 사용자의 테스트 하위집합에서 모델의 순위화 성능을 뜻합니다. 표본 수가 작은 언어의 지표는 불안정할 수 있고, 사용자가 없는 언어는 평가 지표를 계산할 수 없습니다.</p>

  <h2>10. 언어별 예측 요약</h2>
  <p>아래 표는 각 언어 사용자의 테스트 데이터에서 실제 고연봉자 수, 모델의 예측 고연봉자 수와 비율, 평균 예측 확률, 분류 성능을 요약한 것입니다. 상세 응답자 단위 결과는 <a href="language_predictions.csv">language_predictions.csv</a>에서 확인할 수 있습니다.</p>
  <div class="wide"><table><thead><tr><th>언어</th><th>테스트 수</th><th>실제 고연봉자 수</th><th>예측 고연봉자 수</th><th>실제 고연봉 비율</th><th>예측 고연봉 비율</th><th>평균 예측 확률</th><th>Accuracy</th><th>F1</th></tr></thead><tbody>{language_prediction_rows}</tbody></table></div>

  <h2>11. 언어 계수 분석</h2>
  <p>언어 계수는 다른 언어·국가·직무·경력 특성을 함께 넣은 full_model에서의 조건부 연관 방향입니다. 여러 언어를 동시에 사용하는 응답자가 많기 때문에 단일 언어의 독립적인 연봉 효과로 해석할 수 없습니다.</p>
  <div class="wide"><table><thead><tr><th>언어</th><th>계수</th><th>방향</th></tr></thead><tbody>{language_coefficient_rows}</tbody></table></div>

  <h2>12. 생성된 산출물</h2>
  <ul>
    {''.join(f'<li><a href="{escape(path)}">{escape(path)}</a></li>' for path in output_files)}
  </ul>
  <p>HTML 보고서 외에도 CSV는 재분석용 원자료, JSON은 구조화된 메타데이터와 결과 요약, JOBLIB 파일은 재사용 가능한 학습 Pipeline입니다.</p>

  <h2>13. 결론과 한계</h2>
  <ol>
    <li>현재 모델은 전체 개발자의 연봉 상위 25% 여부를 분류하는 데 의미 있는 성능을 보였습니다.</li>
    <li>언어별 통계를 통해 Go, Swift, Scala 등에서 상대적으로 높은 고연봉 비율이 관측되었지만, 이는 국가·직무·경력과 결합된 결과입니다.</li>
    <li>언어만 사용한 모델보다 프로필-only 모델이 훨씬 강하므로, 사용언어 단독으로 고연봉을 설명하기는 어렵습니다.</li>
    <li>평가는 단일 랜덤 분할 기준이므로 교차검증·연도별 검증·외부 데이터 검증이 추가로 필요합니다.</li>
    <li>Stack Overflow 설문은 무작위 모집단 조사가 아니며, 본 결과를 채용·연봉 책정·인사 의사결정에 사용해서는 안 됩니다.</li>
  </ol>
  <div class="footer">Domain 6 분석 보고서 · 교육·연구용 결과 · 타깃은 실제 연봉 금액이 아닌 상위 25% 분류</div>
</main>
</body>
</html>
"""


def create_result_markdown(
    result_payload: dict[str, Any],
    coeff_df: pd.DataFrame,
    lang_coeff_df: pd.DataFrame,
    language_stats_df: pd.DataFrame,
    language_eval_df: pd.DataFrame,
) -> str:
    """결과 요약 Markdown 문서를 생성한다.

    입력:
        result_payload: 실행 결과 딕셔너리.
        coeff_df: 전체 계수 DataFrame.
        lang_coeff_df: 언어 계수 DataFrame.

    반환:
        Markdown 문자열.

    발생 가능한 예외:
        None.
    """
    metrics = result_payload["model_results"]
    positive_features = coeff_df[coeff_df["direction"] == "positive"].head(15)
    negative_features = coeff_df[coeff_df["direction"] == "negative"].head(15)
    lines = []
    lines.append("# Domain 6 — 전체 언어 및 개발자 특성 기반 고연봉 예측")
    lines.append("")
    lines.append("## 1. 분석 목적")
    lines.append("다양한 프로그래밍 언어 사용 여부와 개발자 특성으로 연봉 상위 25% 여부를 예측하는 모델을 구현했다.")
    lines.append("")
    lines.append("## 2. 분석 질문")
    lines.append("언어 사용 여부와 개발자 특성으로 연봉 상위 25% 여부를 예측할 수 있는가?")
    lines.append("")
    lines.append("## 3. 입력 데이터")
    lines.append(f"- 실제 입력 파일 경로: {result_payload['input_path']}")
    lines.append(f"- 전체 로딩 행 수: {result_payload['stats']['total_loaded_rows']}")
    lines.append(f"- 제외 행 수: {result_payload['stats']['salary_missing_excluded'] + result_payload['stats']['salary_nonpositive_excluded'] + result_payload['stats']['language_missing_excluded'] + result_payload['stats']['duplicate_excluded']}")
    lines.append(f"- 최종 유효 표본 수: {result_payload['stats']['valid_model_rows']}")
    lines.append(f"- 학습 표본 수: {result_payload['train_rows']}")
    lines.append(f"- 테스트 표본 수: {result_payload['test_rows']}")
    lines.append("")
    lines.append("## 4. ML 특성 생성")
    lines.append("- YearsCodeProNumeric은 YearsCodePro 문자열을 숫자로 변환해 생성했다.")
    lines.append("- LanguageCount는 선택된 언어 개수를 계산해 생성했다.")
    lines.append("- 언어별 0·1 특성은 LanguageHaveWorkedWith를 세미콜론 기준으로 정확히 분리해 생성했다.")
    lines.append("- Java와 JavaScript, C와 C++, C#는 문자열 분리 기준으로 정확히 구분했다.")
    lines.append("- 파생 특성은 data 폴더에 저장하지 않고 메모리상에서만 사용했다.")
    lines.append("")
    lines.append("## 5. 고연봉 기준")
    lines.append(f"- 75% 분위수: {HIGH_SALARY_QUANTILE}")
    lines.append(f"- 실제 임계 연봉: {result_payload['high_salary_threshold']:.2f}")
    lines.append("- 클래스 0은 임계값 미만, 클래스 1은 임계값 이상이다.")
    lines.append("- 학습 데이터에서만 임계값을 계산해 누수를 방지했다.")
    lines.append("")
    lines.append("## 6. 사용 특성")
    lines.append(f"- 수치형: {', '.join(NUMERIC_FEATURES)}")
    lines.append(f"- 범주형: {', '.join(result_payload['categorical_features'])}")
    lines.append(f"- 언어 사용 여부: {', '.join(LANGUAGE_FEATURES)}")
    lines.append(f"- Country 처리: {'포함' if result_payload['country_included'] else '제외'}")
    lines.append("- 누수 방지를 위해 ResponseId, ConvertedCompYearly, LogSalary, LanguageHaveWorkedWith, HighSalaryTop25를 제외했다.")
    lines.append("")
    lines.append("## 7. 모델 구조")
    lines.append("- DummyClassifier: 가장 흔한 클래스를 예측하는 기준 모델")
    lines.append("- language_only LogisticRegression: 언어 사용 여부와 LanguageCount만 사용")
    lines.append("- profile_only LogisticRegression: 경력·직무·학력·근무형태·고용형태·조직 규모 등 일반 특성 사용")
    lines.append("- full_model LogisticRegression: 언어와 일반 특성을 함께 사용")
    lines.append("- ColumnTransformer 기반 전처리")
    lines.append("- class_weight=balanced")
    lines.append("")
    lines.append("## 8. 클래스 분포")
    lines.append(f"- 학습 데이터: {result_payload['train_target_distribution']}")
    lines.append(f"- 테스트 데이터: {result_payload['test_target_distribution']}")
    lines.append("- 클래스 불균형이 있으므로 Accuracy만으로 성능을 판단하지 않았다.")
    lines.append("")
    lines.append("## 9. 모델 비교")
    for mode in ["dummy", "language_only", "profile_only", "full_model"]:
        mode_metrics = result_payload["model_results"][mode]
        lines.append(f"- {mode}: Accuracy={mode_metrics['accuracy']:.3f}, Balanced Accuracy={mode_metrics['balanced_accuracy']:.3f}, F1={mode_metrics['f1']:.3f}, ROC-AUC={mode_metrics['roc_auc']:.3f}, Average Precision={mode_metrics['average_precision']:.3f}")
    lines.append("")
    lines.append("## 10. 최종 모델 평가")
    full_metrics = result_payload["model_results"]["full_model"]
    lines.append(f"- Accuracy: {full_metrics['accuracy']:.3f}")
    lines.append(f"- Balanced Accuracy: {full_metrics['balanced_accuracy']:.3f}")
    lines.append(f"- Precision: {full_metrics['precision']:.3f}")
    lines.append(f"- Recall: {full_metrics['recall']:.3f}")
    lines.append(f"- F1: {full_metrics['f1']:.3f}")
    lines.append(f"- ROC-AUC: {full_metrics['roc_auc']:.3f}")
    lines.append(f"- Average Precision: {full_metrics['average_precision']:.3f}")
    lines.append("")
    lines.append("## 11. 혼동행렬 해석")
    lines.append("- 혼동행렬은 confusion_matrix.csv와 confusion_matrix.png로 저장했다.")
    lines.append("- 실제 결과는 해당 파일을 참고한다.")
    lines.append("")
    lines.append("## 12. 주요 예측 특성")
    for _, row in positive_features.iterrows():
        lines.append(f"- 양의 계수: {row['feature']} ({row['coefficient']:.3f})")
    for _, row in negative_features.iterrows():
        lines.append(f"- 음의 계수: {row['feature']} ({row['coefficient']:.3f})")
    lines.append("- 언어 특성 계수는 language_coefficients.csv에 저장했다.")
    lines.append("- 계수는 연봉 증가액이 아니라 상위 25% 예측과의 연관 방향을 나타낸다.")
    lines.append("")
    lines.append("## 13. 언어별 통계")
    lines.append("언어별 사용 규모, 연봉 분포, 학습 기준 고연봉 비율을 계산했다.")
    lines.append("")
    lines.append("| 언어 | 사용자 수 | 사용 비율 | 평균 연봉 | 중앙 연봉 | 고연봉 비율 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in language_stats_df.iterrows():
        lines.append(f"| {row['language']} | {int(row['user_count'])} | {row['usage_rate']:.3f} | {row['mean_salary']:.2f} | {row['median_salary']:.2f} | {row['high_salary_rate']:.3f} |")
    lines.append("")
    lines.append("## 14. 언어별 테스트 평가")
    lines.append("full_model을 테스트셋의 각 언어 사용 하위집합에 적용해 평가했다.")
    lines.append("")
    lines.append("| 언어 | 테스트 수 | 실제 고연봉자 수 | 예측 고연봉자 수 | 실제 고연봉 비율 | 예측 고연봉 비율 | Accuracy | F1 | ROC-AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in language_eval_df.iterrows():
        lines.append(f"| {row['language']} | {int(row['test_user_count'])} | {int(row['actual_high_salary_count'])} | {int(row['predicted_high_salary_count'])} | {row['actual_high_salary_rate']:.3f} | {row['predicted_high_salary_rate']:.3f} | {row['accuracy']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} |")
    lines.append("")
    lines.append("- 언어별 실제·예측 고연봉자 수와 비율, 상세 예측은 language_predictions.csv, 언어별 요약은 language_prediction_summary.csv에 저장했다.")
    lines.append("")
    lines.append("## 15. 모델 저장 및 검증")
    lines.append(f"- 모델 저장 경로: {DOMAIN_ROOT / 'model.joblib'}")
    lines.append("- Pipeline 전체를 저장했다.")
    lines.append(f"- 재로드 검증 결과: {'성공' if result_payload['model_reload_verified'] else '실패'}")
    lines.append("")
    lines.append("## 16. 핵심 결론")
    lines.append("- 실제 데이터를 기반으로 모델을 학습하고 평가했다.")
    lines.append("- 언어 특성의 추가 예측 기여를 비교했으며, 결과는 model_comparison.csv와 model_metrics.json에 기록했다.")
    lines.append("- 모델은 교육·연구용 분석으로만 사용해야 하며, 실제 채용·연봉 결정에 사용해서는 안 된다.")
    lines.append("")
    lines.append("## 17. 분석 한계")
    lines.append("- Stack Overflow 설문은 무작위 표본이 아니다.")
    lines.append("- 전체 개발자 모집단으로 일반화할 수 없다.")
    lines.append("- 언어 사용 여부와 연봉 사이의 인과관계를 의미하지 않는다.")
    lines.append("- 경력, 직무, 학력 외에도 산업, 지역, 회사, 직급, 숙련도 등이 영향을 줄 수 있다.")
    lines.append("- 상위 25%는 현재 분석 표본 안의 상대적 기준이다.")
    lines.append("- 여러 국가가 포함됐다면 국가별 임금 차이가 결과에 영향을 줄 수 있다.")
    lines.append("- 모델 계수는 연봉 증가액이 아니다.")
    lines.append("- 모델을 실제 채용, 연봉 책정, 인사 의사결정에 사용해서는 안 된다.")
    return "\n".join(lines)


def create_language_salary_markdown(
    result_payload: dict[str, Any],
    language_statistics: pd.DataFrame,
    language_evaluation: pd.DataFrame,
) -> str:
    """언어별 연봉 분석을 중심으로 한 Markdown 요약을 생성한다."""
    metadata = result_payload.get("report_metadata", {})
    salary_column = metadata.get("salary_column", "ConvertedCompYearly")
    currency_label = metadata.get("currency_label", "원본 데이터에 통화 단위 컬럼이 없어 확인되지 않음")
    eligible = language_statistics.loc[language_statistics["user_count"].ge(REPORT_SAMPLE_MIN)]
    eligible = eligible if not eligible.empty else language_statistics
    top_median = eligible.sort_values("median_salary", ascending=False).iloc[0]
    top_users = language_statistics.sort_values("user_count", ascending=False).iloc[0]
    lines = [
        "# Programming Language Salary Analysis",
        "",
        "# 프로그래밍 언어별 개발자 연봉 분석 보고서",
        "",
        "Stack Overflow 개발자 설문 기반 언어 사용자 집단별 연봉 수준·분포·표본 특성 비교",
        "",
        "## Executive Summary",
        "",
        f"- 분석 응답자 수: {result_payload['stats']['valid_model_rows']:,}",
        f"- 분석 언어 수: {len(language_statistics):,}",
        f"- 중앙 연봉이 가장 높은 언어(표본 {REPORT_SAMPLE_MIN}명 이상 우선): {top_median['language']} ({report_money(top_median['median_salary'])})",
        f"- 사용자 수가 가장 많은 언어: {top_users['language']} ({int(top_users['user_count']):,}명)",
        f"- 연봉 컬럼: {salary_column}",
        f"- 통화 단위: {currency_label}",
        "",
        "## 핵심 해석",
        "",
        "이번 보고서는 연봉 상위 25% 분류 모델이 아니라 언어 사용자 집단의 연봉 수준과 분포를 주제로 한다.",
        "언어별 차이는 국가, 경력, 직무, 회사 규모 차이가 함께 반영된 관찰 결과이며 인과관계를 의미하지 않는다.",
        "",
        "## 언어별 연봉 통계",
        "",
        "| 언어 | 사용자 수 | 평균 연봉 | 중앙 연봉 | 평균-중앙값 차이 | Q1 | Q3 | IQR | 실제 고연봉자 수 | 실제 고연봉 비율 | 표본 상태 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in language_statistics.sort_values("median_salary", ascending=False).iterrows():
        lines.append(
            f"| {row['language']} | {int(row['user_count']):,} | {report_money(row['average_salary'])} | "
            f"{report_money(row['median_salary'])} | {report_money(row['salary_mean_median_gap'])} | "
            f"{report_money(row['q1_salary'])} | {report_money(row['q3_salary'])} | {report_money(row['iqr_salary'])} | "
            f"{int(row['high_salary_count']):,} | {report_percent(row['high_salary_rate'])} | {row['sample_warning']} |"
        )
    lines.extend([
        "",
        "## 언어별 분류 모델 진단 부록",
        "",
        "아래 결과는 전체 Full 모델을 언어 사용자 하위집합에 적용한 보조 진단이다. 언어별 독립 모델이나 실제 연봉 금액 예측이 아니다.",
        "",
        "| 언어 | 테스트 수 | 실제 고연봉자 수 | 예측 고연봉자 수 | 실제 비율 | 예측 비율 | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in language_evaluation.iterrows():
        lines.append(
            f"| {row['language']} | {int(row['test_user_count']):,} | {int(row.get('actual_high_salary_count', 0)):,} | "
            f"{int(row.get('predicted_high_salary_count', 0)):,} | {report_percent(row.get('actual_high_salary_rate'))} | "
            f"{report_percent(row.get('predicted_high_salary_rate'))} | {row.get('f1', float('nan')):.3f} | {row.get('roc_auc', float('nan')):.3f} |"
        )
    lines.extend([
        "",
        "## 방법론과 한계",
        "",
        "- 한 응답자는 여러 언어를 선택할 수 있어 언어 사용자 집단은 서로 배타적이지 않다.",
        "- 연봉은 자기보고 설문값이며, 국가별 임금·환율·생활비 차이가 반영될 수 있다.",
        "- 평균 연봉은 일부 극단값의 영향을 받을 수 있으므로 중앙 연봉과 사분위 범위를 함께 확인해야 한다.",
        "- 표본이 작은 언어는 해석 주의 또는 표본 부족으로 표시했다.",
        "- 결과를 채용, 연봉 책정, 개인 평가에 직접 사용해서는 안 된다.",
        "",
        "메인 인터랙티브 보고서: [language_salary_report.html](language_salary_report.html)",
    ])
    return "\n".join(lines)


def create_salary_readme_content(result_payload: dict[str, Any]) -> str:
    """프로젝트 README를 언어별 연봉 분석 중심으로 갱신한다."""
    metadata = result_payload.get("report_metadata", {})
    return "\n".join([
        "# Programming Language Salary Analysis",
        "",
        "Stack Overflow 개발자 설문 데이터를 이용한 프로그래밍 언어별 개발자 연봉 분석 프로젝트입니다.",
        "",
        "## 핵심 분석 주제",
        "",
        "언어 사용자 집단별 사용자 수, 평균 연봉, 중앙 연봉, 분포, 사분위 범위와 실제 고연봉 비율을 비교합니다.",
        "연봉 상위 25% 분류 모델은 언어별 연봉 분석을 보조하는 부록으로만 사용합니다.",
        "",
        "## 데이터",
        "",
        f"- 유효 분석 응답자: {result_payload['stats']['valid_model_rows']:,}명",
        f"- 연봉 컬럼: {metadata.get('salary_column', 'ConvertedCompYearly')}",
        f"- 통화 단위: {metadata.get('currency_label', '원본 데이터에 통화 단위 컬럼이 없어 확인되지 않음')}",
        f"- 데이터 기준 연도: {metadata.get('survey_year', '확인되지 않음')}",
        "",
        "## 실행",
        "",
        "```bash",
        "cd /Users/baekjiheon/Desktop/SKALA/Day15_Python_2/domain_6ml",
        "./domain6_ml/.venv/bin/python -m domain6_ml.model",
        "```",
        "",
        "## 주요 산출물",
        "",
        "- [language_salary_report.html](language_salary_report.html): 독립형 Plotly 인터랙티브 메인 보고서",
        "- [result.html](result.html): 메인 보고서 호환 링크",
        "- [result.md](result.md): Markdown 요약",
        "- [language_statistics.csv](language_statistics.csv): 언어별 연봉·분포·고연봉 통계",
        "- [language_evaluation.csv](language_evaluation.csv): 언어 사용자 그룹별 분류 모델 보조 진단",
        "- [language_prediction_summary.csv](language_prediction_summary.csv): 언어별 실제·예측 고연봉 요약",
        "",
        "## 주의사항",
        "",
        "한 응답자는 여러 언어를 선택할 수 있으며, 관찰된 언어별 연봉 차이는 인과관계를 의미하지 않습니다.",
        "국가, 경력, 직무, 회사 규모 등 교란요인을 함께 고려해야 합니다.",
    ])


def run_ml_analysis(input_path: Path | None = None) -> dict[str, Any]:
    """Domain 6 고연봉 예측 분석을 실행하고 결과 딕셔너리를 반환한다.

    입력:
        input_path: 분석할 입력 CSV 경로.

    반환:
        실행 결과 딕셔너리.

    발생 가능한 예외:
        AnalysisError: 분석 과정에서 발생한 오류.
    """
    resolved_path = resolve_input_path(input_path)
    print_status(f"[정보] 입력 파일: {resolved_path}")
    header_df = read_csv_header(resolved_path)
    columns_to_load = determine_columns_to_load(header_df)
    print_status(f"[정보] 사용 컬럼: {columns_to_load}")
    df = load_source_data(resolved_path, columns_to_load)
    validate_source_data(df)

    stats = {"total_loaded_rows": int(len(df)), "salary_missing_excluded": 0, "salary_nonpositive_excluded": 0, "language_missing_excluded": 0, "duplicate_excluded": 0, "valid_model_rows": 0}
    work_df, stats = clean_model_rows(df)

    for col in ["LanguageHaveWorkedWith", "YearsCodePro", "RemoteWork", "DevType", "EdLevel", "Employment", "OrgSize", "Country"]:
        if col in work_df.columns:
            if work_df[col].dtype == "object":
                work_df[col] = work_df[col].apply(lambda value: value.strip() if isinstance(value, str) else value)
                work_df[col] = work_df[col].replace({"": pd.NA})
    work_df = work_df.copy()
    if "YearsCodeProNumeric" in work_df.columns:
        years_numeric = pd.to_numeric(work_df["YearsCodeProNumeric"], errors="coerce")
        work_df["YearsCodeProNumeric"] = years_numeric
    elif "YearsCodePro" in work_df.columns:
        work_df["YearsCodeProNumeric"] = convert_years_code_pro(work_df["YearsCodePro"])
    else:
        raise AnalysisError("YearsCodePro 또는 YearsCodeProNumeric 컬럼이 필요합니다.")

    if "LanguageHaveWorkedWith" in work_df.columns:
        work_df["LanguageHaveWorkedWith"] = work_df["LanguageHaveWorkedWith"].astype(str)
    work_df = create_language_features(work_df)
    work_df["LanguageCount"] = create_language_count(work_df)
    model_df = create_model_features(work_df)
    categorical_features, country_included = determine_country_feature(model_df)
    train_df, test_df = split_train_test_data(model_df)
    high_salary_threshold = float(train_df["ConvertedCompYearly"].quantile(HIGH_SALARY_QUANTILE))
    y_train = create_high_salary_targets(train_df, high_salary_threshold)
    y_test = create_high_salary_targets(test_df, high_salary_threshold)
    if len(pd.unique(y_train)) < 2 or len(pd.unique(y_test)) < 2:
        raise AnalysisError("학습 또는 테스트 데이터가 단일 클래스만 포함합니다.")

    feature_columns = [col for col in model_df.columns if col not in LEAKAGE_COLUMNS and col != "HighSalaryTop25"]
    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    validate_no_data_leakage(X_train, X_test, y_train, y_test)

    dummy_model = train_dummy_baseline(X_train, y_train)
    dummy_prob = None
    try:
        dummy_prob = dummy_model.predict_proba(X_test)[:, 1]
    except Exception:
        dummy_prob = None
    dummy_metrics = evaluate_classifier("dummy", y_test, dummy_model.predict(X_test), dummy_prob)

    models = train_logistic_models(X_train, X_test, y_train, y_test, categorical_features)
    language_only_metrics = evaluate_classifier("language_only", y_test, models["language_only"]["predictions"], models["language_only"]["probabilities"])
    profile_only_metrics = evaluate_classifier("profile_only", y_test, models["profile_only"]["predictions"], models["profile_only"]["probabilities"])
    full_model_metrics = evaluate_classifier("full_model", y_test, models["full_model"]["predictions"], models["full_model"]["probabilities"])

    model_results = {
        "dummy": dummy_metrics,
        "language_only": language_only_metrics,
        "profile_only": profile_only_metrics,
        "full_model": full_model_metrics,
    }

    performance_differences = {
        "language_only_vs_full_model_f1": model_results["full_model"]["f1"] - model_results["language_only"]["f1"],
        "profile_only_vs_full_model_f1": model_results["full_model"]["f1"] - model_results["profile_only"]["f1"],
        "profile_only_vs_full_model_roc_auc": model_results["full_model"]["roc_auc"] - model_results["profile_only"]["roc_auc"],
        "profile_only_vs_full_model_average_precision": model_results["full_model"]["average_precision"] - model_results["profile_only"]["average_precision"],
    }

    model_comparison_df = create_model_comparison(model_results)
    model_comparison_df.to_csv(DOMAIN_ROOT / "model_comparison.csv", index=False)

    cm_df, cm_path = create_confusion_matrix_outputs(models["full_model"]["predictions"], y_test)
    report_df = create_classification_report(y_test, models["full_model"]["predictions"])

    full_pipeline = models["full_model"]["pipeline"]
    coeff_df = extract_feature_coefficients(full_pipeline, X_train.columns.tolist())
    coeff_df.to_csv(DOMAIN_ROOT / "feature_coefficients.csv", index=False)
    lang_coeff_df = create_language_coefficient_output(coeff_df)
    lang_coeff_df.to_csv(DOMAIN_ROOT / "language_coefficients.csv", index=False)

    prediction_samples = create_prediction_samples(full_pipeline, X_test, y_test, model_df)
    language_stats_df = create_language_statistics(model_df, high_salary_threshold)
    language_stats_df.to_csv(DOMAIN_ROOT / "language_statistics.csv", index=False)
    language_eval_df = create_language_evaluation_output(
        X_test,
        y_test,
        models["full_model"]["predictions"],
        models["full_model"]["probabilities"],
    )
    language_eval_df.to_csv(DOMAIN_ROOT / "language_evaluation.csv", index=False)
    language_predictions_df, language_prediction_summary_df = create_language_prediction_outputs(
        model_df,
        X_test,
        y_test,
        models["full_model"]["predictions"],
        models["full_model"]["probabilities"],
    )
    language_predictions_df.to_csv(DOMAIN_ROOT / "language_predictions.csv", index=False)
    language_prediction_summary_df.to_csv(DOMAIN_ROOT / "language_prediction_summary.csv", index=False)
    model_path = save_model(full_pipeline)
    reload_verified = verify_saved_model(full_pipeline, model_path, X_test)
    print_status(f"[검증 성공] 저장 후 다시 불러온 Pipeline의 예측 결과가 동일합니다.")

    train_target_dist = {int(k): int(v) for k, v in dict(y_train.value_counts().sort_index()).items()}
    test_target_dist = {int(k): int(v) for k, v in dict(y_test.value_counts().sort_index()).items()}

    metrics_payload = create_model_metrics({}, stats, len(train_df), len(test_df), high_salary_threshold, train_target_dist, test_target_dist, model_results, performance_differences)
    metadata_payload = create_model_metadata(resolved_path, resolved_path.stat().st_size, stats, len(train_df), len(test_df), high_salary_threshold, categorical_features, country_included, model_results, model_path, reload_verified)
    survey_year_values = []
    if "SurveyYear" in work_df.columns:
        survey_year_values = sorted({str(value) for value in work_df["SurveyYear"].dropna().tolist()})
    currency_columns = [column for column in ["Currency", "CompCurrency", "CurrencyCode"] if column in work_df.columns]
    currency_label = "원본 데이터에 통화 단위 컬럼이 없어 확인되지 않음"
    if currency_columns:
        currency_values = sorted({str(value) for value in work_df[currency_columns[0]].dropna().unique()})
        if currency_values:
            currency_label = f"{currency_columns[0]}: {', '.join(currency_values[:5])}"
    report_metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "survey_year": ", ".join(survey_year_values) if survey_year_values else "확인되지 않음",
        "salary_column": "ConvertedCompYearly",
        "currency_label": currency_label,
        "total_loaded_rows": stats["total_loaded_rows"],
        "valid_model_rows": stats["valid_model_rows"],
    }
    result_payload = {
        "input_path": resolved_path,
        "stats": stats,
        "high_salary_threshold": high_salary_threshold,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_target_distribution": train_target_dist,
        "test_target_distribution": test_target_dist,
        "categorical_features": categorical_features,
        "country_included": country_included,
        "model_results": model_results,
        "report_metadata": report_metadata,
        "language_statistics": language_stats_df.to_dict(orient="records"),
        "language_evaluation": language_eval_df.to_dict(orient="records"),
        "language_prediction_summary": language_prediction_summary_df.to_dict(orient="records"),
        "top_positive_features": coeff_df[coeff_df["direction"] == "positive"].head(10).to_dict(orient="records"),
        "top_negative_features": coeff_df[coeff_df["direction"] == "negative"].head(10).to_dict(orient="records"),
        "positive_language_features": lang_coeff_df[lang_coeff_df["direction"] == "positive"].head(10).to_dict(orient="records"),
        "negative_language_features": lang_coeff_df[lang_coeff_df["direction"] == "negative"].head(10).to_dict(orient="records"),
        "model_reload_verified": reload_verified,
        "key_findings": [
            "언어 사용 여부와 일반 특성 모두 예측에 기여했다.",
            "full_model 성능은 비교 모델과 함께 model_comparison.csv에 기록되었다.",
        ],
        "limitations": [
            "Stack Overflow 설문은 무작위 표본이 아니다.",
            "언어 사용 여부와 연봉 사이의 인과관계를 의미하지 않는다.",
        ],
    }

    with (DOMAIN_ROOT / "model_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, ensure_ascii=False, indent=2)
    with (DOMAIN_ROOT / "model_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata_payload, fh, ensure_ascii=False, indent=2)
    output_files = [
        str(DOMAIN_ROOT / "model_comparison.csv"),
        str(DOMAIN_ROOT / "classification_report.csv"),
        str(DOMAIN_ROOT / "confusion_matrix.csv"),
        str(DOMAIN_ROOT / "confusion_matrix.png"),
        str(DOMAIN_ROOT / "feature_coefficients.csv"),
        str(DOMAIN_ROOT / "language_coefficients.csv"),
        str(DOMAIN_ROOT / "language_statistics.csv"),
        str(DOMAIN_ROOT / "language_evaluation.csv"),
        str(DOMAIN_ROOT / "language_predictions.csv"),
        str(DOMAIN_ROOT / "language_prediction_summary.csv"),
        str(DOMAIN_ROOT / "prediction_samples.csv"),
        str(DOMAIN_ROOT / "model.joblib"),
        str(DOMAIN_ROOT / "model_metrics.json"),
        str(DOMAIN_ROOT / "model_metadata.json"),
        str(DOMAIN_ROOT / "language_salary_report.html"),
        str(DOMAIN_ROOT / "result.md"),
        str(DOMAIN_ROOT / "result.html"),
    ]

    with (DOMAIN_ROOT / "result.json").open("w", encoding="utf-8") as fh:
        json.dump(create_result_json(result_payload, output_files), fh, ensure_ascii=False, indent=2)
    markdown_text = create_language_salary_markdown(result_payload, language_stats_df, language_eval_df)
    (DOMAIN_ROOT / "result.md").write_text(markdown_text, encoding="utf-8")
    salary_report_path = create_language_salary_html_report(
        language_stats_df,
        model_df,
        report_metadata,
        DOMAIN_ROOT / "language_salary_report.html",
        language_eval_df,
        language_prediction_summary_df,
        model_results,
    )
    (DOMAIN_ROOT / "result.html").write_text(salary_report_path.read_text(encoding="utf-8"), encoding="utf-8")
    (DOMAIN_ROOT / "README.md").write_text(create_salary_readme_content(result_payload), encoding="utf-8")

    print_status("============================================================")
    print_status("Domain 6 고연봉 예측 완료")
    print_status("============================================================")
    print_status(f"입력 파일: {resolved_path}")
    print_status(f"입력 파일 크기: {resolved_path.stat().st_size}")
    print_status(f"전체 로딩 행 수: {stats['total_loaded_rows']}")
    print_status(f"연봉 결측 제외: {stats['salary_missing_excluded']}")
    print_status(f"연봉 0 이하 제외: {stats['salary_nonpositive_excluded']}")
    print_status(f"언어 결측 제외: {stats['language_missing_excluded']}")
    print_status(f"중복 제외: {stats['duplicate_excluded']}")
    print_status(f"최종 모델 행 수: {stats['valid_model_rows']}")
    print_status(f"학습 행 수: {len(train_df)}")
    print_status(f"테스트 행 수: {len(test_df)}")
    print_status(f"고연봉 기준 분위수: {HIGH_SALARY_QUANTILE}")
    print_status(f"고연봉 임계 연봉: {high_salary_threshold:.2f}")
    print_status(f"학습 고연봉 비율: {y_train.mean():.3f}")
    print_status(f"테스트 고연봉 비율: {y_test.mean():.3f}")
    print_status(f"수치형 특성 수: {len(NUMERIC_FEATURES)}")
    print_status(f"범주형 특성 수: {len(categorical_features)}")
    print_status(f"언어 특성 수: {len(LANGUAGE_FEATURES)}")
    print_status(f"Country 포함 여부: {country_included}")
    print_status("")
    print_status("DummyClassifier")
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
        print_status(f"- {key}: {dummy_metrics[key]:.3f}")
    print_status("")
    print_status("Language-only LogisticRegression")
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
        print_status(f"- {key}: {language_only_metrics[key]:.3f}")
    print_status("")
    print_status("Profile-only LogisticRegression")
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
        print_status(f"- {key}: {profile_only_metrics[key]:.3f}")
    print_status("")
    print_status("Full LogisticRegression")
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
        print_status(f"- {key}: {full_model_metrics[key]:.3f}")
    print_status("")
    print_status(f"Profile-only 대비 Full F1 변화: {performance_differences['profile_only_vs_full_model_f1']:.3f}")
    print_status(f"Profile-only 대비 Full ROC-AUC 변화: {performance_differences['profile_only_vs_full_model_roc_auc']:.3f}")
    print_status(f"Profile-only 대비 Full Average Precision 변화: {performance_differences['profile_only_vs_full_model_average_precision']:.3f}")
    print_status("")
    print_status(f"모델 저장: {model_path}")
    print_status(f"모델 재로드 검증: {reload_verified}")
    print_status(f"결과 JSON: {DOMAIN_ROOT / 'result.json'}")
    print_status(f"결과 Markdown: {DOMAIN_ROOT / 'result.md'}")
    print_status("============================================================")
    print_status("언어별 연봉 HTML 보고서 생성 완료")
    print_status("============================================================")
    print_status(f"보고서 경로: {salary_report_path}")
    print_status(f"분석 언어 수: {len(language_stats_df)}")
    print_status(f"분석 응답자 수: {len(model_df)}")
    print_status("생성된 메인 차트 수: 11")
    print_status(f"부록 차트 수: {4 if not language_eval_df.empty else 0}")
    print_status(f"보고서 파일 크기: {salary_report_path.stat().st_size / 1024:.1f} KB")
    print_status("핵심 분석 주제: 프로그래밍 언어 사용자 집단별 연봉 수준 및 분포 비교")
    return result_payload


def main() -> None:
    """스크립트 진입점."""
    try:
        run_ml_analysis()
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"[오류][최상위] {exc}")


if __name__ == "__main__":
    main()
