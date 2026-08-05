import json
from pathlib import Path

import pandas as pd

from domain6_ml.model import (
    LANGUAGE_FEATURES,
    create_high_salary_targets,
    create_language_coefficient_output,
    create_language_evaluation_output,
    create_language_prediction_outputs,
    create_language_statistics,
    create_model_inputs,
    identify_feature_columns,
    load_model_data,
    split_train_test_data,
    validate_model_data,
)


def test_validate_model_data_detects_missing_columns():
    df = pd.DataFrame({
        "ResponseId": [1, 2],
        "ConvertedCompYearly": [1000, 2000],
        "YearsCodeProNumeric": [1, 2],
        "RemoteWork": ["Remote", "Hybrid"],
        "DevType": ["A", "B"],
        "EdLevel": ["C", "D"],
        "Employment": ["E", "F"],
        "OrgSize": ["G", "H"],
        "LanguageCount": [1, 2],
        "lang_python": [1, 0],
        "lang_r": [0, 1],
    })
    df = df[[c for c in df.columns if c != "lang_r"]]

    try:
        validate_model_data(df)
    except ValueError as exc:
        assert "누락된 필수 컬럼" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing columns")


def test_validate_model_data_detects_empty_dataframe():
    empty_df = pd.DataFrame(columns=["ResponseId"])
    try:
        validate_model_data(empty_df)
    except ValueError as exc:
        assert "비어 있는 데이터프레임" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty dataframe")


def test_validate_model_data_detects_language_values_outside_binary():
    df = pd.DataFrame({
        "ResponseId": [1, 2],
        "ConvertedCompYearly": [1000, 2000],
        "YearsCodeProNumeric": [1, 2],
        "RemoteWork": ["Remote", "Hybrid"],
        "DevType": ["A", "B"],
        "EdLevel": ["C", "D"],
        "Employment": ["E", "F"],
        "OrgSize": ["G", "H"],
        "LanguageCount": [1, 2],
        "lang_python": [1, 2],
        "lang_r": [0, 1],
    })
    try:
        validate_model_data(df)
    except ValueError as exc:
        assert "0과 1 이외" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid language values")


def test_target_creation_uses_train_threshold():
    df = pd.DataFrame({
        "ResponseId": [1, 2, 3, 4],
        "ConvertedCompYearly": [100, 200, 300, 400],
        "YearsCodeProNumeric": [1, 2, 3, 4],
        "RemoteWork": ["Remote", "Hybrid", "Remote", "Hybrid"],
        "DevType": ["A", "B", "A", "B"],
        "EdLevel": ["C", "D", "C", "D"],
        "Employment": ["E", "F", "E", "F"],
        "OrgSize": ["G", "H", "G", "H"],
        "LanguageCount": [1, 2, 1, 2],
        "lang_python": [1, 0, 1, 0],
        "lang_r": [0, 1, 0, 1],
    })
    train_df, test_df = split_train_test_data(df, random_state=42, test_size=0.5)
    threshold = train_df["ConvertedCompYearly"].quantile(0.75)
    y_train = create_high_salary_targets(train_df, threshold)
    y_test = create_high_salary_targets(test_df, threshold)
    assert len(y_train) == len(train_df)
    assert len(y_test) == len(test_df)
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_feature_columns_exclude_leakage_and_target_columns():
    df = pd.DataFrame({
        "ResponseId": [1, 2],
        "ConvertedCompYearly": [1000, 2000],
        "LogSalary": [6.9, 7.6],
        "LanguageHaveWorkedWith": ["Python", "Java"],
        "HighSalaryTop25": [0, 1],
        "Country": ["US", "CA"],
        "YearsCodeProNumeric": [1, 2],
        "RemoteWork": ["Remote", "Hybrid"],
        "DevType": ["A", "B"],
        "EdLevel": ["C", "D"],
        "Employment": ["E", "F"],
        "OrgSize": ["G", "H"],
        "LanguageCount": [1, 2],
        "lang_python": [1, 0],
        "lang_r": [0, 1],
    })
    features = identify_feature_columns(df)
    assert "ConvertedCompYearly" not in features
    assert "LogSalary" not in features
    assert "ResponseId" not in features
    assert "Country" not in features


def test_create_model_inputs_has_expected_feature_shape():
    df = pd.DataFrame({
        "ResponseId": [1, 2, 3, 4],
        "ConvertedCompYearly": [100, 200, 300, 400],
        "YearsCodeProNumeric": [1, 2, 3, 4],
        "RemoteWork": ["Remote", "Hybrid", "Remote", "Hybrid"],
        "DevType": ["A", "B", "A", "B"],
        "EdLevel": ["C", "D", "C", "D"],
        "Employment": ["E", "F", "E", "F"],
        "OrgSize": ["G", "H", "G", "H"],
        "LanguageCount": [1, 2, 1, 2],
        "lang_python": [1, 0, 1, 0],
        "lang_r": [0, 1, 0, 1],
    })
    features = create_model_inputs(df)
    assert features.shape[0] == len(df)
    assert features.shape[1] > 0


def test_load_model_data_uses_available_file():
    path = load_model_data()
    assert path.exists()
    assert path.name.endswith(".csv")


def test_language_coefficient_output_restores_language_name_from_transformer_prefix():
    coeff_df = pd.DataFrame({
        "feature": ["lang__lang_python", "lang__lang_php"],
        "coefficient": [0.2, -0.3],
        "abs_coefficient": [0.2, 0.3],
        "direction": ["positive", "negative"],
    })
    output = create_language_coefficient_output(coeff_df)
    assert set(output["language"]) == {"Python", "PHP"}


def test_language_statistics_and_prediction_outputs_are_language_specific():
    language_values = {feature: [0, 0] for feature in LANGUAGE_FEATURES}
    language_values["lang_python"] = [1, 0]
    model_df = pd.DataFrame({
        "ResponseId": [1, 2],
        "ConvertedCompYearly": [100000, 150000],
        **language_values,
    })
    stats = create_language_statistics(model_df, 120000)
    python_row = stats.loc[stats["language"].eq("Python")].iloc[0]
    assert python_row["user_count"] == 1
    assert python_row["high_salary_rate"] == 0
    assert python_row["average_salary"] == 100000
    assert python_row["q1_salary"] == 100000
    assert python_row["q3_salary"] == 100000
    assert python_row["iqr_salary"] == 0
    assert python_row["sample_warning"] == "표본 부족"

    X_test = model_df.drop(columns=["ResponseId", "ConvertedCompYearly"]).copy()
    y_test = pd.Series([0, 1], index=X_test.index)
    detail, summary = create_language_prediction_outputs(
        model_df,
        X_test,
        y_test,
        predictions=[0, 1],
        probabilities=[0.2, 0.8],
    )
    assert len(detail) == 1
    assert detail.iloc[0]["language"] == "Python"
    python_summary = summary.loc[summary["language"].eq("Python")].iloc[0]
    assert python_summary["test_user_count"] == 1
    assert python_summary["actual_high_salary_count"] == 0
    assert python_summary["predicted_high_salary_count"] == 0

    evaluation = create_language_evaluation_output(
        X_test,
        y_test,
        predictions=[0, 1],
        probabilities=[0.2, 0.8],
    )
    python_evaluation = evaluation.loc[evaluation["language"].eq("Python")].iloc[0]
    assert python_evaluation["actual_high_salary_count"] == 0
    assert python_evaluation["predicted_high_salary_count"] == 0
