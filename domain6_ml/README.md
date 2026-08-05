# Domain 6 — 전체 언어 및 개발자 특성 기반 고연봉 예측

## 분석 목표
- 여러 프로그래밍 언어 사용 여부와 개발자 특성으로 연봉 중앙값 이상 여부를 예측하는 모델을 구현했습니다.
- 실제 입력 데이터로 학습하고, 모델 성능과 산출물을 저장했습니다.

## 데이터와 타깃
- 입력 파일: `data/results.csv`
- Domain 1~5 산출물은 입력으로 사용하지 않음
- 최종 유효 표본 수: 21678
- 학습/테스트 표본 수: 17342/4336
- 고연봉 기준: 학습 데이터의 ConvertedCompYearly 중앙값 이상
- 고연봉 임계값: 76340.50

## 모델 성능 요약
- Accuracy: 0.810
- Balanced Accuracy: 0.811
- Precision: 0.800
- Recall: 0.821
- F1: 0.810
- ROC-AUC: 0.890
- Average Precision: 0.878

## 실행 방법
```bash
python -m domain6_ml.model
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
- 모델 파일: [high_salary_model.joblib](high_salary_model.joblib)
- 메타데이터: [model_metrics.json](model_metrics.json) / [model_metadata.json](model_metadata.json)

## 참고
- 이 프로젝트는 교육·연구용 분석 목적입니다.
- 결과 보고서를 열어보려면 브라우저에서 [result.html](result.html)를 열면 됩니다.
