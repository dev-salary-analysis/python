# Programming Language Salary Analysis

Stack Overflow 개발자 설문 데이터를 이용한 프로그래밍 언어별 개발자 연봉 분석 프로젝트입니다.

## 핵심 분석 주제

언어 사용자 집단별 사용자 수, 평균 연봉, 중앙 연봉, 분포, 사분위 범위와 실제 고연봉 비율을 비교합니다.
연봉 상위 25% 분류 모델은 언어별 연봉 분석을 보조하는 부록으로만 사용합니다.

## 데이터

- 유효 분석 응답자: 21,678명
- 연봉 컬럼: ConvertedCompYearly
- 통화 단위: 원본 데이터에 통화 단위 컬럼이 없어 확인되지 않음
- 데이터 기준 연도: 2025

## 실행

```bash
cd /Users/baekjiheon/Desktop/SKALA/Day15_Python_2/domain_6ml
./domain6_ml/.venv/bin/python -m domain6_ml.model
```

## 주요 산출물

- [language_salary_report.html](language_salary_report.html): 독립형 Plotly 인터랙티브 메인 보고서
- [result.html](result.html): 메인 보고서 호환 링크
- [result.md](result.md): Markdown 요약
- [language_statistics.csv](language_statistics.csv): 언어별 연봉·분포·고연봉 통계
- [language_evaluation.csv](language_evaluation.csv): 언어 사용자 그룹별 분류 모델 보조 진단
- [language_prediction_summary.csv](language_prediction_summary.csv): 언어별 실제·예측 고연봉 요약

## 주의사항

한 응답자는 여러 언어를 선택할 수 있으며, 관찰된 언어별 연봉 차이는 인과관계를 의미하지 않습니다.
국가, 경력, 직무, 회사 규모 등 교란요인을 함께 고려해야 합니다.