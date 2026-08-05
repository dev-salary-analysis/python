# Domain 6 — 전체 언어 및 개발자 특성 기반 고연봉 예측

## 1. 분석 목적

프로그래밍 언어 사용 여부와 경력·직무·학력·근무 형태·기업 규모로 연봉 중앙값 이상 여부를 예측한다.

## 2. 대상 범위

- 목표 변수: 학습 데이터 연봉 중앙값($76,340.50) 이상이면 `HighSalary=1`
- 수치형 특성: YearsCodeProNumeric, LanguageCount
- 범주형 특성: RemoteWork, DevType, EdLevel, Employment, OrgSize, Country
- 언어 특성: 15개 사용 여부

## 3. 데이터 전처리

- 공통 입력 `data/results.csv`를 직접 사용하며 Domain 1~5 산출물은 입력으로 사용하지 않는다.
- train/test를 먼저 분리하고 학습 데이터에서만 중앙값 임계값을 계산해 테스트 정보 누수를 방지한다.
- 수치형 중앙값 대치·표준화, 범주형 최빈값 대치·One-Hot Encoding을 ColumnTransformer로 구성한다.
- ResponseId, 연봉, 원본 언어 문자열과 목표 변수는 입력 특성에서 제외한다.

## 4. 기술통계 및 모델 구조

- 전체/학습/테스트 표본: 21,678 / 17,342 / 4,336
- 학습 클래스 분포: {0: 8671, 1: 8671}
- 테스트 클래스 분포: {0: 2202, 1: 2134}
- 비교 모델: Dummy, 언어 전용, 프로필 전용, 전체 LogisticRegression
- 최종 모델: 전처리와 LogisticRegression(class_weight='balanced')을 묶은 sklearn Pipeline
- 표본 30명 이상 언어 중 중앙 연봉 최고: Scala ($95,674)

## 5. 시각화 결과

- 혼동행렬: [confusion_matrix.png](confusion_matrix.png)
- 전체 언어 인터랙티브 분석: [language_salary_report.html](language_salary_report.html)
- 상세 ML 보고서: [result.html](result.html)

## 6. 모델 평가

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.508 | 0.000 | 0.000 | 0.000 | 0.500 |
| 언어 전용 | 0.563 | 0.550 | 0.624 | 0.584 | 0.595 |
| 프로필 전용 | 0.804 | 0.796 | 0.808 | 0.802 | 0.885 |
| 전체 모델 | 0.810 | 0.800 | 0.821 | 0.810 | 0.890 |

- 최종 모델 Average Precision: 0.878
- 언어 하위집합 중 최고 F1: Swift (0.856)
- 저장 모델: [high_salary_model.joblib](high_salary_model.joblib)
- 저장 모델 재로드 검증: 성공

## 7. 결과 해석

1. 전체 모델은 언어 전용 모델보다 F1과 ROC-AUC가 높아 경력·직무 등 프로필 특성이 중요한 예측 정보를 제공한다.
2. 프로필 전용 모델과 전체 모델의 성능 차이는 작아 언어 정보의 추가 기여는 제한적이다.
3. 로지스틱 회귀 계수는 연봉 증가액이나 인과 효과가 아니라 다른 변수를 함께 고려한 예측 연관 방향이다.

### 계수가 큰 양의 특성

- cat__Country_Switzerland: 4.152
- cat__Country_United States of America: 3.415
- cat__Country_Denmark: 2.967
- cat__Country_Israel: 2.435
- cat__Country_Norway: 2.337

### 양의 방향 언어 특성

- Scala: 0.556
- TypeScript: 0.399
- Go: 0.373

## 8. 분석 한계

- Stack Overflow 자기보고 설문은 무작위 표본이 아니며 국가별 임금·환율·생활비 차이를 충분히 통제하지 못했다.
- 단일 train/test 분할 결과이므로 교차검증과 외부 연도 검증이 필요하며, 채용·연봉 책정에 직접 사용해서는 안 된다.