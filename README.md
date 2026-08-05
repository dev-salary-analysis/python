# Stack Overflow 개발자 연봉 분석

프로그래밍 언어 사용 경험에 따른 개발자 연봉 차이를 분석하고, 전체 응답자의
언어·경력·직무·학력·근무 형태·기업 규모로 고연봉 여부를 예측하는 팀 프로젝트입니다.

## 프로젝트 구조

여섯 도메인은 모두 중앙에서 한 번 정제한 `data/results.csv`를 직접 읽습니다.
Domain 1~5의 분석 결과가 Domain 6의 입력으로 들어가지는 않습니다. 각 도메인의
결과는 마지막에 `report.md`에서만 합칩니다.

```text
data/results.csv
├── domain1_dynamic      Python, R, Julia
├── domain2_jvm          Java, Kotlin, Scala
├── domain3_web          JavaScript, TypeScript, PHP
├── domain4_system       C, C++, Rust
├── domain5_platform     C#, Go, Swift
└── domain6_ml           전체 언어·경력·직무 기반 고연봉 예측
            ↓
        report.md
```

## 빠른 실행

저장소 루트에서 다음 명령을 실행하면 여섯 도메인, 통합 Markdown 보고서와
실행 로그를 순서대로 다시 생성합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
```

주요 산출물은 다음과 같습니다.

- `report.md`: 여섯 도메인의 자동 통합 보고서
- `output/evidence/execution_log.txt`: 단계별 명령·종료 코드·실행 출력
- `domain6_ml/high_salary_model.joblib`: 전처리와 모델을 묶은 학습 Pipeline

모델 학습 시간이 부담되면 기존 Domain 6 산출물을 유지한 채 나머지만 다시
생성할 수 있습니다.

```bash
python scripts/run_all.py --skip-ml
```

## 공통 데이터 생성

공식 Stack Overflow Developer Survey CSV를 한 번만 전처리해 다음 파일을 만듭니다.

```text
data/results.csv
data/preprocessing_metadata.json
```

```bash
python scripts/build_dataset.py --year 2025
```

이미 내려받은 원본을 사용할 때는 네트워크 없이 실행할 수 있습니다.

```bash
python scripts/build_dataset.py --year 2025 --source /경로/survey_results_public.csv
```

현재 팀 결과는 **2025 설문으로 통일**되어 있습니다. 과제 안내서에는 2024 설문
링크가 기재되어 있으므로, 연도 자체가 채점 조건인지 제출 전에 강사에게 확인하세요.
2024로 바꾸려면 팀 전체가 같은 명령으로 데이터를 다시 만들고 모든 도메인을
재실행해야 합니다.

```bash
python scripts/build_dataset.py --year 2024
python scripts/run_all.py
```

## 공통 전처리 원칙

- `ResponseId` 중복 제거
- 연봉·언어 필수값 결측 제거 및 연봉 0 이하 제거
- 동일한 국가 범위 적용(기본값: 전 세계)
- 국가 필터 적용 후 연봉 하위 1%, 상위 1% 제거
- 전문 경력을 숫자로 변환(`Less than 1 year` → `0.5`)
- 언어 문자열을 `;`로 정확히 분리해 Java와 JavaScript를 구분
- 언어별 사용 여부를 0/1 특성으로 생성
- 표본 100명 미만인 언어를 언어별 비교에서 제외

2025 원본에는 Julia 응답이 없어 표본 기준에 따라 Domain 1 비교에서 제외됩니다.

## 개별 도메인 실행

```bash
python domain1_dynamic/analysis.py
python domain2_jvm/analysis.py
python domain3_web/domain3_analysis.py
python domain4_system/analysis.py
python domain5_platform/analysis.py
python -m domain6_ml.model
python scripts/generate_report.py
```

Domain 2는 Pandas와 Polars가 같은 행·열·연봉 요약값을 읽는지 비교하며,
Seaborn 정적 차트와 Pearson 상관계수, Welch t-test를 생성합니다. Domain 3은
Plotly 인터랙티브 HTML을 생성합니다.

Domain 6은 공통 CSV 전체를 train/test로 먼저 분리한 뒤 **학습 데이터의 연봉
중앙값 이상**을 `HighSalary=1`로 정의합니다. 수치형·범주형 전처리와
`LogisticRegression`을 하나의 `Pipeline`으로 묶고 Accuracy, Precision, Recall,
F1-score, ROC-AUC, 혼동행렬을 출력한 뒤 모델을 저장합니다.

## 테스트

```bash
python -m pytest -q
```

테스트는 중복 처리, 언어 정확 분리, 공통 CSV 로딩, Pandas·Polars 결과 일치,
통계 검정 표본 구성, 목표값 정의와 모델 데이터 누수 방지를 확인합니다.

## 제출 보고서

`report.md`는 분석 실행 결과를 반영해 자동 생성됩니다. 저장소 상위의
`integrated_report.html`은 제출용 정적 원본으로 별도 관리하며,
`scripts/run_all.py`가 생성하거나 덮어쓰지 않습니다. 과제 안내서의 PDF 제출
요구에 맞춘 최종본은 최상위 폴더의 `판교_3반_1조_최종레포트.pdf`입니다.

제출 전에는 `python scripts/run_all.py`를 실행하고
`output/evidence/execution_log.txt`에서 Domain 1~6, 보고서 생성, 자동 테스트의
종료 코드를 확인합니다. 인터랙티브 Plotly 산출물은 PDF에 포함될 수 없으므로
`domain3_web/salary_interactive.html`을 포함한 전체 코드 폴더를 함께 제출합니다.

한 응답자는 여러 언어를 선택할 수 있으므로 도메인 표본 수를 단순 합산하면 전체
응답자 수와 같지 않습니다. 관찰된 연봉 차이는 인과관계를 의미하지 않으며 국가,
경력, 직무, 조직 규모 같은 교란요인을 함께 고려해야 합니다.
