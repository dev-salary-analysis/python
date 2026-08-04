# Stack Overflow 연봉 분석용 데이터 생성

공식 Stack Overflow Developer Survey CSV를 내려받아 한 번만 전처리하고,
6개 도메인이 **하나의 공통 데이터**를 사용하도록 만드는 코드입니다.

## 생성되는 파일

```text
data/results.csv                  # 전 팀원이 공유하는 공통 정제 데이터
data/preprocessing_metadata.json # 행 수, 이상치 경계, 중앙값, 언어별 표본 수
```

도메인별 CSV를 기본으로 만들지 않으므로 파일마다 전처리 결과가 달라지는 문제를
막을 수 있습니다. 한 응답자가 여러 언어를 사용했다면 여러 도메인 분석에 포함될 수 있습니다.
따라서 도메인별 결과를 단순 합산하면 전체 응답자 수와 같지 않습니다.

## 실행

저장소에 포함된 `data/results.csv`를 바로 사용할 수 있습니다. 원본에서 데이터를
다시 만들고 싶다면 저장소 루트에서 다음 명령을 실행하세요.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_dataset.py --year 2025
```

Stack Overflow 설문 데이터는 REST API가 아니라 공식 공개 CSV로 배포됩니다.
스크립트가 CSV를 스트리밍 다운로드하고 로컬에 캐시합니다.

2025 설문의 `WorkExp`는 공통 컬럼 `YearsCodePro`로 자동 변환됩니다.
2024 데이터도 같은 코드로 처리할 수 있습니다.

> 주의: 2025 원본에는 `Julia` 응답이 없어 `MIN_SAMPLE_SIZE = 100` 규칙에
> 따라 동적·데이터 언어 분석에서 자동 제외됩니다. Julia 비교가 필수라면
> 팀 전체가 2024 데이터로 통일해 다시 생성한 뒤 표본 수를 확인하세요.

```bash
python scripts/build_dataset.py --year 2024
```

이미 받은 CSV를 사용할 때는 네트워크 호출 없이 다음처럼 실행합니다.

```bash
python scripts/build_dataset.py --year 2025 --source /경로/results.csv
```

## 도메인별로 같은 데이터 읽기

각 팀원의 `analysis.py`에서는 공통 로더를 사용합니다.

```python
from common.data import load_domain

# 팀원 1
df = load_domain("domain1_dynamic")

# 팀원 2라면 "domain2_jvm", 팀원 3이라면 "domain3_web" 사용
print(df.shape)
```

팀원 6은 필터 없이 전체 공통 데이터를 읽습니다.

```python
from common.data import load_all

df = load_all()
X = df.drop(columns=["HighSalary"])
y = df["HighSalary"]
```

부득이하게 팀원별 CSV가 필요할 때만 아래 옵션을 사용합니다.

```bash
python scripts/build_dataset.py --year 2025 --write-domain-files
```

## 국가 범위 통일

기본값은 전 세계(`COUNTRIES = None`)입니다. 특정 국가만 비교하려면 모든 팀원이
같은 명령으로 데이터를 다시 만들면 됩니다.

```bash
python scripts/build_dataset.py --year 2025 \
  --countries "United States of America" Canada
```

반복 실행할 설정이라면 `common/config.py`의 `COUNTRIES`를 변경하세요.

## 적용한 공통 전처리

- 연봉/언어 결측 제거 및 연봉 0 이하 제거
- 국가 필터를 모든 도메인에 동일 적용
- 국가 필터 후 연봉 하위 1%, 상위 1% 제거
- 전문 경력을 숫자로 변환 (`Less than 1 year` → `0.5`)
- 세미콜론으로 언어를 정확히 분리 (`Java`와 `JavaScript` 구분)
- 각 언어 사용 여부를 `lang_Python`, `lang_C++` 같은 0/1 컬럼으로 생성
- 공통 연봉 중앙값 이상이면 `HighSalary = 1`
- 정제 후 표본이 100명 미만인 언어는 도메인 분할 기준에서 제외

분석 재현성을 위해 이상치 경계와 목표 변수 중앙값은
`data/preprocessing_metadata.json`에 저장됩니다.

## 팀원 2: JVM 언어 End-to-End 분석

Java, Kotlin, Scala 사용자의 EDA·통계·시각화를 한 번에
다시 생성하려면 저장소 루트에서 실행합니다.

```bash
python domain2_jvm/analysis.py
```

실행 시 Pandas·Polars 로딩 비교, Seaborn 정적 차트, 상관계수와
Java–Kotlin Welch t-test를 수행합니다.
연봉, 경력, 상관관계, 조직 규모를 위 2개·아래 2개로 배치한
`salary_chart.png` 하나로 통합합니다. ML Pipeline은 팀원 6이 담당합니다.
2025 설문 데이터를 사용하는 것이
이 프로젝트의 팀 결정입니다.
