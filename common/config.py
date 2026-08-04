"""Single source of truth for every analysis domain."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

SALARY_COLUMN = "ConvertedCompYearly"
LANGUAGE_COLUMN = "LanguageHaveWorkedWith"
EXPERIENCE_COLUMN = "YearsCodePro"

RANDOM_STATE = 42
MIN_SAMPLE_SIZE = 100

# None이면 모든 국가를 포함합니다. 예: ("South Korea",) 또는
# ("United States of America", "Canada")처럼 팀 전체가 같은 값으로 바꾸세요.
COUNTRIES: tuple[str, ...] | None = None

# 국가 필터 적용 후 연봉의 하위/상위 1%를 제거합니다.
SALARY_LOWER_QUANTILE = 0.01
SALARY_UPPER_QUANTILE = 0.99

LANGUAGE_DOMAINS = {
    "domain1_dynamic": ("Python", "R", "Julia"),
    "domain2_jvm": ("Java", "Kotlin", "Scala"),
    "domain3_web": ("JavaScript", "TypeScript", "PHP"),
    "domain4_system": ("C", "C++", "Rust"),
    "domain5_platform": ("C#", "Go", "Swift"),
}

ALL_LANGUAGES = tuple(
    language
    for languages in LANGUAGE_DOMAINS.values()
    for language in languages
)

# GitHub 저장소의 대용량 파일(LFS)을 실제 CSV로 제공하는 공식 주소입니다.
OFFICIAL_DATA_URLS = {
    2024: (
        "https://media.githubusercontent.com/media/StackExchange/Survey/"
        "refs/heads/main/packages/archive/2024/results.csv"
    ),
    2025: (
        "https://media.githubusercontent.com/media/StackExchange/Survey/"
        "refs/heads/main/packages/archive/2025/results.csv"
    ),
}
