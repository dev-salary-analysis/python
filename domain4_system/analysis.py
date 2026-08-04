"""
Domain4 - System Programming Language Analysis

목적
- 시스템 프로그래밍 언어(C, C++, Rust) 사용자의 특성을 분석한다.

분석 내용
1. 언어별 사용자 수
2. 언어별 연봉 중앙값(Median Salary)
3. 언어별 평균 및 중앙 경력(YearsCodePro)
4. 개발자 직무(DevType) 분포 비교
5. C++와 Rust 사용자 연봉 차이 t-test
6. 언어별 연봉 박스플롯 생성

입력 데이터
- data/results.csv

출력 파일
- summary.csv
- salary_boxplot.png
- result.md

작성자 : 문관영
작성일 : 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# 데이터 불러오기
df = pd.read_csv("data/results.csv")

# 결측치 제거
df = df.dropna(subset=["ConvertedCompYearly", "YearsCodePro"])

# 언어별 사용자 추출
c = df[df["lang_C"] == 1]
cpp = df[df["lang_C++"] == 1]
rust = df[df["lang_Rust"] == 1]

# 사용자 수 집계 및 확인
user_count = {
    "C": len(c),
    "C++": len(cpp),
    "Rust": len(rust)
}

print(user_count)

# 경력 평균 및 중앙값
career = pd.DataFrame({
    "Language": ["C", "C++", "Rust"],
    "Mean": [
        c["YearsCodePro"].mean(),
        cpp["YearsCodePro"].mean(),
        rust["YearsCodePro"].mean()
    ],
    "Median": [
        c["YearsCodePro"].median(),
        cpp["YearsCodePro"].median(),
        rust["YearsCodePro"].median()
    ]
})
print(career)

# 개발자 직무 비교
devtype = pd.concat([
    c.assign(Language="C"),
    cpp.assign(Language="C++"),
    rust.assign(Language="Rust")
])

dev_summary = (
    devtype.groupby("Language")["DevType"]
    .value_counts()
)

print(dev_summary)

# t-test
# C++와 Rust의 연봉 차이가 통계적으로 유의한지 확인
t, p = ttest_ind(
    cpp["ConvertedCompYearly"],
    rust["ConvertedCompYearly"],
    equal_var=False,
    nan_policy="omit"
)

print(t)
print(p)

if p < 0.05:
    print("유의한 차이가 있음")
else:
    print("유의한 차이가 없음")

# 박스플롯
plt.figure(figsize=(8,6))

plt.boxplot(
    [
        c["ConvertedCompYearly"],
        cpp["ConvertedCompYearly"],
        rust["ConvertedCompYearly"]
    ],
    tick_labels=["C", "C++", "Rust"]
)

plt.ylabel("Salary")
plt.title("Salary Distribution")

plt.savefig("domain4_system/salary_boxplot.png")

# summary.csv 저장
summary = pd.DataFrame({
    "Language":["C","C++","Rust"],
    "Users":[len(c),len(cpp),len(rust)],
    "MedianSalary":[
        c["ConvertedCompYearly"].median(),
        cpp["ConvertedCompYearly"].median(),
        rust["ConvertedCompYearly"].median()
    ],
    "MeanExperience":[
        c["YearsCodePro"].mean(),
        cpp["YearsCodePro"].mean(),
        rust["YearsCodePro"].mean()
    ],
    "MedianExperience":[
        c["YearsCodePro"].median(),
        cpp["YearsCodePro"].median(),
        rust["YearsCodePro"].median()
    ]
})

summary.to_csv("domain4_system/summary.csv", index=False)

