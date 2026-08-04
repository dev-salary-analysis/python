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

# result.md 파일 작성
# result.md 생성
with open("domain4_system/result.md", "w", encoding="utf-8") as f:
    f.write("# 시스템 프로그래밍 언어 분석 결과\n\n")

    f.write("## 1. 언어별 사용자 수\n")
    f.write(f"- C : {len(c)}명\n")
    f.write(f"- C++ : {len(cpp)}명\n")
    f.write(f"- Rust : {len(rust)}명\n\n")

    f.write("## 2. 언어별 연봉 중앙값\n")
    f.write(f"- C : {c['ConvertedCompYearly'].median():,.0f}\n")
    f.write(f"- C++ : {cpp['ConvertedCompYearly'].median():,.0f}\n")
    f.write(f"- Rust : {rust['ConvertedCompYearly'].median():,.0f}\n\n")

    f.write("## 3. 경력 비교\n")
    for _, row in summary.iterrows():
        f.write(
            f"- {row['Language']} : "
            f"평균 경력 {row['MeanExperience']:.2f}년, "
            f"중앙 경력 {row['MedianExperience']:.2f}년\n"
        )
    f.write("\n")

    f.write("## 4. 개발자 직무 비교\n")
    for (lang, devtype), count in dev_summary.items():
        f.write(f"- {lang} | {devtype} : {count}명\n")
    f.write("\n")

    f.write("## 5. C++와 Rust 연봉 t-test\n")
    f.write(f"- t-value : {t:.3f}\n")
    f.write(f"- p-value : {p:.5f}\n\n")

    if p < 0.05:
        f.write("결과 : p < 0.05 이므로 C++와 Rust 사용자의 평균 연봉에는 통계적으로 유의한 차이가 있다.\n\n")
    else:
        f.write("결과 : p ≥ 0.05 이므로 C++와 Rust 사용자의 평균 연봉에는 통계적으로 유의한 차이가 없다.\n\n")

    f.write("## 6. 연봉 박스플롯\n")
    f.write("언어별 연봉 분포는 salary_boxplot.png 파일을 통해 확인할 수 있다.\n\n")

    f.write("---\n")
    f.write("분석 결과는 summary.csv, salary_boxplot.png, result.md 파일로 저장되었다.\n")