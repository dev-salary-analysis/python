

"""
Domain 3: 웹 개발 언어 분석
JavaScript, TypeScript, PHP 개발자의 연봉 및 직무 분석
"""
 
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.data import load_domain
from common.config import SALARY_COLUMN, EXPERIENCE_COLUMN, LANGUAGE_DOMAINS
 
OUTPUT_DIR = Path(__file__).parent
DOMAIN_NAME = "domain3_web"
 
TARGET_LANGUAGES = LANGUAGE_DOMAINS[DOMAIN_NAME]
LANGUAGE_COLUMNS = {
    lang: f"lang_{lang}" for lang in TARGET_LANGUAGES
}

EXPERIENCE_BINS = [-float("inf"), 2, 5, 10, 15, 20, float("inf")]
EXPERIENCE_LABELS = ["0–2년", "3–5년", "6–10년", "11–15년", "16–20년", "21년 이상"]


def classify_web_role(value):
    """Map survey role text to the three web roles used in the team plan."""
    if not isinstance(value, str) or not value.strip():
        return "기타·미응답"
    text = value.lower()
    front = "front-end" in text or "frontend" in text
    back = "back-end" in text or "backend" in text
    full = "full-stack" in text or "full stack" in text
    if full or (front and back):
        return "풀스택"
    if front:
        return "프론트엔드"
    if back:
        return "백엔드"
    return "기타·미응답"
 
 
class WebDevelopmentAnalysis:
    
    def __init__(self):
        self.df = None
        self.stats_table = None
        self.role_table = None
        self.experience_salary_table = None
        self.results = {}
 
    def load_and_prepare(self):
        """데이터 로드"""
        print("="*80)
        print("1. 데이터 로딩")
        print("="*80)
        
        self.df = load_domain(DOMAIN_NAME)
        print(f"\n데이터 로드 완료: {len(self.df):,}행")
 
    def descriptive_statistics(self):
        """기술통계 계산"""
        print("\n" + "="*80)
        print("2. 기술통계")
        print("="*80)
        
        stats_list = []
        for lang in TARGET_LANGUAGES:
            lang_col = LANGUAGE_COLUMNS[lang]
            subset = self.df[self.df[lang_col] == 1]
            
            stats_dict = {
                "language": lang,
                "count": len(subset),
                "mean_salary_usd": subset[SALARY_COLUMN].mean(),
                "median_salary_usd": subset[SALARY_COLUMN].median(),
                "std_salary": subset[SALARY_COLUMN].std(),
                "q1_salary": subset[SALARY_COLUMN].quantile(0.25),
                "q3_salary": subset[SALARY_COLUMN].quantile(0.75),
            }
            stats_list.append(stats_dict)
        
        self.stats_table = pd.DataFrame(stats_list)
        for col in ["mean_salary_usd", "median_salary_usd", "std_salary", "q1_salary", "q3_salary"]:
            self.stats_table[col] = self.stats_table[col].round(0)
        
        print("\n" + self.stats_table.to_string(index=False))
        
        print("\n경력별 통계:")
        for lang in TARGET_LANGUAGES:
            lang_col = LANGUAGE_COLUMNS[lang]
            subset = self.df[self.df[lang_col] == 1]
            exp = subset[EXPERIENCE_COLUMN]
            print(f"{lang:15s}: 평균 {exp.mean():5.1f}년, 중앙값 {exp.median():5.1f}년")

        role_rows = []
        experience_rows = []
        for lang in TARGET_LANGUAGES:
            subset = self.df[self.df[LANGUAGE_COLUMNS[lang]] == 1].copy()
            roles = subset["DevType"].map(classify_web_role)
            role_counts = roles.value_counts().reindex(
                ["프론트엔드", "백엔드", "풀스택", "기타·미응답"], fill_value=0
            )
            for role, count in role_counts.items():
                role_rows.append(
                    {
                        "language": lang,
                        "role": role,
                        "count": int(count),
                        "rate_pct": float(count / len(subset) * 100) if len(subset) else 0.0,
                    }
                )

            experience = pd.to_numeric(subset[EXPERIENCE_COLUMN], errors="coerce")
            groups = pd.cut(
                experience,
                bins=EXPERIENCE_BINS,
                labels=EXPERIENCE_LABELS,
                right=True,
            )
            grouped = (
                subset.assign(experience_group=groups)
                .dropna(subset=["experience_group", SALARY_COLUMN])
                .groupby("experience_group", observed=False)[SALARY_COLUMN]
                .agg(["count", "median"])
                .reindex(EXPERIENCE_LABELS)
            )
            for experience_group, row in grouped.iterrows():
                experience_rows.append(
                    {
                        "language": lang,
                        "experience_group": str(experience_group),
                        "count": int(row["count"]) if pd.notna(row["count"]) else 0,
                        "median_salary_usd": float(row["median"]) if pd.notna(row["median"]) else float("nan"),
                    }
                )
        self.role_table = pd.DataFrame(role_rows)
        self.experience_salary_table = pd.DataFrame(experience_rows)
 
    def statistical_test(self):
        """JavaScript vs TypeScript t-test"""
        print("\n" + "="*80)
        print("3. 통계 검정")
        print("="*80)
        
        js_only = self.df[LANGUAGE_COLUMNS["JavaScript"]].eq(1) & self.df[LANGUAGE_COLUMNS["TypeScript"]].eq(0)
        ts_only = self.df[LANGUAGE_COLUMNS["TypeScript"]].eq(1) & self.df[LANGUAGE_COLUMNS["JavaScript"]].eq(0)
        js_subset = self.df.loc[js_only]
        ts_subset = self.df.loc[ts_only]
        
        js_salaries = js_subset[SALARY_COLUMN].values
        ts_salaries = ts_subset[SALARY_COLUMN].values
        
        t_stat, p_value = stats.ttest_ind(js_salaries, ts_salaries, equal_var=False)
        
        print(f"\nJavaScript vs TypeScript")
        print(f"JavaScript: n={len(js_salaries)}, mean=${js_salaries.mean():,.0f}")
        print(f"TypeScript: n={len(ts_salaries)}, mean=${ts_salaries.mean():,.0f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"유의미성: {'Yes (p<0.05)' if p_value < 0.05 else 'No (p>=0.05)'}")
        
        self.results["ttest"] = {
            "t_stat": t_stat,
            "p_value": p_value,
            "js_only_n": len(js_salaries),
            "ts_only_n": len(ts_salaries),
            "js_mean": js_salaries.mean(),
            "ts_mean": ts_salaries.mean(),
        }
 
    def create_chart(self):
        """matplotlib 정적 차트 생성"""
        print("\n" + "="*80)
        print("4. 시각화")
        print("="*80)
        
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.stats_table))
        width = 0.35
        ax.bar(x - width / 2, self.stats_table["mean_salary_usd"], width, label="Mean", color="#2563EB")
        ax.bar(x + width / 2, self.stats_table["median_salary_usd"], width, label="Median", color="#60A5FA")
        ax.set_xticks(x, self.stats_table["language"])
        ax.set_ylabel("Annual salary (USD)")
        ax.set_title("Web Development Languages: Salary Comparison")
        ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(lambda value, _: f"${value / 1000:,.0f}K")
        
        for container in ax.containers:
            ax.bar_label(container, labels=[f"${v / 1000:,.0f}K" for v in container.datavalues], padding=3)
        
        fig.tight_layout()
        
        output_path = OUTPUT_DIR / "salary_comparison.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        print(f"저장: {output_path}")

    def create_interactive_chart(self):
        """Plotly 인터랙티브 연봉·경력·직무 차트를 생성한다."""
        figure = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "언어별 연봉 분포",
                "경력 구간별 중앙 연봉",
                "웹 직무 구성 비율",
            ),
            vertical_spacing=0.1,
        )
        colors = {"JavaScript": "#f7df1e", "TypeScript": "#3178c6", "PHP": "#777bb4"}
        for lang in TARGET_LANGUAGES:
            subset = self.df[self.df[LANGUAGE_COLUMNS[lang]] == 1]
            figure.add_trace(
                go.Box(
                    y=subset[SALARY_COLUMN],
                    name=lang,
                    marker_color=colors[lang],
                    boxpoints=False,
                    legendgroup=lang,
                ),
                row=1,
                col=1,
            )
            exp_rows = self.experience_salary_table[
                self.experience_salary_table["language"].eq(lang)
            ]
            figure.add_trace(
                go.Scatter(
                    x=exp_rows["experience_group"],
                    y=exp_rows["median_salary_usd"],
                    mode="lines+markers",
                    name=lang,
                    marker_color=colors[lang],
                    legendgroup=lang,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            role_rows = self.role_table[self.role_table["language"].eq(lang)]
            figure.add_trace(
                go.Bar(
                    x=role_rows["role"],
                    y=role_rows["rate_pct"],
                    name=lang,
                    marker_color=colors[lang],
                    legendgroup=lang,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        figure.update_yaxes(title_text="연봉 (USD)", row=1, col=1)
        figure.update_yaxes(title_text="중앙 연봉 (USD)", row=2, col=1)
        figure.update_yaxes(title_text="비율 (%)", row=3, col=1)
        figure.update_xaxes(title_text="경력 구간", row=2, col=1)
        figure.update_xaxes(title_text="직무", row=3, col=1)
        figure.update_layout(
            title="웹 개발 언어 사용자 연봉·경력·직무 인터랙티브 분석",
            height=1200,
            barmode="group",
            template="plotly_white",
        )
        output_path = OUTPUT_DIR / "salary_interactive.html"
        figure.write_html(output_path, include_plotlyjs=True, full_html=True)
        print(f"저장: {output_path}")
 
    def save_outputs(self):
        """결과 저장"""
        csv_file = OUTPUT_DIR / "summary.csv"
        self.stats_table.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"저장: {csv_file}")
        self.role_table.to_csv(OUTPUT_DIR / "role_distribution.csv", index=False, encoding="utf-8-sig")
        self.experience_salary_table.to_csv(
            OUTPUT_DIR / "experience_salary.csv", index=False, encoding="utf-8-sig"
        )
        
        md_file = OUTPUT_DIR / "result.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Domain 3: 웹 개발 언어 분석\n\n")

            f.write("## 1. 분석 목적\n")
            f.write("웹 개발 언어(JavaScript, TypeScript, PHP) 사용자의 연봉 수준, 경력 분포, 직무 특성 비교\n\n")

            f.write("## 2. 대상 언어\n")
            f.write("- JavaScript, TypeScript, PHP\n\n")

            f.write("## 3. 데이터 전처리\n")
            f.write("- 공통 정제 데이터 `data/results.csv` 사용\n")
            f.write("- 세미콜론으로 정확히 분리된 언어 0/1 플래그 사용\n")
            f.write("- 경력은 숫자형으로 변환하고 직무는 프론트엔드·백엔드·풀스택·기타로 분류\n")
            f.write(f"- 총 표본: {len(self.df):,}명\n")
            f.write(f"- JavaScript: {(self.df[LANGUAGE_COLUMNS['JavaScript']]==1).sum():,}명\n")
            f.write(f"- TypeScript: {(self.df[LANGUAGE_COLUMNS['TypeScript']]==1).sum():,}명\n")
            f.write(f"- PHP: {(self.df[LANGUAGE_COLUMNS['PHP']]==1).sum():,}명\n\n")

            f.write("## 4. 기술통계\n\n")
            f.write("| 언어 | 표본수 | 평균연봉 | 중앙값 | 표준편차 | Q1 | Q3 |\n")
            f.write("|------|--------|---------|--------|----------|----|----|\n")
            for _, row in self.stats_table.iterrows():
                f.write(f"| {row['language']} | {int(row['count']):,} | ${row['mean_salary_usd']:,.0f} | ${row['median_salary_usd']:,.0f} | ${row['std_salary']:,.0f} | ${row['q1_salary']:,.0f} | ${row['q3_salary']:,.0f} |\n")
            f.write("\n")

            f.write("### 직무 구성\n\n")
            f.write("| 언어 | 직무 | 인원 | 비율 |\n|---|---|---:|---:|\n")
            for _, row in self.role_table.iterrows():
                f.write(f"| {row['language']} | {row['role']} | {int(row['count']):,} | {row['rate_pct']:.1f}% |\n")
            f.write("\n")
            
            f.write("## 5. 시각화 결과\n\n")
            f.write("- 정적 차트: [salary_comparison.png](salary_comparison.png)\n")
            f.write("- 인터랙티브 차트: [salary_interactive.html](salary_interactive.html)\n\n")

            f.write("## 6. 통계 검정\n\n")
            f.write("- 검정: JavaScript 전용군과 TypeScript 전용군의 양측 Welch t-test\n")
            ttest = self.results["ttest"]
            f.write(f"- JavaScript 전용군: n={ttest['js_only_n']:,}, 평균 ${ttest['js_mean']:,.0f}\n")
            f.write(f"- TypeScript 전용군: n={ttest['ts_only_n']:,}, 평균 ${ttest['ts_mean']:,.0f}\n")
            f.write(f"- t-statistic: {ttest['t_stat']:.4f}\n")
            f.write(f"- p-value: {ttest['p_value']:.6g}\n")
            f.write(f"- 해석: 5% 유의수준에서 평균 연봉 차이는 {'통계적으로 유의하다' if ttest['p_value'] < 0.05 else '통계적으로 유의하지 않다'}.\n\n")

            f.write("## 7. 결과 해석\n\n")
            f.write("1. 웹 언어별 연봉은 평균·중앙값·사분위 범위를 함께 비교해야 한다.\n\n")
            f.write("2. 경력 구간별 중앙 연봉 차트로 경력 구성 차이가 언어별 연봉 차이에 미치는 영향을 확인할 수 있다.\n\n")
            f.write("3. 직무 비율과 복수 언어 사용을 함께 고려해야 단순한 언어 효과로 과대해석하지 않을 수 있다.\n\n")

            f.write("## 8. 분석 한계\n\n")
            f.write("- Stack Overflow 설문은 온라인 커뮤니티 기반 표본이므로 모집단 대표성에 한계가 있다.\n")
            f.write("- 대부분의 개발자가 여러 언어를 함께 사용하므로 한 언어의 순수한 인과 효과를 분리하기 어렵다.\n")
        
        print(f"저장: {md_file}")
 
    def run(self):
        """전체 분석 파이프라인 실행"""
        print("\n" + "="*80)
        print("Domain 3: 웹 개발 언어 분석")
        print("="*80)
        
        self.load_and_prepare()
        self.descriptive_statistics()
        self.statistical_test()
        self.create_chart()
        self.create_interactive_chart()
        self.save_outputs()
        
        print("\n" + "="*80)
        print("분석 완료")
        print("="*80)
 
 
if __name__ == "__main__":
    analysis = WebDevelopmentAnalysis()
    analysis.run()
 

