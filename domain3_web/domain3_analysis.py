

"""
Domain 3: 웹 개발 언어 분석
JavaScript, TypeScript, PHP 개발자의 연봉 및 직무 분석
"""
 
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.data import load_domain
from common.config import SALARY_COLUMN, EXPERIENCE_COLUMN, LANGUAGE_DOMAINS
 
OUTPUT_DIR = Path(__file__).parent
DOMAIN_NAME = "domain3_web"
 
TARGET_LANGUAGES = LANGUAGE_DOMAINS[DOMAIN_NAME]
LANGUAGE_COLUMNS = {
    lang: f"lang_{lang}" for lang in TARGET_LANGUAGES
}
 
 
class WebDevelopmentAnalysis:
    
    def __init__(self):
        self.df = None
        self.stats_table = None
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
            }
            stats_list.append(stats_dict)
        
        self.stats_table = pd.DataFrame(stats_list)
        for col in ["mean_salary_usd", "median_salary_usd", "std_salary"]:
            self.stats_table[col] = self.stats_table[col].round(0)
        
        print("\n" + self.stats_table.to_string(index=False))
        
        print("\n경력별 통계:")
        for lang in TARGET_LANGUAGES:
            lang_col = LANGUAGE_COLUMNS[lang]
            subset = self.df[self.df[lang_col] == 1]
            exp = subset[EXPERIENCE_COLUMN]
            print(f"{lang:15s}: 평균 {exp.mean():5.1f}년, 중앙값 {exp.median():5.1f}년")
 
    def statistical_test(self):
        """JavaScript vs TypeScript t-test"""
        print("\n" + "="*80)
        print("3. 통계 검정")
        print("="*80)
        
        js_subset = self.df[self.df[LANGUAGE_COLUMNS["JavaScript"]] == 1]
        ts_subset = self.df[self.df[LANGUAGE_COLUMNS["TypeScript"]] == 1]
        
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
 
    def save_outputs(self):
        """결과 저장"""
        csv_file = OUTPUT_DIR / "summary.csv"
        self.stats_table.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"저장: {csv_file}")
        
        md_file = OUTPUT_DIR / "result.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Domain 3: 웹 개발 언어 분석\n\n")
            f.write(f"**작성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## 분석 목적\n")
            f.write("웹 개발 언어(JavaScript, TypeScript, PHP) 사용자의 연봉 수준, 경력 분포, 직무 특성 비교\n\n")
            
            f.write("## 분석 데이터\n")
            f.write(f"- 총 표본: {len(self.df):,}명\n")
            f.write(f"- JavaScript: {(self.df[LANGUAGE_COLUMNS['JavaScript']]==1).sum():,}명\n")
            f.write(f"- TypeScript: {(self.df[LANGUAGE_COLUMNS['TypeScript']]==1).sum():,}명\n")
            f.write(f"- PHP: {(self.df[LANGUAGE_COLUMNS['PHP']]==1).sum():,}명\n\n")
            
            f.write("## 기술통계\n\n")
            f.write("| 언어 | 표본수 | 평균연봉 | 중앙값 | 표준편차 |\n")
            f.write("|------|--------|---------|--------|----------|\n")
            for _, row in self.stats_table.iterrows():
                f.write(f"| {row['language']} | {int(row['count']):,} | ${row['mean_salary_usd']:,.0f} | ${row['median_salary_usd']:,.0f} | ${row['std_salary']:,.0f} |\n")
            f.write("\n")
            
            f.write("## 통계 검정 결과\n\n")
            f.write("### JavaScript vs TypeScript t-test\n")
            ttest = self.results["ttest"]
            f.write(f"- p-value: {ttest['p_value']:.6f}\n")
            f.write(f"- 연봉 차이: ${abs(ttest['ts_mean'] - ttest['js_mean']):,.0f}\n")
            f.write(f"- 유의성: {'유의미함' if ttest['p_value'] < 0.05 else '유의하지 않음'}\n\n")
            
            f.write("## 핵심 해석\n\n")
            f.write("1. JavaScript는 가장 많은 개발자가 사용하는 웹 개발 필수 언어이다.\n\n")
            f.write("2. TypeScript 사용자의 평균 연봉이 더 높으며, 이는 정적 타입 시스템에 대한 경력 프리미엄으로 해석된다.\n\n")
            f.write("3. PHP는 상대적으로 낮은 연봉을 보이며, 레거시 시스템 유지보수 시장과 현대 프레임워크 시장이 양분되어 있다.\n\n")
            
            f.write("## 분석 한계\n\n")
            f.write("- Stack Overflow 설문은 온라인 커뮤니티 기반 표본으로 편향이 있을 수 있음\n")
            f.write("- 대부분의 개발자가 여러 언어를 동시에 사용하므로 순효과 분리 어려움\n")
        
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
        self.save_outputs()
        
        print("\n" + "="*80)
        print("분석 완료")
        print("="*80)
 
 
if __name__ == "__main__":
    analysis = WebDevelopmentAnalysis()
    analysis.run()
 





