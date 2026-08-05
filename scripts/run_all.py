#!/usr/bin/env python3
"""Run every independent domain and create the Markdown report."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / "output" / "evidence"
EXECUTION_LOG = EVIDENCE_DIR / "execution_log.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="전체 도메인 분석과 통합 보고서를 생성합니다.")
    parser.add_argument("--skip-ml", action="store_true", help="Domain 6 모델 재학습 생략")
    return parser.parse_args()


def run_step(label: str, command: list[str], environment: dict[str, str]) -> str:
    started = datetime.now()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout
    if completed.stderr:
        output += ("\n" if output else "") + completed.stderr
    record = (
        f"\n{'=' * 72}\n[{label}]\n"
        f"command: {' '.join(command)}\n"
        f"started: {started.isoformat(timespec='seconds')}\n"
        f"exit_code: {completed.returncode}\n"
        f"{'-' * 72}\n{output.rstrip()}\n"
    )
    print(record)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} 실행 실패(exit={completed.returncode})")
    return record


def main() -> None:
    args = parse_args()
    data_path = PROJECT_ROOT / "data" / "results.csv"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"공통 정제 데이터가 없습니다: {data_path}\n"
            "먼저 `python scripts/build_dataset.py --year 2025`를 실행하세요."
        )

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "output" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(cache_dir / "xdg")

    python = sys.executable
    steps: list[tuple[str, list[str]]] = [
        ("Domain 1", [python, "domain1_dynamic/analysis.py"]),
        ("Domain 2", [python, "domain2_jvm/analysis.py"]),
        ("Domain 3", [python, "domain3_web/domain3_analysis.py"]),
        ("Domain 4", [python, "domain4_system/analysis.py"]),
        ("Domain 5", [python, "domain5_platform/analysis.py"]),
    ]
    if not args.skip_ml:
        steps.append(("Domain 6", [python, "-m", "domain6_ml.model"]))
    steps.append(("통합 report.md", [python, "scripts/generate_report.py"]))
    # 보고서 테스트는 생성된 도메인 산출물을 읽으므로 깨끗한 상태에서도
    # 실행할 수 있게 모든 산출물 생성이 끝난 뒤 수행한다.
    steps.append(("자동 테스트", [python, "-m", "pytest", "-q"]))

    records = [
        "Stack Overflow Salary Analysis - Execution Log\n"
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}\n"
    ]
    try:
        for label, command in steps:
            records.append(run_step(label, command, environment))
    finally:
        EXECUTION_LOG.write_text("".join(records), encoding="utf-8")

    print(f"[완료] 실행 로그: {EXECUTION_LOG}")


if __name__ == "__main__":
    main()
