#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GRADE_SUCCESS_LOG = "Grading completed successfully"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def safe_load_json(path: Path, label: str) -> Optional[Any]:
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [WARN] failed to load {label}: {path} ({exc})")
        return None


def safe_load_jsonl(path: Path, label: str) -> Optional[List[Dict[str, Any]]]:
    try:
        return load_jsonl(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [WARN] failed to load {label}: {path} ({exc})")
        return None


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def format_score(score: Any) -> str:
    if score is None:
        return "n/a"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def is_run_group_dir(path: Path) -> bool:
    return path.is_dir() and "run-group" in path.name


def find_re_grade_score(re_grade_dir: Path, paper_name: str) -> Tuple[Optional[Any], Optional[bool]]:
    if not re_grade_dir.is_dir():
        return None, None

    run_groups = sorted(
        (
            run_group
            for run_group in re_grade_dir.iterdir()
            if run_group.is_dir() and "run-group" in run_group.name
        ),
        key=lambda run_group: run_group.name,
        reverse=True,
    )

    for run_group in run_groups:
        for entry in run_group.iterdir():
            if not entry.is_dir() or not entry.name.startswith(f"{paper_name}_"):
                continue

            grade_file = entry / "grade.json"
            if not grade_file.exists():
                continue

            grade_data = safe_load_json(grade_file, "re_grade grade.json")
            if grade_data is None:
                continue

            grader_log = grade_data.get("grader_log", "")
            if grader_log != GRADE_SUCCESS_LOG:
                continue

            return grade_data.get("score"), True

    return None, None


def load_runtime_seconds(run_dir: Path) -> Optional[float]:
    metadata_file = run_dir / "metadata.json"
    if metadata_file.exists():
        metadata_data = safe_load_json(metadata_file, "metadata.json")
        if metadata_data is None:
            return None
        runtime_seconds = metadata_data.get("runtime_in_seconds")
        if runtime_seconds is None:
            print(f"  [WARN] runtime_in_seconds not found for {run_dir.name}")
            return None
        try:
            return float(runtime_seconds)
        except (TypeError, ValueError):
            print(f"  [WARN] invalid runtime_in_seconds for {run_dir.name}: {runtime_seconds}")
            return None

    conversation_file = run_dir / "conversation.jsonl"
    if conversation_file.exists():
        conversation_data = safe_load_jsonl(conversation_file, "conversation.jsonl")
        if not conversation_data:
            print(f"  [WARN] conversation.jsonl is empty or invalid for {run_dir.name}")
            return None

        try:
            return float(conversation_data[-1]["ts"] - conversation_data[0]["ts"])
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  [WARN] invalid conversation timestamps for {run_dir.name}: {exc}")
            return None

    print(f"  [WARN] metadata.json and conversation.jsonl both missing for {run_dir.name}")
    return None


def summarize_run_group(run_group_dir: Path) -> Dict[str, Any]:
    re_grade_dir = run_group_dir / "re_grade"
    has_re_grade = re_grade_dir.is_dir()
    if has_re_grade:
        print(f"Found re_grade directory: {re_grade_dir}")

    result: Dict[str, Any] = {
        "all_result": {},
        "each_paper": {},
    }
    all_scores: List[float] = []
    all_status: List[bool] = []
    all_time_diff_seconds: List[float] = []
    re_graded_papers: List[str] = []

    for paper_run_dir in sorted(
        (entry for entry in run_group_dir.iterdir() if entry.is_dir() and entry.name != "re_grade"),
        key=lambda entry: entry.name,
    ):
        paper_name = paper_run_dir.name.split("_", 1)[0]
        original_grade_file = paper_run_dir / "grade.json"

        original_score = None
        original_status = None
        if original_grade_file.exists():
            grade_data = safe_load_json(original_grade_file, "original grade.json")
            if grade_data is not None:
                original_score = grade_data.get("score")
                original_status = grade_data.get("grader_log") == GRADE_SUCCESS_LOG

        new_score, new_status = (None, None)
        if has_re_grade:
            new_score, new_status = find_re_grade_score(re_grade_dir, paper_name)

        if new_score is not None:
            final_score = new_score
            final_status = new_status
            re_graded_papers.append(paper_name)
            print(
                f"  [RE-GRADE] {paper_name}: {format_score(original_score)} -> {format_score(new_score)}"
            )
        elif original_score is not None:
            final_score = original_score
            final_status = original_status
        else:
            print(f"  [SKIP] {paper_name}: no grade.json found")
            continue

        all_status.append(bool(final_status))
        if final_status and final_score is not None:
            try:
                all_scores.append(float(final_score))
            except (TypeError, ValueError):
                print(f"  [WARN] invalid score for {paper_name}: {final_score}")

        result["each_paper"][paper_name] = {
            "score": final_score,
            "original_score": original_score if new_score is not None else None,
            "re_graded": new_score is not None,
            "status": bool(final_status),
        }

        runtime_seconds = load_runtime_seconds(paper_run_dir)
        if runtime_seconds is not None:
            result["each_paper"][paper_name]["time_diff (s)"] = runtime_seconds
            result["each_paper"][paper_name]["time_diff (h)"] = runtime_seconds / 3600
            all_time_diff_seconds.append(runtime_seconds)
        else:
            result["each_paper"][paper_name]["time_diff (s)"] = None
            result["each_paper"][paper_name]["time_diff (h)"] = None

    result["all_result"] = {
        "scores": mean(all_scores),
        "grade_success_num": float(sum(all_status)),
        "grade_total_num": float(len(all_status)),
        "time_diff (h)": mean(all_time_diff_seconds) / 3600 if all_time_diff_seconds else 0.0,
        "re_graded_count": len(re_graded_papers),
        "re_graded_papers": re_graded_papers,
    }
    result["each_paper"] = dict(sorted(result["each_paper"].items(), key=lambda item: item[0]))

    output_file = run_group_dir / "all_result.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4, ensure_ascii=False)

    print(f"\n=== Results for {run_group_dir} ===")
    print(f"Total papers: {len(all_status)}")
    print(f"Grade success: {sum(all_status)}")
    print(f"Average score: {mean(all_scores):.4f}")
    print(f"Re-graded papers: {len(re_graded_papers)}")
    if re_graded_papers:
        print(f"  - {', '.join(re_graded_papers)}")
    print(f"Output saved to: {output_file}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate PaperBench scores for one or more run-group directories and "
            "write all_result.json into each run-group directory."
        )
    )
    parser.add_argument(
        "run_dirs",
        metavar="RUN_DIR",
        nargs="+",
        type=Path,
        help="Path(s) to PaperBench run-group directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        if not is_run_group_dir(run_dir):
            raise ValueError(f"Not a run-group directory: {run_dir}")
        summarize_run_group(run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
