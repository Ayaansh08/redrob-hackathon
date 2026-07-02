#!/usr/bin/env python3
"""
Generate the final Redrob challenge submission CSV from scored candidates.

This script is intentionally downstream-only: it reads final_score and related
diagnostic columns produced by 03_score.py, selects the top 100, formats the
required CSV, and validates it with validate_submission.py.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import pandas as pd


REQUIRED_COLUMNS = [
    "candidate_id",
    "final_score",
    "jd_match",
    "career_quality",
    "behavioral",
    "stability",
    "honeypot_multiplier",
]

SUBMISSION_HEADER = ["candidate_id", "rank", "score", "reasoning"]
EXPECTED_ROWS = 100

FLAG_COLUMNS = [
    "flag_keyword_stuffer",
    "flag_transition_summary",
    "flag_consulting_only",
    "flag_experience_mismatch",
    "flag_low_completeness",
    "flag_inactive",
    "flag_no_github",
    "flag_no_career",
    "flag_expert_no_duration",
]


def repo_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"scored_candidates.pkl is missing required columns: {missing}")


def as_bool(value) -> bool:
    try:
        return bool(value) and not pd.isna(value)
    except TypeError:
        return bool(value)


def parse_skills(raw_value) -> list[dict]:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []
    if isinstance(raw_value, list):
        return [item for item in raw_value if isinstance(item, dict)]
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def top_skill_names(row: pd.Series, limit: int = 3) -> list[str]:
    explicit = row.get("expert_skill_names")
    if isinstance(explicit, str) and explicit.strip():
        names = [part.strip() for part in explicit.split(",") if part.strip()]
        if names:
            return names[:limit]

    skills = parse_skills(row.get("skills_json"))
    ranked = []
    proficiency_rank = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}
    for skill in skills:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        proficiency = str(skill.get("proficiency", "")).lower()
        endorsements = skill.get("endorsements", 0) or 0
        duration = skill.get("duration_months", 0) or 0
        if proficiency in {"expert", "advanced"}:
            ranked.append((proficiency_rank.get(proficiency, 0), endorsements, duration, name))

    ranked.sort(reverse=True)
    return [item[-1] for item in ranked[:limit]]


def true_flags(row: pd.Series) -> list[str]:
    flags = []
    for col in FLAG_COLUMNS:
        if col in row.index and as_bool(row.get(col)):
            flags.append(col.replace("flag_", ""))
    return flags


def format_score(score: float) -> str:
    return f"{float(score):.12g}"


def format_component(value) -> str:
    return f"{float(value):.1f}"


def build_reasoning(row: pd.Series) -> str:
    title = str(row.get("current_title") or "Candidate").strip() or "Candidate"
    company = str(row.get("current_company") or "unknown company").strip() or "unknown company"

    parts = [
        f"{title} at {company}",
        (
            "scores "
            f"jd_match {format_component(row['jd_match'])}, "
            f"career {format_component(row['career_quality'])}, "
            f"behavioral {format_component(row['behavioral'])}, "
            f"stability {format_component(row['stability'])}"
        ),
    ]

    skills = top_skill_names(row)
    if skills:
        parts.append(f"strong skills: {', '.join(skills)}")

    if as_bool(row.get("has_tech_engineering_title")):
        parts.append("engineering-relevant title history")

    multiplier = float(row["honeypot_multiplier"])
    flags = true_flags(row)
    if multiplier == 1.0:
        parts.append("honeypot_multiplier 1.0 clean candidate")
    elif flags:
        parts.append(f"honeypot_multiplier {multiplier:.1f} due to {', '.join(flags[:3])}")
    else:
        parts.append(f"honeypot_multiplier {multiplier:.1f}")

    return "; ".join(parts)


def load_ranked_candidates(input_path: Path) -> pd.DataFrame:
    df = pd.read_pickle(input_path)
    require_columns(df)

    ranked = df.sort_values(
        by=["final_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(EXPECTED_ROWS).copy()

    if len(ranked) != EXPECTED_ROWS:
        raise ValueError(f"Expected at least {EXPECTED_ROWS} scored candidates; found {len(ranked)}")

    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def write_submission(ranked: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUBMISSION_HEADER)
        for _, row in ranked.iterrows():
            writer.writerow(
                [
                    row["candidate_id"],
                    int(row["rank"]),
                    format_score(row["final_score"]),
                    build_reasoning(row),
                ]
            )


def validate_submission(base_dir: Path, output_path: Path) -> list[str]:
    sys.path.insert(0, str(base_dir))
    try:
        from validate_submission import validate_submission as run_validation
    finally:
        sys.path.pop(0)
    return run_validation(output_path)


def score_band_counts(scores: pd.Series) -> dict[str, int]:
    bands = {
        "90-100": ((scores >= 90) & (scores <= 100)).sum(),
        "80-89.999": ((scores >= 80) & (scores < 90)).sum(),
        "70-79.999": ((scores >= 70) & (scores < 80)).sum(),
        "60-69.999": ((scores >= 60) & (scores < 70)).sum(),
        "50-59.999": ((scores >= 50) & (scores < 60)).sum(),
        "<50": (scores < 50).sum(),
    }
    return {band: int(count) for band, count in bands.items()}


def print_summary(ranked: pd.DataFrame) -> None:
    scores = ranked["final_score"].astype(float)
    soft_or_penalized = ranked[ranked["honeypot_multiplier"].astype(float) < 1.0]

    print("\n--- Submission Summary ---")
    print(f"Candidate count: {len(ranked)}")
    print(
        "Score range: "
        f"min={scores.min():.4f}, max={scores.max():.4f}, mean={mean(scores):.4f}"
    )
    print(f"Top-100 candidates with honeypot_multiplier < 1.0: {len(soft_or_penalized)}")
    if not soft_or_penalized.empty:
        print("Penalized candidates in final output:")
        for _, row in soft_or_penalized.iterrows():
            print(
                f"  rank {int(row['rank'])}: {row['candidate_id']} "
                f"score={float(row['final_score']):.4f} "
                f"honeypot_multiplier={float(row['honeypot_multiplier']):.1f}"
            )

    print("Score bands:")
    for band, count in score_band_counts(scores).items():
        print(f"  {band}: {count}")


def main() -> int:
    base_dir = repo_base_dir()
    parser = argparse.ArgumentParser(description="Generate and validate final submission CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "outputs" / "scored_candidates.pkl",
        help="Path to scored_candidates.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "outputs" / "redrob_submission.csv",
        help="Output CSV path. Use your registered participant ID filename before submission if required.",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else base_dir / args.input
    output_path = args.output if args.output.is_absolute() else base_dir / args.output

    ranked = load_ranked_candidates(input_path)
    write_submission(ranked, output_path)

    errors = validate_submission(base_dir, output_path)
    print(f"Generated submission: {output_path}")
    if errors:
        print(f"\nValidation failed ({len(errors)} issue(s)):")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nValidator output: Submission is valid.")
    print_summary(ranked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
