"""
Extract specific candidates for manual review.

This script reads raw profile records from candidates.jsonl and, when available,
cross-references processed/candidates_clean.pkl and outputs/scored_candidates.pkl.

Edit HARDCODED_CANDIDATE_IDS at the top for quick runs, or override with
`--ids CAND_0001,CAND_0002`.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# Edit this list when you want to review a different set of IDs.
HARDCODED_CANDIDATE_IDS = [
    # Example:
    # "CAND_0088898",
    # "CAND_0089350",
    # "CAND_0052367",
]

DIVIDER = "\n" + "=" * 80 + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract full candidate profiles for manual review."
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated candidate IDs to extract (overrides hardcoded list).",
    )
    return parser.parse_args()


def candidate_ids_from_args(arg_ids: str) -> list[str]:
    if arg_ids:
        return [candidate_id.strip() for candidate_id in arg_ids.split(",") if candidate_id.strip()]
    return [candidate_id for candidate_id in HARDCODED_CANDIDATE_IDS if candidate_id.strip()]


def load_raw_candidates(jsonl_path: Path, desired_ids: set[str]) -> dict[str, dict]:
    raw_by_id: dict[str, dict] = {}
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing raw candidates file: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: failed to parse line {line_number} in {jsonl_path}: {exc}")
                continue

            candidate_id = raw.get("candidate_id")
            if candidate_id in desired_ids:
                raw_by_id[candidate_id] = raw
                if len(raw_by_id) == len(desired_ids):
                    break

    return raw_by_id


def load_optional_dataframe(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_pickle(path)
    return None


def to_plain_python(value):
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: to_plain_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_python(v) for v in value]
    try:
        return value.item()
    except Exception:
        return str(value)


def row_to_jsonable(row: pd.Series) -> dict:
    return {str(k): to_plain_python(v) for k, v in row.to_dict().items()}


def summarize_job_history(raw: dict) -> list[str]:
    history = raw.get("career_history") or []
    rows = []
    for job in history:
        title = job.get("title") or "(no title)"
        company = job.get("company") or "(no employer)"
        months = job.get("duration_months")
        when = job.get("duration")
        current = job.get("is_current")

        pieces = [title, f"@ {company}"]
        if current:
            pieces.append("[current]")
        if months is not None:
            pieces.append(f"{months} months")
        elif when:
            pieces.append(str(when))
        rows.append(" — ".join(pieces))
    return rows


def get_skills(raw: dict) -> list[str]:
    skills = raw.get("skills")
    if isinstance(skills, list):
        return [str(skill) for skill in skills]
    profile = raw.get("profile") or {}
    skills = profile.get("skills")
    if isinstance(skills, list):
        return [str(skill) for skill in skills]
    return []


def get_summary(raw: dict) -> str:
    profile = raw.get("profile") or {}
    summary = profile.get("summary") or profile.get("headline") or ""
    return str(summary).strip()


def get_name(raw: dict) -> str:
    profile = raw.get("profile") or {}
    return str(profile.get("anonymized_name") or profile.get("name") or "").strip()


def get_current_title(raw: dict) -> str:
    profile = raw.get("profile") or {}
    return str(profile.get("current_title") or "").strip()


def get_current_company(raw: dict) -> str:
    profile = raw.get("profile") or {}
    return str(profile.get("current_company") or "").strip()


def extract_scored_fields(row: dict) -> dict:
    fields = {}
    for key, value in row.items():
        if key in {
            "jd_match",
            "career_quality",
            "behavioral",
            "stability",
            "final_score",
            "honeypot_multiplier",
        } or key.startswith("flag_"):
            fields[key] = to_plain_python(value)
    return fields


def build_candidate_block(candidate_id: str, raw: dict, cleaned: dict | None, scored: dict | None) -> tuple[str, dict]:
    name = get_name(raw)
    current_title = get_current_title(raw)
    current_company = get_current_company(raw)
    history = summarize_job_history(raw)
    skills = get_skills(raw)
    summary = get_summary(raw)

    lines = []
    lines.append(DIVIDER)
    lines.append(f"# Candidate: {candidate_id}")
    lines.append(f"Name: {name or '(unknown)'}")
    lines.append(f"Current title: {current_title or '(unknown)'}")
    lines.append(f"Current employer: {current_company or '(unknown)'}")
    lines.append("")
    lines.append("## Job history")
    if history:
        for row in history:
            lines.append(f"- {row}")
    else:
        lines.append("- (no career history available)")

    lines.append("")
    lines.append("## Skills")
    if skills:
        lines.append(", ".join(skills))
    else:
        lines.append("(no skills list available)")

    lines.append("")
    lines.append("## Summary / bio")
    lines.append(summary or "(no summary available)")

    lines.append("")
    lines.append("## Raw record")
    lines.append("```json")
    lines.append(json.dumps(raw, indent=2, ensure_ascii=False))
    lines.append("```")

    if cleaned is not None:
        lines.append("")
        lines.append("## Cleaned / flattened record")
        lines.append("```json")
        lines.append(json.dumps(cleaned, indent=2, ensure_ascii=False))
        lines.append("```")

    if scored is not None:
        lines.append("")
        lines.append("## Scored fields")
        if scored:
            for key, value in scored.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("(scored candidate record found, but no review fields were present)")

    text_block = "\n".join(lines)

    output_object = {
        "candidate_id": candidate_id,
        "name": name,
        "current_title": current_title,
        "current_company": current_company,
        "job_history": history,
        "skills": skills,
        "summary": summary,
        "raw_record": raw,
        "cleaned_record": cleaned,
        "scored_fields": scored,
    }

    return text_block, output_object


def main():
    args = parse_args()
    candidate_ids = candidate_ids_from_args(args.ids)

    if not candidate_ids:
        raise SystemExit(
            "No candidate IDs provided. Edit HARDCODED_CANDIDATE_IDS or pass --ids." 
        )

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    raw_path = base_dir / "candidates.jsonl"
    processed_path = base_dir / "processed" / "candidates_clean.pkl"
    scored_path = base_dir / "outputs" / "scored_candidates.pkl"

    desired_ids = set(candidate_ids)
    raw_candidates = load_raw_candidates(raw_path, desired_ids)
    cleaned_df = load_optional_dataframe(processed_path)
    scored_df = load_optional_dataframe(scored_path)

    review_objects: list[dict] = []
    markdown_blocks: list[str] = []

    for candidate_id in candidate_ids:
        raw = raw_candidates.get(candidate_id)
        if raw is None:
            warning = f"WARNING: candidate_id {candidate_id} NOT FOUND in {raw_path}."
            print(warning)
            markdown_blocks.append(DIVIDER + warning)
            review_objects.append({"candidate_id": candidate_id, "found": False})
            continue

        cleaned = None
        if cleaned_df is not None:
            cleaned_row = cleaned_df.loc[cleaned_df["candidate_id"] == candidate_id]
            if not cleaned_row.empty:
                cleaned = row_to_jsonable(cleaned_row.iloc[0])

        scored = None
        if scored_df is not None:
            scored_row = scored_df.loc[scored_df["candidate_id"] == candidate_id]
            if not scored_row.empty:
                scored = extract_scored_fields(scored_row.iloc[0].to_dict())

        block_text, review_object = build_candidate_block(candidate_id, raw, cleaned, scored)
        print(block_text)
        markdown_blocks.append(block_text)
        review_objects.append(review_object)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = outputs_dir / f"candidate_review_{timestamp}.json"
    md_path = outputs_dir / f"candidate_review_{timestamp}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(review_objects, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_blocks))

    print(DIVIDER)
    print(f"Wrote review files:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
