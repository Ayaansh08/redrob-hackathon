"""
01_preprocess.py

Preprocesses candidate data from JSON/JSONL format, flattens it, adds sanity flags,
and saves the cleaned data for downstream embedding and scoring.
"""

import json
import argparse
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import numpy as np

CONSULTING_FIRMS = [
    "infosys", "tcs", "wipro", "accenture", "capgemini",
    "cognizant", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "lti", "ltts"
]

TRANSITION_PHRASES = [
    "transition", "transitioning", "looking to move",
    "curious about ai", "experimenting with",
    "self-taught", "recently started"
]

def load_jsonl(path: Path) -> list[dict]:
    """
    Load candidates from a JSONL file.
    
    Args:
        path: Path to the JSONL file.
        
    Returns:
        List of parsed candidate dictionaries.
    """
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for count, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                candidates.append(cand)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {count}: {e}")
            
            if count % 10000 == 0:
                print(f"Loaded {count} candidates...")
    print(f"Total candidates loaded from JSONL: {len(candidates)}")
    return candidates

def load_json_array(path: Path) -> list[dict]:
    """
    Load candidates from a JSON array file.
    
    Args:
        path: Path to the JSON file.
        
    Returns:
        List of parsed candidate dictionaries.
    """
    with open(path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"Total candidates loaded from JSON array: {len(candidates)}")
    return candidates

def _safe_str(val) -> str:
    return str(val).strip() if val is not None else ""

def _safe_float(val) -> float:
    try:
        return float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0

def _safe_int(val) -> int:
    try:
        return int(val) if val is not None else 0
    except (ValueError, TypeError):
        return 0

def _safe_bool(val) -> bool:
    return bool(val) if val is not None else False

def _safe_list(val) -> list:
    return val if isinstance(val, list) else []

def _safe_dict(val) -> dict:
    return val if isinstance(val, dict) else {}

def flatten_candidate(raw: dict) -> dict:
    """
    Flatten a nested candidate dictionary into a single flat dictionary.
    
    Args:
        raw: Raw candidate dictionary.
        
    Returns:
        Flattened candidate dictionary.
    """
    flat = {}
    
    # IDENTITY
    flat["candidate_id"] = _safe_str(raw.get("candidate_id"))
    profile = _safe_dict(raw.get("profile"))
    flat["name"] = _safe_str(profile.get("anonymized_name"))
    flat["headline"] = _safe_str(profile.get("headline"))
    flat["summary"] = _safe_str(profile.get("summary"))
    flat["location"] = _safe_str(profile.get("location"))
    flat["country"] = _safe_str(profile.get("country"))
    flat["years_of_experience"] = _safe_float(profile.get("years_of_experience"))
    flat["current_title"] = _safe_str(profile.get("current_title"))
    flat["current_company"] = _safe_str(profile.get("current_company"))
    flat["current_company_size"] = _safe_str(profile.get("current_company_size"))
    flat["current_industry"] = _safe_str(profile.get("current_industry"))
    
    # CAREER
    career_history = _safe_list(raw.get("career_history"))
    flat["num_jobs"] = len(career_history)
    flat["total_career_months"] = sum(_safe_int(job.get("duration_months")) for job in career_history)
    
    companies = [_safe_str(job.get("company")) for job in career_history if job.get("company")]
    flat["all_companies"] = ", ".join(companies)
    
    industries = list(set(_safe_str(job.get("industry")) for job in career_history if job.get("industry")))
    flat["all_industries"] = ", ".join(industries)
    
    descriptions = [_safe_str(job.get("description")) for job in career_history if job.get("description")]
    flat["all_job_descriptions"] = " | ".join(descriptions)
    
    flat["longest_tenure_months"] = max([_safe_int(job.get("duration_months")) for job in career_history], default=0)
    flat["has_current_job"] = any(_safe_bool(job.get("is_current")) for job in career_history)
    
    company_sizes = [_safe_str(job.get("company_size")) for job in career_history if job.get("company_size")]
    flat["career_company_sizes"] = ", ".join(company_sizes)
    
    job_titles = [_safe_str(job.get("title")) for job in career_history if job.get("title")]
    flat["all_job_titles"] = ", ".join(job_titles)
    
    # EDUCATION
    education = _safe_list(raw.get("education"))
    highest_degree = "Other"
    edu_tier = "unknown"
    tier_ranks = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    
    has_phd = False
    has_masters = False
    has_bachelors = False
    
    best_tier_rank = -1
    
    for edu in education:
        deg = _safe_str(edu.get("degree")).lower()
        if any(x in deg for x in ["phd", "doctorate"]):
            has_phd = True
        elif any(x in deg for x in ["master", "m.tech", "m.s."]):
            has_masters = True
        elif any(x in deg for x in ["bachelor", "b.tech", "b.e.", "b.s."]):
            has_bachelors = True
            
        tier_val = _safe_str(edu.get("tier")).lower()
        if tier_val in tier_ranks:
            if tier_ranks[tier_val] > best_tier_rank:
                best_tier_rank = tier_ranks[tier_val]
                edu_tier = tier_val
                
    if has_phd:
        highest_degree = "PhD"
    elif has_masters:
        highest_degree = "Masters"
    elif has_bachelors:
        highest_degree = "Bachelors"
        
    flat["highest_degree"] = highest_degree
    flat["edu_tier"] = edu_tier if best_tier_rank >= 0 else "unknown"
    
    # SKILLS
    skills = _safe_list(raw.get("skills"))
    flat["num_skills"] = len(skills)
    skill_names = [_safe_str(s.get("name")) for s in skills if s.get("name")]
    flat["skills_text"] = ", ".join(skill_names)
    
    clean_skills = []
    expert_names = []
    has_expert_no_duration = False
    
    for s in skills:
        prof = _safe_str(s.get("proficiency")).lower()
        s_name = _safe_str(s.get("name"))
        dur = _safe_int(s.get("duration_months"))
        
        if prof == "expert":
            expert_names.append(s_name)
            if dur == 0:
                has_expert_no_duration = True
                
        clean_skills.append({
            "name": s_name,
            "proficiency": prof,
            "endorsements": _safe_int(s.get("endorsements")),
            "duration_months": dur
        })
        
    flat["skills_json"] = json.dumps(clean_skills)
    flat["has_expert_skills"] = len(expert_names) > 0
    flat["expert_skill_names"] = ", ".join(expert_names)
    flat["_has_expert_no_duration"] = has_expert_no_duration # Helper for flags
    
    # REDROB SIGNALS
    signals = _safe_dict(raw.get("redrob_signals"))
    flat["completeness"] = _safe_float(signals.get("profile_completeness_score"))
    flat["open_to_work"] = _safe_bool(signals.get("open_to_work_flag"))
    
    g_score = signals.get("github_activity_score")
    flat["github_score"] = float(g_score) if g_score is not None else -1.0
    
    flat["notice_days"] = _safe_int(signals.get("notice_period_days"))
    flat["recruiter_response_rate"] = _safe_float(signals.get("recruiter_response_rate"))
    flat["avg_response_hours"] = _safe_float(signals.get("avg_response_time_hours"))
    flat["interview_completion"] = _safe_float(signals.get("interview_completion_rate"))
    
    o_acc = signals.get("offer_acceptance_rate")
    flat["offer_acceptance"] = float(o_acc) if o_acc is not None else -1.0
    
    flat["willing_to_relocate"] = _safe_bool(signals.get("willing_to_relocate"))
    flat["search_appearances"] = _safe_int(signals.get("search_appearance_30d"))
    flat["saved_by_recruiters"] = _safe_int(signals.get("saved_by_recruiters_30d"))
    flat["profile_views"] = _safe_int(signals.get("profile_views_received_30d"))
    flat["applications_30d"] = _safe_int(signals.get("applications_submitted_30d"))
    flat["connection_count"] = _safe_int(signals.get("connection_count"))
    flat["endorsements_received"] = _safe_int(signals.get("endorsements_received"))
    flat["verified_email"] = _safe_bool(signals.get("verified_email"))
    flat["verified_phone"] = _safe_bool(signals.get("verified_phone"))
    flat["linkedin_connected"] = _safe_bool(signals.get("linkedin_connected"))
    flat["preferred_work_mode"] = _safe_str(signals.get("preferred_work_mode"))
    
    salary = _safe_dict(signals.get("expected_salary_range_inr_lpa"))
    flat["salary_min_lpa"] = _safe_float(salary.get("min"))
    flat["salary_max_lpa"] = _safe_float(salary.get("max"))
    
    last_active = _safe_str(signals.get("last_active_date"))
    days_since = 0
    if last_active:
        try:
            dt = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_since = (date(2026, 6, 29) - dt).days
        except ValueError:
            pass
    flat["days_since_active"] = days_since
    
    flat["skill_assessment_json"] = json.dumps(_safe_dict(signals.get("skill_assessment_scores")))
    
    # COMBINED TEXT
    combined_parts = [
        flat["summary"],
        flat["all_job_descriptions"],
        flat["skills_text"],
        flat["headline"]
    ]
    flat["candidate_full_text"] = " ".join(combined_parts).strip()
    flat["candidate_full_text"] = " ".join(flat["candidate_full_text"].split())
    
    return flat

def add_sanity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean sanity flag columns to the dataframe based on candidate fields.
    
    Args:
        df: The pandas DataFrame of flattened candidates.
        
    Returns:
        The updated DataFrame with flag columns.
    """
    df["flag_no_github"] = df["github_score"] == -1
    df["flag_no_career"] = df["num_jobs"] == 0
    df["flag_experience_mismatch"] = (df["years_of_experience"] - (df["total_career_months"] / 12.0)).abs() > 3
    df["flag_expert_no_duration"] = df["has_expert_skills"] & df["_has_expert_no_duration"]
    df["flag_low_completeness"] = df["completeness"] < 40
    df["flag_inactive"] = df["days_since_active"] > 180
    
    def check_transition(summary: str) -> bool:
        if not isinstance(summary, str):
            return False
        summary_low = summary.lower()
        return any(phrase in summary_low for phrase in TRANSITION_PHRASES)
        
    df["flag_transition_summary"] = df["summary"].apply(check_transition)
    
    def check_consulting(row) -> bool:
        if row["num_jobs"] == 0:
            return False
        companies = str(row["all_companies"]).lower()
        if not companies:
            return False
        comp_list = [c.strip() for c in companies.split(",")]
        # ALL companies must be in the consulting list
        for c in comp_list:
            if not any(firm in c for firm in CONSULTING_FIRMS):
                return False
        return True
        
    df["flag_consulting_only"] = df.apply(check_consulting, axis=1)
    
    # Drop the helper column
    df.drop(columns=["_has_expert_no_duration"], inplace=True)
    
    return df

def print_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics for the processed DataFrame.
    
    Args:
        df: The pandas DataFrame of flattened and flagged candidates.
    """
    print(f"\n--- SUMMARY ---")
    print(f"Total rows in output dataframe: {len(df)}")
    
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    print("\nSanity Flags Count (True):")
    for col in flag_cols:
        print(f"  {col}: {df[col].sum()}")
        
    print("\nYears of Experience Distribution:")
    bins = [-1, 2, 4, 6, 8, 10, 100]
    labels = ["<2", "2-4", "4-6", "6-8", "8-10", "10+"]
    yoe_dist = pd.cut(df["years_of_experience"], bins=bins, labels=labels, right=False).value_counts().sort_index()
    for label, count in yoe_dist.items():
        print(f"  {label}: {count}")
        
    print("\nGitHub Score:")
    gh_minus_1 = (df["github_score"] == -1).sum()
    gh_positive = (df["github_score"] > 0).sum()
    print(f"  Score == -1: {gh_minus_1}")
    print(f"  Score > 0: {gh_positive}")
    
    print("\nTop 10 Current Industries:")
    top_industries = df["current_industry"].replace("", "UNKNOWN").value_counts().head(10)
    for ind, count in top_industries.items():
        print(f"  {ind}: {count}")
    print("-----------------\n")

def main(use_sample: bool) -> None:
    """
    Main entry point for preprocessing script.
    
    Args:
        use_sample: If True, uses the sample dataset instead of the full JSONL.
    """
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    if use_sample:
        input_path = base_dir / "sample_candidates.json"
        print(f"Running in SAMPLE mode. Reading from {input_path}")
        raw_candidates = load_json_array(input_path)
    else:
        input_path = base_dir / "candidates.jsonl"
        print(f"Running in FULL mode. Reading from {input_path}")
        raw_candidates = load_jsonl(input_path)
        
    print("Flattening candidates...")
    flat_candidates = [flatten_candidate(cand) for cand in raw_candidates]
    
    print("Creating DataFrame and adding flags...")
    df = pd.DataFrame(flat_candidates)
    df = add_sanity_flags(df)
    
    print_summary(df)
    
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    pkl_path = processed_dir / "candidates_clean.pkl"
    csv_path = processed_dir / "preview.csv"
    
    print(f"Saving pickle to {pkl_path}...")
    df.to_pickle(pkl_path)
    
    print(f"Saving CSV preview to {csv_path}...")
    df.head(200).to_csv(csv_path, index=False)
    
    print(f"Done! Final dataframe shape: {df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Redrob candidates.")
    parser.add_argument("--sample", action="store_true", help="Run on sample_candidates.json instead of candidates.jsonl")
    args = parser.parse_args()
    
    main(args.sample)
