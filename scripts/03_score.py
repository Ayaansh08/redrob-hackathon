"""
03_score.py

Scores all candidates across various features and computes the final weighted sum.
"""

import pickle
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

def load_data(base_dir: Path, use_sample: bool = False):
    processed_dir = base_dir / "processed"
    
    print("Loading candidate data...")
    df = pd.read_pickle(processed_dir / "candidates_clean.pkl")
    if use_sample:
        df = df.head(50).copy()
    
    print("Loading embeddings and BM25 index...")
    embeddings = np.load(processed_dir / "embeddings.npy")
    if use_sample:
        embeddings = embeddings[:50]
        
    jd_embedding = np.load(processed_dir / "jd_embedding.npy")
    
    with open(processed_dir / "bm25_index.pkl", "rb") as f:
        bm25_index = pickle.load(f)
        
    with open(processed_dir / "jd_tokens.pkl", "rb") as f:
        jd_tokens = pickle.load(f)
        
    # Attempt to load chunked JD embeddings if available
    jd_chunk_embeddings = None
    try:
        jd_chunk_path = processed_dir / 'jd_chunk_embeddings.npy'
        if jd_chunk_path.exists():
            jd_chunk_embeddings = np.load(jd_chunk_path)
    except Exception:
        jd_chunk_embeddings = None

    return df, embeddings, jd_embedding, jd_chunk_embeddings, bm25_index, jd_tokens


def score_jd_match(df: pd.DataFrame, embeddings: np.ndarray, jd_embedding: np.ndarray, jd_chunk_embeddings: np.ndarray, bm25_index, jd_tokens) -> pd.DataFrame:
    """JD Match Component — percentile-clipped normalization + weighted geometric mean."""
    N = len(df)
    
    # Compute raw signals
    # If chunked JD embeddings are present, compute per-chunk similarity and
    # take the max per-candidate (best matching JD chunk). Otherwise use the
    # single JD embedding.
    if jd_chunk_embeddings is not None and jd_chunk_embeddings.ndim == 2:
        # embeddings: (N, d), jd_chunk_embeddings: (num_chunks, d)
        sims = embeddings @ jd_chunk_embeddings.T
        cosine_raw = sims.max(axis=1)
    else:
        cosine_raw = embeddings @ jd_embedding
    bm25_raw = bm25_index.get_scores(jd_tokens)
    
    if N < bm25_index.corpus_size:
        bm25_raw = bm25_raw[:N]
    
    # FIX 1 — Percentile-clipped normalization (instead of raw min-max)
    p5_cos, p95_cos = np.percentile(cosine_raw, [5, 95])
    cosine_norm = np.clip((cosine_raw - p5_cos) / (p95_cos - p5_cos + 1e-9), 0, 1)
    
    p5_bm, p95_bm = np.percentile(bm25_raw, [5, 95])
    bm25_norm = np.clip((bm25_raw - p5_bm) / (p95_bm - p5_bm + 1e-9), 0, 1)
    
    # FIX 2 — Weighted geometric mean (instead of harmonic mean)
    # cosine gets 0.6 weight, bm25 gets 0.4 weight
    eps = 0.01
    cosine_safe = np.clip(cosine_norm, eps, 1.0)
    bm25_safe = np.clip(bm25_norm, eps, 1.0)
    jd_blend = 0.85 * bm25_norm + 0.15 * cosine_norm
    
    # Keyword Stuffer Flag (unchanged logic)
    bm25_rank = np.argsort(np.argsort(bm25_norm))
    cosine_rank = np.argsort(np.argsort(cosine_norm))
    flag_keyword_stuffer = (bm25_rank < N * 0.10) & (cosine_rank > N * 0.70)
    
    # Final jd_match score
    jd_match = jd_blend * 100
    
    df["bm25_norm"] = bm25_norm
    df["cosine_norm"] = cosine_norm
    df["jd_blend"] = jd_blend
    df["jd_match"] = jd_match
    df["flag_keyword_stuffer"] = flag_keyword_stuffer
    
    print(f"Mean jd_match score: {np.mean(jd_match):.4f}")
    print(f"Std dev of jd_match scores: {np.std(jd_match):.4f}")
    print(f"Count of flag_keyword_stuffer == True: {flag_keyword_stuffer.sum()}")
    print(f"Cosine percentiles used for normalization: p5={p5_cos:.4f}, p95={p95_cos:.4f}")
    print(f"BM25 percentiles used for normalization: p5={p5_bm:.4f}, p95={p95_bm:.4f}")
    
    top5_idx = np.argsort(jd_match)[::-1][:5]
    print("\nTop 5 candidate_ids by jd_match:")
    for i in top5_idx:
        print(f"  {df['candidate_id'].iloc[i]}: jd_match={jd_match[i]:.2f}, cosine_norm={cosine_norm[i]:.4f}, bm25_norm={bm25_norm[i]:.4f}")
        
    bm25_top5_idx = np.argsort(bm25_norm)[::-1][:5]
    print("\nTop 5 candidate_ids by bm25_norm alone:")
    for i in bm25_top5_idx:
        print(f"  {df['candidate_id'].iloc[i]}: bm25_norm={bm25_norm[i]:.4f}")
        
    cosine_top5_idx = np.argsort(cosine_norm)[::-1][:5]
    print("\nTop 5 candidate_ids by cosine_norm alone:")
    for i in cosine_top5_idx:
        print(f"  {df['candidate_id'].iloc[i]}: cosine_norm={cosine_norm[i]:.4f}")
        
    overlap = len(set(df["candidate_id"].iloc[bm25_top5_idx]).intersection(set(df["candidate_id"].iloc[top5_idx])))
    print(f"\nHow many candidates from bm25 top-5 survived into jd_match top-5: {overlap}/5")
    
    return df


def score_career_quality(df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing career quality score...")
    
    # FIX 3 — Differentiate non-tech "engineer" titles from tech engineering titles.
    NON_TECH_ENGINEER_TITLES = [
        "mechanical engineer", "civil engineer", "electrical engineer",
        "chemical engineer", "structural engineer", "industrial engineer",
        "site engineer", "sales engineer"
    ]

    TECH_ENGINEER_KEYWORDS = [
        "software engineer", "ml engineer", "ai engineer", "data engineer",
        "machine learning engineer", "platform engineer", "backend engineer",
        "frontend engineer", "full stack engineer", "fullstack engineer",
        "infrastructure engineer", "devops engineer", "mlops engineer",
        "search engineer", "applied scientist", "research scientist",
        "data scientist", "nlp engineer", "sde", "swe", "tech lead",
        "applied ml", "architect", "developer"
    ]

    all_titles_combined = (
        df["current_title"].fillna("") + " " +
        df["all_job_titles"].fillna("")
    ).str.lower()

    has_non_tech_engineer = all_titles_combined.str.contains(
        '|'.join(NON_TECH_ENGINEER_TITLES), na=False
    )
    has_tech_title = all_titles_combined.str.contains(
        '|'.join(TECH_ENGINEER_KEYWORDS), na=False
    )

    # Only counts as having an engineering title if it's a tech title,
    # AND not purely a non-tech "engineer" title with no tech title present
    has_engineering_title = has_tech_title & ~(has_non_tech_engineer & ~has_tech_title)

    # persist so other functions (honeypot gate) can reuse the same signal
    df["has_tech_engineering_title"] = has_engineering_title

    title_cap_mask = ~has_engineering_title
    
    # A. Experience score (max 25 pts)
    yoe = df["years_of_experience"].fillna(0)
    exp_score = np.select(
        [yoe < 2, (yoe >= 2) & (yoe < 4), (yoe >= 4) & (yoe < 6), (yoe >= 6) & (yoe < 8), (yoe >= 8) & (yoe < 10), yoe >= 10],
        [0, 8, 16, 25, 20, 14],
        default=0
    )
    
    # B. Product company score (max 20 pts)
    sizes = df["career_company_sizes"].fillna("").str.lower()
    has_startup = sizes.str.contains(r"51-200|201-500|501-1000", regex=True)
    has_growth = sizes.str.contains(r"1001-5000|5001-10000", regex=True)
    has_ent = sizes.str.contains(r"10001\+|1-10", regex=True)
    
    prod_base = np.select(
        [has_startup & has_growth, has_startup, has_growth, has_ent & ~has_startup & ~has_growth],
        [20, 15, 12, 5],
        default=8
    )
    consulting_penalty = np.where(df["flag_consulting_only"] == True, -15, 0)
    product_score = np.clip(prod_base + consulting_penalty, 0, 20)
    
    # C. Action verb score (max 20 pts)
    desc = df["all_job_descriptions"].fillna("").str.lower()
    strong = ["built", "designed", "deployed", "developed", "architected", "implemented", "launched", "created", "led", "owned", "scaled", "optimized", "shipped", "productionized", "established", "drove", "reduced", "improved"]
    weak = ["assisted", "supported", "helped", "learnt", "learned", "studied", "explored", "familiar", "exposure", "interested in", "worked alongside", "shadow"]
    
    strong_count = pd.Series(np.zeros(len(df)), index=df.index)
    for v in strong:
        strong_count += desc.str.contains(rf"\b{v}\b", regex=True).astype(int)
        
    weak_count = pd.Series(np.zeros(len(df)), index=df.index)
    for v in weak:
        weak_count += desc.str.contains(rf"\b{v}\b", regex=True).astype(int)
        
    action_raw = np.clip(strong_count * 2, 0, 14) - weak_count
    action_score = np.clip((action_raw / 14.0) * 20.0, 0, 20)
    
    # D. Relevant domain score (max 20 pts)
    all_terms = [
        "retrieval", "search", "ranking", "recommendation", "reranking", "recall", "precision",
        "embedding", "embeddings", "vector db", "vector store", "faiss", "milvus", "pinecone", 
        "weaviate", "qdrant", "chroma", "opensearch", "elasticsearch",
        "production", "serving", "inference", "latency", "throughput", "pipeline", "mlflow", 
        "kubeflow", "feature store", "model registry",
        "ndcg", "mrr", "map", "a/b test", "offline eval", "online eval", "experiment", "metrics",
        "python"
    ]
    
    desc_sum = (df["all_job_descriptions"].fillna("") + " " + df["summary"].fillna("")).str.lower()
    domain_count = pd.Series(np.zeros(len(df)), index=df.index)
    
    for t in all_terms:
        if t == "a/b test":
            domain_count += desc_sum.str.contains(r"\ba/b test\b", regex=True).astype(int)
        else:
            domain_count += desc_sum.str.contains(rf"\b{t}\b", regex=True).astype(int)
            
    domain_score = np.clip(domain_count * 2, 0, 20)
    
    # FIX 6 — Production ML experience bonus (max 10 pts)
    PROD_ML_TERMS = [
        "deployed model", "model serving", "model in production",
        "a/b test", "online experiment", "feature store",
        "training pipeline", "inference", "latency",
        "ranking model", "recommendation system", "search system",
        "embedding-based retrieval", "learning-to-rank",
        "offline-online", "feature engineering", "feature pipeline"
    ]
    
    prod_ml_count = pd.Series(np.zeros(len(df)), index=df.index)
    for t in PROD_ML_TERMS:
        prod_ml_count += desc_sum.str.contains(re.escape(t), regex=True).astype(int)
    
    prod_ml_bonus = np.clip(prod_ml_count * 2, 0, 10)
    
    # E. Transition penalty (max -15 pts)
    summary = df["summary"].fillna("").str.lower()
    trans_pen = np.zeros(len(df))
    trans_pen = np.where(df["flag_transition_summary"] == True, trans_pen - 15, trans_pen)
    
    kaggle_cond = summary.str.contains(r"\bkaggle\b", regex=True) & (domain_count == 0)
    trans_pen = np.where(kaggle_cond, trans_pen - 5, trans_pen)
    
    proj_cond = summary.str.contains(r"side project|personal project", regex=True) & (domain_score < 6)
    trans_pen = np.where(proj_cond, trans_pen - 5, trans_pen)
    
    transition_penalty = np.clip(trans_pen, -15, 0)
    
    # COMBINE
    career_quality_raw = exp_score + product_score + action_score + domain_score + prod_ml_bonus + transition_penalty
    career_quality = pd.Series(np.clip(career_quality_raw, 0, 100), index=df.index)
    
    # Apply title cap: non-engineering candidates cannot exceed 20
    career_quality = career_quality.where(~title_cap_mask, career_quality.clip(upper=20))
    df["career_quality"] = career_quality
    
    print(f"Mean career_quality: {career_quality.mean():.4f}")
    print(f"Std dev career_quality: {career_quality.std():.4f}")
    
    top5_idx = np.argsort(career_quality)[::-1][:5]
    print("\nTop 5 candidate_ids by career_quality + score:")
    for i in top5_idx:
        print(f"  {df['candidate_id'].iloc[i]}: {career_quality.iloc[i]:.2f}")
        
    print(f"Count where career_quality < 20: {(career_quality < 20).sum()}")
    print(f"Count where transition_penalty < 0: {(transition_penalty < 0).sum()}")
    print(f"Count where title_cap_mask == True (Capped at 20 — no engineering title): {title_cap_mask.sum()}")
    print(f"Count where has_engineering_title == True: {has_engineering_title.sum()}")
    print(f"Count where prod_ml_bonus > 0: {(prod_ml_bonus > 0).sum()}")
    
    return df


def score_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing behavioral score...")
    
    # A. GitHub score
    g = df["github_score"].fillna(-1)
    github_pts = np.select(
        [g == -1, (g >= 0) & (g <= 20), (g >= 21) & (g <= 40), (g >= 41) & (g <= 60), (g >= 61) & (g <= 80), g >= 81],
        [0, 3, 8, 14, 20, 25],
        default=0
    )
    
    # B. Recruiter engagement score
    rrr = df["recruiter_response_rate"].fillna(0)
    resp_pts = np.select([rrr < 0.3, (rrr >= 0.3) & (rrr <= 0.6), rrr > 0.6], [0, 5, 10], default=0)
    sbr = df["saved_by_recruiters"].fillna(0)
    saved_pts = np.select([sbr == 0, (sbr >= 1) & (sbr <= 3), (sbr >= 4) & (sbr <= 10), sbr > 10], [0, 3, 6, 10], default=0)
    recruiter_pts = np.clip(resp_pts + saved_pts, 0, 20)
    
    # C. Activity score
    dsa = df["days_since_active"].fillna(999)
    activity_pts = np.select(
        [dsa <= 7, (dsa >= 8) & (dsa <= 30), (dsa >= 31) & (dsa <= 90), (dsa >= 91) & (dsa <= 180), dsa > 180],
        [15, 12, 8, 4, 0],
        default=0
    )
    
    # D. Profile quality score
    comp = df["completeness"].fillna(0)
    completeness_pts = np.select(
        [comp < 50, (comp >= 50) & (comp <= 70), (comp > 70) & (comp <= 85), comp > 85],
        [0, 5, 10, 15],
        default=0
    )
    
    # E. Availability score
    nd = df["notice_days"].fillna(999)
    notice_pts = np.select(
        [nd <= 15, (nd >= 16) & (nd <= 30), (nd >= 31) & (nd <= 60), (nd >= 61) & (nd <= 90), nd > 90],
        [15, 12, 8, 4, 0],
        default=0
    )
    open_bonus = np.where(df["open_to_work"] == True, 3, 0)
    reloc_bonus = np.where(df["willing_to_relocate"] == True, 2, 0)
    availability_pts = np.clip(notice_pts + open_bonus + reloc_bonus, 0, 15)
    
    # F. Verification bonus
    v_email = np.where(df["verified_email"] == True, 3, 0)
    v_phone = np.where(df["verified_phone"] == True, 3, 0)
    v_li = np.where(df["linkedin_connected"] == True, 4, 0)
    verification_pts = np.clip(v_email + v_phone + v_li, 0, 10)
    
    # COMBINE
    behavioral_raw = github_pts + recruiter_pts + activity_pts + completeness_pts + availability_pts + verification_pts
    behavioral = pd.Series(np.clip(behavioral_raw, 0, 100), index=df.index)
    df["behavioral"] = behavioral
    
    print(f"Mean behavioral score: {behavioral.mean():.4f}")
    print(f"Std dev behavioral score: {behavioral.std():.4f}")
    
    top5_idx = np.argsort(behavioral)[::-1][:5]
    print("\nTop 5 candidate_ids by behavioral + score:")
    for i in top5_idx:
        print(f"  {df['candidate_id'].iloc[i]}: {behavioral.iloc[i]:.2f}")
        
    print("\nDistribution:")
    bins = [-1, 25, 50, 75, 100]
    labels = ["0-25", "26-50", "51-75", "76-100"]
    dist = pd.cut(behavioral, bins=bins, labels=labels, right=True).value_counts().sort_index()
    for lbl, cnt in dist.items():
        print(f"  {lbl}: {cnt}")
        
    print(f"Count where github_pts > 0: {(github_pts > 0).sum()}")
    
    return df


def score_stability(df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing stability score...")
    
    # A. Longest tenure score
    ltm = df["longest_tenure_months"].fillna(0)
    tenure_pts = np.select(
        [ltm < 12, (ltm >= 12) & (ltm < 24), (ltm >= 24) & (ltm < 36), (ltm >= 36) & (ltm < 48), ltm >= 48],
        [0, 10, 20, 30, 40],
        default=0
    )
    
    # B. Job hop penalty
    tcm = df["total_career_months"].fillna(0)
    nj = df["num_jobs"].fillna(0)
    avg_tenure = tcm / np.maximum(nj, 1)
    hop_penalty = np.select(
        [avg_tenure >= 36, (avg_tenure >= 24) & (avg_tenure < 36), (avg_tenure >= 18) & (avg_tenure < 24), (avg_tenure >= 12) & (avg_tenure < 18), avg_tenure < 12],
        [0, -5, -10, -18, -30],
        default=0
    )
    
    # C. Career progression score
    titles = df["all_job_titles"].fillna("").str.lower()
    has_senior = titles.str.contains(r"(?:senior|sr\.|lead|principal|staff|head of|director|manager|architect)", regex=True)
    has_junior = titles.str.contains(r"(?:junior|jr\.|associate|intern|trainee|entry)", regex=True)
    
    nj_cond = df["num_jobs"] >= 2
    
    progression_pts = np.select(
        [has_senior & nj_cond, has_senior, ~has_senior & ~has_junior, has_junior],
        [30, 20, 15, 5],
        default=15
    )
    
    # D. Current employment bonus
    hcj = df["has_current_job"] == True
    current_pts = np.where(hcj, 10, 0)
    dsa = df["days_since_active"].fillna(999)
    active_bonus = np.where(hcj & (dsa <= 30), 5, 0)
    employment_pts = np.clip(current_pts + active_bonus, 0, 15)
    
    # E. Experience consistency check
    consistency_pts = np.where(df["flag_experience_mismatch"] == True, 0, 15)
    
    # COMBINE (before dampening)
    stability_raw = tenure_pts + hop_penalty + progression_pts + employment_pts + consistency_pts
    stability = pd.Series(np.clip(stability_raw, 0, 100), index=df.index)
    
    # DAMPENING by career quality
    mean_before = stability.mean()
    career_quality_ratio = (df["career_quality"] / 50.0).clip(upper=1.0)
    stability_dampened = (stability * career_quality_ratio).clip(0, 100)
    
    mean_after = stability_dampened.mean()
    reduced_gt_20 = ((stability - stability_dampened) > 20).sum()
    
    stability = stability_dampened
    df["stability"] = stability
    
    print(f"Mean stability BEFORE dampening: {mean_before:.4f}")
    print(f"Mean stability AFTER dampening: {mean_after:.4f}")
    print(f"Count where dampening reduced score by more than 20 points: {reduced_gt_20}")
    print(f"Mean stability score: {stability.mean():.4f}")
    print(f"Std dev stability score: {stability.std():.4f}")
    
    top5_idx = np.argsort(stability)[::-1][:5]
    print("\nTop 5 candidate_ids by stability + score:")
    for i in top5_idx:
        print(f"  {df['candidate_id'].iloc[i]}: {stability.iloc[i]:.2f}")
        
    print(f"Count where hop_penalty < -15: {(hop_penalty < -15).sum()}")
    print(f"Count where progression_pts == 30: {(progression_pts == 30).sum()}")
    
    return df


def apply_honeypot_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Honeypot gate with summary-level detection and tightened transition handling."""
    print("\nApplying honeypot gate...")
    
    NON_TECHNICAL_TITLES = [
        "marketing", "accountant", "hr manager", "human resources",
        "content writer", "graphic designer", "civil engineer",
        "mechanical engineer", "sales executive", "sales manager",
        "customer support", "customer service", "business development",
        "recruiter", "operations manager", "project coordinator",
        "supply chain", "finance manager", "legal", "lawyer",
        "teacher", "professor", "doctor", "nurse", "pharmacist"
    ]
    
    RETRIEVAL_ML_TERMS = [
        "retrieval", "embedding", "embeddings", "vector", "ranking",
        "recommendation", "search", "nlp", "machine learning", "ml model",
        "neural", "transformer", "bert", "faiss", "milvus", "pinecone",
        "weaviate", "qdrant", "elasticsearch", "opensearch",
        "sentence transformer", "fine-tun", "inference", "model serving",
        "feature store", "mlflow", "kubeflow", "xgboost", "lightgbm",
        "deep learning", "pytorch", "tensorflow", "scikit"
    ]
    
    # FIX 3 — Removed bare "analyst", added "data analyst"
    ENGINEERING_TITLE_KEYWORDS = [
        "engineer", "scientist", "developer", "architect", "researcher",
        "data analyst", "ml", "ai", "data", "nlp", "sde", "swe", "tech lead",
        "backend", "fullstack", "full stack", "platform", "infrastructure",
        "devops", "mlops", "applied"
    ]
    
    # FIX 4 — Summary-level non-technical detection
    NON_TECHNICAL_SUMMARY_PHRASES = [
        "marketing manager", "sales manager", "operations manager",
        "hr manager", "human resources", "content writer",
        "graphic designer", "accountant", "customer support",
        "customer service", "business development", "recruiter"
    ]
    
    # 2a — Title-based non-technical
    is_non_technical_title = df["current_title"].str.lower().str.contains(
        '|'.join(NON_TECHNICAL_TITLES), na=False
    )
    
    # FIX 4 — Summary-based non-technical
    is_non_technical_summary = df["summary"].fillna("").str.lower().str.contains(
        '|'.join(NON_TECHNICAL_SUMMARY_PHRASES), na=False
    )
    
    # 2b — Retrieval/ML in career
    has_retrieval_in_career = df["all_job_descriptions"].fillna("").str.lower().str.contains(
        '|'.join(RETRIEVAL_ML_TERMS), na=False
    )
    
    # 2c — Engineering title in any position
    # Prefer the signal computed earlier in `score_career_quality` to avoid drift.
    if "has_tech_engineering_title" in df.columns:
        has_engineering_title = df["has_tech_engineering_title"]
    else:
        # Fallback: compute using the same tech-vs-nontech logic so this
        # function is safe to call standalone.
        NON_TECH_ENGINEER_TITLES = [
            "mechanical engineer", "civil engineer", "electrical engineer",
            "chemical engineer", "structural engineer", "industrial engineer",
            "site engineer", "sales engineer"
        ]

        TECH_ENGINEER_KEYWORDS = [
            "software engineer", "ml engineer", "ai engineer", "data engineer",
            "machine learning engineer", "platform engineer", "backend engineer",
            "frontend engineer", "full stack engineer", "fullstack engineer",
            "infrastructure engineer", "devops engineer", "mlops engineer",
            "search engineer", "applied scientist", "research scientist",
            "data scientist", "nlp engineer", "sde", "swe", "tech lead",
            "applied ml", "architect", "developer"
        ]

        all_titles_combined = (df["current_title"].fillna("") + " " + df["all_job_titles"].fillna("")).str.lower()
        has_non_tech_engineer = all_titles_combined.str.contains(
            '|'.join(NON_TECH_ENGINEER_TITLES), na=False
        )
        has_tech_title = all_titles_combined.str.contains(
            '|'.join(TECH_ENGINEER_KEYWORDS), na=False
        )
        has_engineering_title = has_tech_title & ~(has_non_tech_engineer & ~has_tech_title)
    
    # 2d, 2e
    career_quality_low = df["career_quality"] < 25
    career_quality_very_low = df["career_quality"] < 15
    
    # 3. Apply Multipliers
    multiplier = pd.Series(np.ones(len(df)), index=df.index)
    
    # --- Hard elimination (0.0) ---
    rule_0_a = is_non_technical_title & career_quality_low
    rule_0_b = (~has_engineering_title) & career_quality_very_low
    rule_0_c = df["flag_keyword_stuffer"] & career_quality_low
    # FIX 4 — Summary-based honeypot: non-technical summary AND no retrieval/ML in career
    rule_0_d = is_non_technical_summary & (~has_retrieval_in_career)
    # FIX 5 — Transition + no ML experience = hard eliminate
    rule_0_e = (df["flag_transition_summary"] == True) & (~has_retrieval_in_career)
    
    mask_0 = rule_0_a | rule_0_b | rule_0_c | rule_0_d | rule_0_e
    multiplier[mask_0] = 0.0
    
    # --- Soft penalty (0.4) — only where multiplier is still 1.0 ---
    rule_04_a = df["flag_consulting_only"] == True
    # FIX 5 — Tightened threshold from 35 to 45
    rule_04_b = (df["flag_transition_summary"] == True) & (df["career_quality"] < 45)
    rule_04_c = (df["years_of_experience"] > 12) & career_quality_low
    rule_04_d = (~has_retrieval_in_career) & (df["career_quality"] < 30)
    
    mask_04 = rule_04_a | rule_04_b | rule_04_c | rule_04_d
    multiplier[(multiplier == 1.0) & mask_04] = 0.4
    
    df["honeypot_multiplier"] = multiplier
    
    print(f"Count where honeypot_multiplier == 0.0 (Hard eliminated): {(multiplier == 0.0).sum()}")
    print(f"Count where honeypot_multiplier == 0.4 (Soft penalized): {(multiplier == 0.4).sum()}")
    print(f"Count where honeypot_multiplier == 1.0 (Clean candidates): {(multiplier == 1.0).sum()}")
    print(f"  Rule 0a (non-tech title + low career): {rule_0_a.sum()}")
    print(f"  Rule 0b (no eng title + very low career): {rule_0_b.sum()}")
    print(f"  Rule 0c (keyword stuffer + low career): {rule_0_c.sum()}")
    print(f"  Rule 0d (non-tech summary + no ML): {rule_0_d.sum()}")
    print(f"  Rule 0e (transition + no ML): {rule_0_e.sum()}")
    
    eliminated_ids = df.loc[multiplier == 0.0, "candidate_id"].tolist()
    if len(df) <= 50:
        print(f"\nHard eliminated candidate_ids: {eliminated_ids}")
    else:
        print(f"\nHard eliminated candidate_ids (first 20): {eliminated_ids[:20]}")
    
    return df


def compute_final_score(df: pd.DataFrame) -> pd.DataFrame:
    df["final_score"] = (
        df["jd_match"]       * 0.35 +
        df["career_quality"] * 0.25 +
        df["behavioral"]     * 0.25 +
        df["stability"]      * 0.15
    ) * df["honeypot_multiplier"]
    return df


def save_outputs(df: pd.DataFrame, base_dir: Path):
    out_dir = base_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "scored_candidates.pkl"
    print(f"\nSaving scored candidates to {out_path}...")
    df.to_pickle(out_path)


def main(use_sample: bool = False):
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    df, embeddings, jd_embedding, jd_chunk_embeddings, bm25_index, jd_tokens = load_data(base_dir, use_sample)
    
    print("\n--- 1. JD MATCH ---")
    df = score_jd_match(df, embeddings, jd_embedding, jd_chunk_embeddings, bm25_index, jd_tokens)
    
    print("\n--- 2. CAREER QUALITY ---")
    df = score_career_quality(df)
    
    print("\n--- 3. BEHAVIORAL ---")
    df = score_behavioral(df)
    
    print("\n--- 4. STABILITY ---")
    df = score_stability(df)
    
    print("\n--- 5. HONEYPOT GATE ---")
    df = apply_honeypot_gate(df)
    
    df = compute_final_score(df)
    
    print("\n--- FINAL SUMMARY ---")
    print(f"Mean final_score across all candidates: {df['final_score'].mean():.4f}")
    print(f"Std dev final_score: {df['final_score'].std():.4f}")
    print(f"Count where final_score == 0.0: {(df['final_score'] == 0.0).sum()}")
    print(f"Count where final_score > 50: {(df['final_score'] > 50).sum()}")
    
    print("\nTop 10 candidate_ids by final_score:")
    print(f"{'candidate_id':<15} | {'final':<6} | {'jd_match':<8} | {'career':<6} | {'behavioral':<10} | {'stability':<9} | {'honeypot':<7}")
    print("-" * 80)
    
    top10_idx = np.argsort(df["final_score"])[::-1][:10]
    for i in top10_idx:
        row = df.iloc[i]
        print(f"{row['candidate_id']:<15} | {row['final_score']:<6.2f} | {row['jd_match']:<8.2f} | {row['career_quality']:<6.2f} | {row['behavioral']:<10.2f} | {row['stability']:<9.2f} | {row['honeypot_multiplier']:<7.1f}")
        
    print("\nScore distribution buckets:")
    bins = [-1, 20, 40, 60, 80, 100]
    labels = ["0-20", "21-40", "41-60", "61-80", "81-100"]
    dist = pd.cut(df["final_score"], bins=bins, labels=labels, right=True).value_counts().sort_index()
    for lbl, cnt in dist.items():
        print(f"  {lbl}: {cnt}")
    
    save_outputs(df, base_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score candidates across all features.")
    parser.add_argument("--sample", action="store_true", help="Run on first 50 rows only")
    args = parser.parse_args()
    
    main(args.sample)
