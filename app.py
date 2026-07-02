"""
Lightweight Gradio demo for the Redrob Candidate Ranker.

This file is intentionally a wrapper around the existing pipeline modules. It
uses a demo-only cache under processed/gradio_demo/ and writes demo submissions
under outputs/gradio_demo/ so it does not overwrite competition artifacts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DEMO_PROCESSED_DIR = BASE_DIR / "processed" / "gradio_demo"
DEMO_OUTPUTS_DIR = BASE_DIR / "outputs" / "gradio_demo"

DEFAULT_CANDIDATES = BASE_DIR / "sample_candidates.json"
DEFAULT_JD = BASE_DIR / "job_description.md"
FALLBACK_JD = BASE_DIR / "jd.txt"
MAX_DEMO_CANDIDATES = 100
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MODEL_NAME = "all-MiniLM-L6-v2"


def import_script(module_name: str, script_name: str) -> Any:
    """Import an existing script with a filename that is not a valid module name."""
    script_path = SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess = import_script("redrob_preprocess", "01_preprocess.py")
embed = import_script("redrob_embed", "02_embed.py")
score = import_script("redrob_score", "03_score.py")
submission = import_script("redrob_submission", "04_generate_submission.py")


def resolve_upload_path(uploaded_file: Any, default_path: Path) -> Path:
    """Return the Gradio upload path, or a repo default when no upload is given."""
    if uploaded_file is None:
        return default_path
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    return Path(uploaded_file.name)


def default_jd_path() -> Path:
    """Prefer the requested markdown default, falling back to the repo JD text."""
    return DEFAULT_JD if DEFAULT_JD.exists() else FALLBACK_JD


def load_demo_candidates(path: Path) -> list[dict]:
    """Load at most 100 candidates and reject the full competition dataset."""
    if path.resolve() == (BASE_DIR / "candidates.jsonl").resolve():
        raise ValueError("The demo refuses candidates.jsonl. Use sample_candidates.json or upload <=100 candidates.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Candidate file is too large for the demo. Upload a sample file with <=100 candidates.")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        candidates = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                if len(candidates) >= MAX_DEMO_CANDIDATES:
                    raise ValueError("Candidate JSONL contains more than 100 records; the demo only supports samples.")
                candidates.append(json.loads(line))
        return candidates

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Candidate JSON must be an array of candidate objects.")
    if len(data) > MAX_DEMO_CANDIDATES:
        raise ValueError("Candidate JSON contains more than 100 records; the demo only supports samples.")
    return data


def load_job_description(path: Path) -> str:
    """Load a plain-text or markdown job description."""
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Please upload a .txt or .md job description for the demo.")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Job description file is empty.")
    return text


def cache_key(candidate_path: Path, jd_text: str) -> str:
    """Create a cache key from the sample input file and JD contents."""
    digest = hashlib.sha256()
    digest.update(candidate_path.read_bytes())
    digest.update(jd_text.encode("utf-8"))
    digest.update(MODEL_NAME.encode("utf-8"))
    return digest.hexdigest()[:16]


def preprocess_candidates(raw_candidates: list[dict], cache_dir: Path) -> pd.DataFrame:
    """Reuse the existing flattening and sanity-flag logic."""
    df = pd.DataFrame([preprocess.flatten_candidate(candidate) for candidate in raw_candidates])
    df = preprocess.add_sanity_flags(df)
    df.to_pickle(cache_dir / "candidates_clean.pkl")
    df.head(200).to_csv(cache_dir / "preview.csv", index=False)
    return df


def load_or_create_artifacts(df: pd.DataFrame, jd_text: str, cache_dir: Path):
    """Generate or load sample embeddings and BM25 artifacts."""
    embeddings_path = cache_dir / "embeddings.npy"
    jd_embedding_path = cache_dir / "jd_embedding.npy"
    bm25_path = cache_dir / "bm25_index.pkl"
    jd_tokens_path = cache_dir / "jd_tokens.pkl"
    jd_chunk_embeddings_path = cache_dir / "jd_chunk_embeddings.npy"
    jd_chunk_texts_path = cache_dir / "jd_chunk_texts.pkl"

    if embeddings_path.exists() and jd_embedding_path.exists():
        embeddings = np.load(embeddings_path)
        jd_embedding = np.load(jd_embedding_path)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
        embeddings = embed.compute_candidate_embeddings(df["candidate_full_text"].tolist(), model)
        jd_embedding = embed.compute_jd_embedding(jd_text, model)
        jd_chunks, jd_chunk_embeddings = embed.compute_jd_chunk_embeddings(jd_text, model)
        np.save(embeddings_path, embeddings)
        np.save(jd_embedding_path, jd_embedding)
        np.save(jd_chunk_embeddings_path, jd_chunk_embeddings)
        with jd_chunk_texts_path.open("wb") as f:
            pickle.dump(jd_chunks, f)

    if bm25_path.exists() and jd_tokens_path.exists():
        with bm25_path.open("rb") as f:
            bm25_index = pickle.load(f)
        with jd_tokens_path.open("rb") as f:
            jd_tokens = pickle.load(f)
    else:
        corpus = [embed.tokenize(text) for text in df["candidate_full_text"].tolist()]
        bm25_index = embed.build_bm25_index(corpus)
        jd_tokens = embed.tokenize(jd_text)
        with bm25_path.open("wb") as f:
            pickle.dump(bm25_index, f)
        with jd_tokens_path.open("wb") as f:
            pickle.dump(jd_tokens, f)

    jd_chunk_embeddings = np.load(jd_chunk_embeddings_path) if jd_chunk_embeddings_path.exists() else None
    return embeddings, jd_embedding, jd_chunk_embeddings, bm25_index, jd_tokens


def run_existing_scoring_pipeline(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    jd_chunk_embeddings: np.ndarray | None,
    bm25_index: Any,
    jd_tokens: list[str],
) -> pd.DataFrame:
    """Call the existing scoring functions in the same order as scripts/03_score.py."""
    scored = df.copy()
    scored = score.score_jd_match(scored, embeddings, jd_embedding, jd_chunk_embeddings, bm25_index, jd_tokens)
    scored = score.score_career_quality(scored)
    scored = score.score_behavioral(scored)
    scored = score.score_stability(scored)
    scored = score.apply_honeypot_gate(scored)
    scored = score.compute_final_score(scored)
    return scored.sort_values(["final_score", "candidate_id"], ascending=[False, True], kind="mergesort")


def write_demo_submission(ranked: pd.DataFrame, output_dir: Path) -> Path:
    """Use the existing submission writer without enforcing the competition top-100 size."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.csv"
    export = ranked.copy()
    export.insert(0, "rank", range(1, len(export) + 1))
    submission.write_submission(export, output_path)
    return output_path


def display_table(ranked: pd.DataFrame) -> pd.DataFrame:
    """Build the UI table requested by the challenge demo."""
    table = ranked.reset_index(drop=True).copy()
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table.rename(
        columns={
            "candidate_id": "Candidate ID",
            "name": "Candidate Name",
            "final_score": "Final Score",
            "jd_match": "JD Match",
            "career_quality": "Career Quality",
            "behavioral": "Behavioral Score",
            "stability": "Stability Score",
        }
    )[
        [
            "Rank",
            "Candidate ID",
            "Candidate Name",
            "Final Score",
            "JD Match",
            "Career Quality",
            "Behavioral Score",
            "Stability Score",
        ]
    ].round(3)


def run_ranking(candidate_file: Any, jd_file: Any):
    """Gradio button handler."""
    start = time.perf_counter()
    candidate_path = resolve_upload_path(candidate_file, DEFAULT_CANDIDATES)
    jd_path = resolve_upload_path(jd_file, default_jd_path())

    raw_candidates = load_demo_candidates(candidate_path)
    jd_text = load_job_description(jd_path)

    run_key = cache_key(candidate_path, jd_text)
    cache_dir = DEMO_PROCESSED_DIR / run_key
    output_dir = DEMO_OUTPUTS_DIR / run_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = cache_dir / "candidates_clean.pkl"
    if clean_path.exists():
        df = pd.read_pickle(clean_path)
    else:
        df = preprocess_candidates(raw_candidates, cache_dir)

    embeddings, jd_embedding, jd_chunk_embeddings, bm25_index, jd_tokens = load_or_create_artifacts(df, jd_text, cache_dir)
    ranked = run_existing_scoring_pipeline(df, embeddings, jd_embedding, jd_chunk_embeddings, bm25_index, jd_tokens)
    ranked.to_pickle(output_dir / "scored_candidates.pkl")
    submission_path = write_demo_submission(ranked, output_dir)

    runtime = time.perf_counter() - start
    status = (
        "Ranking complete. "
        f"Processed {len(ranked)} candidates in {runtime:.2f} seconds."
    )
    metrics = f"Total runtime: {runtime:.2f}s\nCandidates processed: {len(ranked)}"
    return display_table(ranked), str(submission_path), metrics, status


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown("# Redrob Candidate Ranker")
    gr.Markdown(
        "A hybrid AI-powered candidate ranking system for ranking candidates "
        "against a Senior AI Engineer job description."
    )

    with gr.Row():
        candidate_input = gr.File(
            label="Upload Candidate JSON (default: sample_candidates.json)",
            file_types=[".json", ".jsonl"],
            value=str(DEFAULT_CANDIDATES) if DEFAULT_CANDIDATES.exists() else None,
        )
        jd_input = gr.File(
            label="Upload Job Description (default: job_description.md)",
            file_types=[".md", ".txt"],
            value=str(default_jd_path()) if default_jd_path().exists() else None,
        )

    run_button = gr.Button("Run Ranking", variant="primary")
    status_output = gr.Textbox(label="Status", interactive=False)
    metrics_output = gr.Textbox(label="Run Details", interactive=False)
    results_output = gr.Dataframe(label="Ranked Candidates", interactive=False, wrap=True)
    download_output = gr.File(label="Download submission.csv")

    run_button.click(
        fn=run_ranking,
        inputs=[candidate_input, jd_input],
        outputs=[results_output, download_output, metrics_output, status_output],
    )


if __name__ == "__main__":
    demo.launch()
