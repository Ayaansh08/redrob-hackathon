"""
02_embed.py

Computes embeddings and BM25 index for the preprocessed candidate data.
"""

import os
import time
import argparse
import re
import pickle
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for",
    "of","with","is","was","are","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","i","my","we","our","you","your",
    "it","its","this","that","these","those","not","no","by",
    "as","from","into","through","during","about","than","then"
}

def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    
    Args:
        text: Input text string.
        
    Returns:
        List of cleaned tokens.
    """
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]

def load_inputs(use_sample: bool) -> tuple[pd.DataFrame, str]:
    """
    Load candidate dataframe and job description text.
    
    Args:
        use_sample: If True, returns only the first 50 rows.
        
    Returns:
        Tuple of (dataframe, jd_text)
    """
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    pkl_path = base_dir / "processed" / "candidates_clean.pkl"
    jd_path = base_dir / "jd.txt"
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}. Run 01_preprocess.py first.")
        
    df = pd.read_pickle(pkl_path)
    if use_sample:
        df = df.head(50).copy()
        
    if not jd_path.exists():
        raise FileNotFoundError(f"Missing {jd_path}.")
        
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read().strip()
        
    if not jd_text:
        raise ValueError("jd.txt is empty. Please provide a job description.")
        
    return df, jd_text

def compute_candidate_embeddings(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Compute embeddings for candidate texts.
    
    Args:
        texts: List of candidate full text strings.
        model: SentenceTransformer model.
        
    Returns:
        Numpy array of shape (N, 384) in float32.
    """
    clean_texts = [
        t if isinstance(t, str) and t.strip() else "no information provided" 
        for t in texts
    ]
    
    embeddings = model.encode(
        clean_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embeddings.astype(np.float32)

def compute_jd_embedding(jd_text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Compute embedding for the job description.
    
    Args:
        jd_text: Job description string.
        model: SentenceTransformer model.
        
    Returns:
        Numpy array of shape (384,) in float32.
    """
    emb = model.encode(
        [jd_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    
    emb_f32 = emb.astype(np.float32)
    
    # Sanity check: dot product with itself
    sim = np.dot(emb_f32, emb_f32)
    print(f"JD self-similarity (should be ~1.0): {sim:.4f}")
    
    return emb_f32


def chunk_jd_text(jd_text: str, max_words_per_chunk: int = 300) -> list[str]:
    """Split JD into semantically coherent chunks suitable for embedding.

    Strategy: split on double-newline paragraphs, then accumulate paragraphs
    until reaching roughly max_words_per_chunk, producing multiple chunks.
    """
    paras = [p.strip() for p in jd_text.split('\n\n') if p.strip()]
    chunks = []
    cur = []
    cur_words = 0
    for p in paras:
        pw = len(p.split())
        if cur_words + pw > max_words_per_chunk and cur:
            chunks.append(' '.join(cur))
            cur = [p]
            cur_words = pw
        else:
            cur.append(p)
            cur_words += pw

    if cur:
        chunks.append(' '.join(cur))

    # As a fallback ensure at least one chunk
    if not chunks:
        chunks = [jd_text[:max_words_per_chunk * 10]]

    return chunks


def compute_jd_chunk_embeddings(jd_text: str, model: SentenceTransformer, max_words_per_chunk: int = 300):
    chunks = chunk_jd_text(jd_text, max_words_per_chunk=max_words_per_chunk)
    print(f"JD split into {len(chunks)} chunks for embedding")
    emb = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    emb_f32 = emb.astype(np.float32)
    return chunks, emb_f32

def build_bm25_index(corpus: list[list[str]]) -> BM25Okapi:
    """
    Build BM25 index from tokenized corpus.
    
    Args:
        corpus: List of lists of tokens.
        
    Returns:
        Fitted BM25Okapi object.
    """
    return BM25Okapi(corpus)

def run_sanity_check(df: pd.DataFrame, embeddings: np.ndarray, jd_embedding: np.ndarray, bm25_index: BM25Okapi, jd_tokens: list[str]) -> None:
    """
    Run quick sanity check between cosine similarity and BM25 scores.
    """
    print("\n--- SANITY CHECK ---")
    scores = embeddings @ jd_embedding
    
    # Cosine top and bottom 5
    top_indices = np.argsort(scores)[::-1]
    
    print("\nTop 5 by Cosine Similarity:")
    for idx in top_indices[:5]:
        c_id = df.iloc[idx]["candidate_id"]
        print(f"  {c_id}: {scores[idx]:.4f}")
        
    print("\nBottom 5 by Cosine Similarity:")
    for idx in top_indices[-5:][::-1]:
        c_id = df.iloc[idx]["candidate_id"]
        print(f"  {c_id}: {scores[idx]:.4f}")
        
    print(f"\nMean Similarity: {np.mean(scores):.4f}")
    print(f"Std Dev Similarity: {np.std(scores):.4f}")
    
    # BM25 top 5
    bm25_scores = bm25_index.get_scores(jd_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[::-1]
    
    print("\nTop 5 by BM25 Score:")
    for idx in bm25_top_indices[:5]:
        c_id = df.iloc[idx]["candidate_id"]
        print(f"  {c_id}: {bm25_scores[idx]:.4f}")
        
    top_5_cosine = set(df.iloc[top_indices[:5]]["candidate_id"])
    top_5_bm25 = set(df.iloc[bm25_top_indices[:5]]["candidate_id"])
    overlap = len(top_5_cosine.intersection(top_5_bm25))
    
    print(f"\nOverlap between top-5 cosine and top-5 BM25: {overlap}/5")
    print("--------------------\n")

def main(use_sample: bool) -> None:
    """Main execution function."""
    start_time = time.time()
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    emb_path = processed_dir / "embeddings.npy"
    jd_emb_path = processed_dir / "jd_embedding.npy"
    bm25_path = processed_dir / "bm25_index.pkl"
    jd_tokens_path = processed_dir / "jd_tokens.pkl"
    
    print("Loading inputs...")
    df, jd_text = load_inputs(use_sample)
    
    # Ask about embeddings
    recompute_emb = True
    if emb_path.exists() and jd_emb_path.exists():
        ans = input("Embeddings already exist. Recompute? [y/N]: ").strip().lower()
        if ans not in ('y', 'yes'):
            recompute_emb = False
            
    if recompute_emb:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer (all-MiniLM-L6-v2) on device={device}...")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        print(f"Embedding {len(df)} candidates...")
        emb_start = time.time()
        texts = df["candidate_full_text"].tolist()
        embeddings = compute_candidate_embeddings(texts, model)
        emb_time = time.time() - emb_start
        print(f"Candidate embedding took {emb_time:.2f} seconds.")
        
        print("Embedding JD (full)...")
        jd_embedding = compute_jd_embedding(jd_text, model)

        # Also compute chunked JD embeddings for more robust matching
        try:
            print("Embedding JD chunks...")
            jd_chunks, jd_chunk_embeddings = compute_jd_chunk_embeddings(jd_text, model)
        except Exception as e:
            print(f"Failed to compute JD chunks: {e}")
            jd_chunks = None
            jd_chunk_embeddings = None
        
        print(f"Saving embeddings to {emb_path} and {jd_emb_path}...")
        np.save(emb_path, embeddings)
        np.save(jd_emb_path, jd_embedding)
        if jd_chunk_embeddings is not None:
            np.save(processed_dir / 'jd_chunk_embeddings.npy', jd_chunk_embeddings)
            with open(processed_dir / 'jd_chunk_texts.pkl', 'wb') as f:
                pickle.dump(jd_chunks, f)
        print(f"Embeddings saved. Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    else:
        print("Loading existing embeddings from disk...")
        embeddings = np.load(emb_path)
        jd_embedding = np.load(jd_emb_path)
        
    # Ask about BM25
    recompute_bm25 = True
    if bm25_path.exists() and jd_tokens_path.exists():
        ans = input("BM25 index already exists. Recompute? [y/N]: ").strip().lower()
        if ans not in ('y', 'yes'):
            recompute_bm25 = False
            
    if recompute_bm25:
        print(f"Tokenizing {len(df)} candidates...")
        bm25_start = time.time()
        corpus = [tokenize(t) for t in df["candidate_full_text"].tolist()]
        
        print("Building BM25 index...")
        bm25_index = build_bm25_index(corpus)
        bm25_time = time.time() - bm25_start
        print(f"BM25 building took {bm25_time:.2f} seconds.")
        
        jd_tokens = tokenize(jd_text)
        
        print(f"Saving BM25 index to {bm25_path}...")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_index, f)
            
        with open(jd_tokens_path, "wb") as f:
            pickle.dump(jd_tokens, f)
            
        avg_doc_len = np.mean([len(doc) for doc in corpus])
        print(f"BM25 index saved. Corpus size: {len(corpus)}, Avg doc length: {avg_doc_len:.1f}")
    else:
        print("Loading existing BM25 index from disk...")
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)
        with open(jd_tokens_path, "rb") as f:
            jd_tokens = pickle.load(f)

    # Sanity Check
    run_sanity_check(df, embeddings, jd_embedding, bm25_index, jd_tokens)
    
    total_time = time.time() - start_time
    
    print("\nOUTPUT SUMMARY")
    print("--------------")
    print(f"  Candidates embedded : {len(df)}")
    print(f"  Embedding shape     : {embeddings.shape}")
    print(f"  JD embedding shape  : {jd_embedding.shape}")
    print(f"  BM25 corpus size    : {bm25_index.corpus_size} documents")
    print(f"  BM25 avg doc length : {bm25_index.avgdl:.1f} tokens")
    print("  All artifacts saved to processed/")
    print(f"Total runtime: {total_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed and BM25 index candidates.")
    parser.add_argument("--sample", action="store_true", help="Run on first 50 rows only")
    args = parser.parse_args()
    
    main(args.sample)
