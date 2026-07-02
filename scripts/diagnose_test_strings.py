from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent.parent
proc = base / 'processed'

print('Loading model...')
model = SentenceTransformer('all-MiniLM-L6-v2')

jd_emb = np.load(proc / 'jd_embedding.npy')

tests = [
    'Senior AI Engineer with production experience in embeddings, retrieval, and ranking systems',
    'Marketing Manager with experience in sales and customer outreach',
    'Machine Learning Engineer experienced in feature stores, model serving, inference, and A/B testing',
    'Experienced HR professional managing recruitment and operations',
]

print('\nEncoding test strings and comparing to JD embedding...')
embs = model.encode(tests, convert_to_numpy=True, normalize_embeddings=True)

for t, e in zip(tests, embs):
    sim = float(np.dot(e.astype(np.float32), jd_emb.astype(np.float32)))
    print(f"Sim to JD for '{t[:60]}...': {sim:.6f}")

print('\nAlso compute self-similarity of JD (sanity):', float(np.dot(jd_emb, jd_emb)))
