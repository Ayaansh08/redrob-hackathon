from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent.parent
proc = base / 'processed'

# load
jd_path = base / 'jd.txt'
with open(jd_path, 'r', encoding='utf-8') as f:
    jd_text = f.read()

df_emb = np.load(proc / 'embeddings.npy')

model = SentenceTransformer('all-MiniLM-L6-v2')

# naive extraction: find 'The skills inventory' and capture until 'Things we'd like' or 'Things we explicitly do NOT want'
start_kw = 'the skills inventory'
end_kw = "things we'd like"

low = jd_text.lower().find(start_kw)
if low == -1:
    # fallback to 'What you'd actually be doing'
    low = jd_text.lower().find('the skills inventory')

if low != -1:
    snippet = jd_text[low:]
    hi = snippet.lower().find(end_kw)
    if hi != -1:
        snippet = snippet[:hi]
else:
    # fallback: use first 800 chars after the intro
    snippet = jd_text[:800]

snippet = snippet.strip()
print('Excerpt length words:', len(snippet.split()))
print('\nExcerpt preview:\n', snippet[:800])

# embed snippet and compute similarity distribution against precomputed embeddings
snippet_emb = model.encode([snippet], convert_to_numpy=True, normalize_embeddings=True)[0]

sims = df_emb @ snippet_emb
p5, p95 = np.percentile(sims, [5,95])
print('\nSnippet cosine p5, p95:', p5, p95)
print('Mean, std:', sims.mean(), sims.std())

# also compute for full jd (load existing jd_embedding)
jd_emb = np.load(proc / 'jd_embedding.npy')
sims_full = df_emb @ jd_emb
p5f, p95f = np.percentile(sims_full, [5,95])
print('\nFull JD cosine p5, p95:', p5f, p95f)
print('Mean, std full:', sims_full.mean(), sims_full.std())

# display top 5 candidate ids for snippet vs full
import pandas as pd
candidates = pd.read_pickle(proc.parent / 'processed' / 'candidates_clean.pkl')

print('\nTop 5 for snippet:')
for i in np.argsort(sims)[::-1][:5]:
    print(candidates.iloc[i]['candidate_id'], sims[i])

print('\nTop 5 for full JD:')
for i in np.argsort(sims_full)[::-1][:5]:
    print(candidates.iloc[i]['candidate_id'], sims_full[i])
