import numpy as np
import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent.parent
proc = base / "processed"

print('Loading processed data...')
df = pd.read_pickle(proc / 'candidates_clean.pkl')
emb = np.load(proc / 'embeddings.npy')
jd_emb = np.load(proc / 'jd_embedding.npy')

print(f'df shape: {df.shape}')
print('columns:', df.columns.tolist())

# JD text
jd_path = base / 'jd.txt'
with open(jd_path, 'r', encoding='utf-8') as f:
    jd_text = f.read()

print('\nJD text length (chars):', len(jd_text))
print('JD text length (words):', len(jd_text.split()))
print('\nJD preview (first 800 chars):')
print(jd_text[:800])

# Show if 'Final note' exists
if 'final note' in jd_text.lower() or 'final note for the participants' in jd_text.lower():
    print('\nFound final note phrase in JD text (possible meta commentary).')

# Embedding stats
scores = emb @ jd_emb
import numpy as np
p5, p95 = np.percentile(scores, [5,95])
print('\nCosine raw p5, p95:', p5, p95)
print('Mean, std:', scores.mean(), scores.std())

# Show top candidate indices by cosine and their candidate_full_text lengths
top_idx = np.argsort(scores)[::-1]
print('\nTop 10 candidates by cosine:')
for i in top_idx[:10]:
    cid = df.iloc[i]['candidate_id']
    text = df.iloc[i]['candidate_full_text']
    print(f"{cid}: score={scores[i]:.6f}, text_len_chars={len(text) if isinstance(text,str) else 0}, words={len(text.split()) if isinstance(text,str) else 0}")

# Find candidates mentioning 'recommend'/'search'/'embedding' in their text
keywords = ['recommend', 'search', 'embedding', 'retrieval', 'ranking', 'nlp']
print('\nCandidates mentioning recommendation/search terms (first 20):')
mask = df['candidate_full_text'].fillna('').str.lower().str.contains('|'.join(keywords))
matched = df[mask]
print('Count:', matched.shape[0])
for _, row in matched.head(20).iterrows():
    print(row['candidate_id'], 'title=>', row.get('current_title',''), '--- first 200 chars =>', (row.get('candidate_full_text','')[:200].replace('\n',' ')))

# Show example of one 'best' matching candidate text fully (if any matched)
if matched.shape[0] > 0:
    idx = matched.index[0]
    print('\nExample matched candidate full text (first 2000 chars):')
    print(df.loc[idx,'candidate_full_text'][:2000])

# Also show candidate summaries distribution of lengths
lens = df['candidate_full_text'].fillna('').str.len()
print('\nCandidate full_text length stats (chars): min, median, max ->', lens.min(), lens.median(), lens.max())

print('\nDone.')
