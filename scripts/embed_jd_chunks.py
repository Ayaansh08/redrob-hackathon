from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import pickle

base = Path(__file__).resolve().parent.parent
proc = base / 'processed'

jd_path = base / 'jd.txt'
with open(jd_path, 'r', encoding='utf-8') as f:
    jd_text = f.read()

print('Loading model...')
model = SentenceTransformer('all-MiniLM-L6-v2')

# chunking helper (split on paragraphs and group to ~max words)
def chunk_jd_text(jd_text: str, max_words_per_chunk: int = 300):
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
    if not chunks:
        chunks = [jd_text[:max_words_per_chunk * 10]]
    return chunks

chunks = chunk_jd_text(jd_text)
print(f'Embedding {len(chunks)} JD chunks...')
embs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True, batch_size=32)
embs_f32 = embs.astype(np.float32)

np.save(proc / 'jd_chunk_embeddings.npy', embs_f32)
with open(proc / 'jd_chunk_texts.pkl', 'wb') as f:
    pickle.dump(chunks, f)

# also save mean jd_embedding for backwards compatibility
mean_emb = embs_f32.mean(axis=0)
mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
np.save(proc / 'jd_embedding.npy', mean_emb.astype(np.float32))

print('Saved jd_chunk_embeddings.npy and jd_chunk_texts.pkl and updated jd_embedding.npy')
