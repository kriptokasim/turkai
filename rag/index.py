# rag/index.py
import os
import argparse
import pickle
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    raise SystemExit("faiss-cpu kurulu olmalı: pip install faiss-cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--docs", default="./data/docs")
parser.add_argument("--out", default="./vectorstore")
parser.add_argument("--embedding", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

INDEX_PATH = os.path.join(args.out, "index.faiss")
META_PATH = os.path.join(args.out, "meta.pkl")

def chunk_text(text, size=700, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += size - overlap
        if start <= 0:
            break
    return chunks

# Read docs
texts = []
for name in os.listdir(args.docs):
    p = os.path.join(args.docs, name)
    if os.path.isfile(p) and any(name.lower().endswith(ext) for ext in [".txt", ".md"]):
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            texts.extend(chunk_text(fh.read()))

if not texts:
    raise SystemExit("Hiç metin bulunamadı. .txt/.md ekleyin.")

# Embed and write FAISS
model = SentenceTransformer(args.embedding)
vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
vecs = np.array(vecs, dtype=np.float32)

index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, INDEX_PATH)

ids = [str(uuid.uuid4()) for _ in texts]
with open(META_PATH, "wb") as f:
    pickle.dump({"ids": ids, "texts": texts}, f)

print(f"OK · chunks={len(texts)} -> {args.out}")
