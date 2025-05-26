import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_DIR = "vector_db"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
META_FILE = os.path.join(VECTOR_DIR, "metadata.json")

os.makedirs(VECTOR_DIR, exist_ok=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

def _save_metadata(metadata):
    with open(META_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def _load_metadata():
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            return json.load(f)
    return []

def add_to_vector_memory(prompt: str, name: str):
    """
    Add a prompt to the vector index and store its metadata.
    """
    embedding = model.encode([prompt])[0].astype('float32')
    embedding = np.array([embedding])

    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = faiss.IndexFlatL2(embedding.shape[1])

    index.add(embedding)
    faiss.write_index(index, INDEX_FILE)

    metadata = _load_metadata()
    metadata.append({
        "name": name.lower(),
        "prompt": prompt
    })
    _save_metadata(metadata)

def search_similar_prompt(query: str, top_k=1):
    """
    Return the closest matching previous prompt based on semantic similarity.
    """
    if not os.path.exists(INDEX_FILE):
        return None

    embedding = model.encode([query])[0].astype('float32')
    embedding = np.array([embedding])
    index = faiss.read_index(INDEX_FILE)
    metadata = _load_metadata()

    D, I = index.search(embedding, top_k)
    if I[0][0] != -1 and I[0][0] < len(metadata):
        return metadata[I[0][0]]
    return None
