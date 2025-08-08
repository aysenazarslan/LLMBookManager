"""
embeddings.py
-----------------
Tek sorumluluğu: metinleri embedding'e çevirmek ve FAISS index işlemlerini yönetmek.

Kullanım:
    from embeddings import get_embedder, embed_texts, build_faiss_index,
        save_embeddings, load_embeddings, save_faiss_index, load_faiss_index

Ortam Değişkenleri:
    EMBED_MODEL_ID : HF SentenceTransformers model id (varsayılan: paraphrase-multilingual-mpnet-base-v2)
    EMBED_BATCH    : Batch size (varsayılan: 32)
    DATA_DIR       : Veri klasörü (varsayılan: ./data)

Notlar:
    - Çıktılar float32 olarak döndürülür (FAISS uyumu için).
    - Normalize (L2) opsiyonu ile kosinüs benzerliğine yakın davranış elde edilir.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from pathlib import Path
import os
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# -----------------------------
# Konfigürasyon
# -----------------------------
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Tekil (lazy) yüklenen model nesnesi
__EMBEDDER: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    """SentenceTransformer embedder'ı lazy-load eder ve cache'ler."""
    global __EMBEDDER
    if __EMBEDDER is None:
        __EMBEDDER = SentenceTransformer(EMBED_MODEL_ID)
    return __EMBEDDER


def _to_float32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def embed_texts(texts: List[str], normalize: bool = False, batch_size: Optional[int] = None) -> np.ndarray:
    """Metin listesini embedding vektörlerine çevirir.

    Args:
        texts: Girdi metinleri
        normalize: L2 normalize edip etmeyeceği (cosine benzerliğine yakın arama için önerilir)
        batch_size: Override batch size

    Returns:
        shape = (N, D) float32 numpy array
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embedder = get_embedder()
    bs = batch_size or EMBED_BATCH
    vecs = embedder.encode(texts, batch_size=bs, show_progress_bar=False)
    vecs = _to_float32(vecs)

    if normalize:
        faiss.normalize_L2(vecs)
    return vecs


# -----------------------------
# FAISS Yardımcıları
# -----------------------------

def build_faiss_index(vectors: np.ndarray, metric: str = "l2") -> faiss.Index:
    """Verilen vektörler için basit bir FAISS index döndürür.

    Args:
        vectors: shape (N, D) float32
        metric : "l2" (IndexFlatL2) veya "ip" (IndexFlatIP)
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D array [N, D]")
    n, d = vectors.shape
    if metric == "ip":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index


def search_index(index: faiss.Index, query_vectors: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS index'te arama yapar.

    Returns:
        distances: (Q, k)
        indices:   (Q, k)
    """
    if query_vectors.dtype != np.float32:
        query_vectors = _to_float32(query_vectors)
    D, I = index.search(query_vectors, top_k)
    return D, I


# -----------------------------
# Disk Persist
# -----------------------------

def save_embeddings(book_id: str, vectors: np.ndarray) -> Path:
    path = PROC_DIR / f"{book_id}_embeddings.npy"
    np.save(path, vectors)
    return path


def load_embeddings(book_id: str) -> np.ndarray:
    path = PROC_DIR / f"{book_id}_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    arr = np.load(path)
    return _to_float32(arr)


def save_faiss_index(book_id: str, index: faiss.Index) -> Path:
    path = PROC_DIR / f"{book_id}.faiss"
    faiss.write_index(index, str(path))
    return path


def load_faiss_index(book_id: str) -> faiss.Index:
    path = PROC_DIR / f"{book_id}.faiss"
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    index = faiss.read_index(str(path))
    return index


# -----------------------------
# Yüksek seviye akışlar
# -----------------------------

def embed_and_index(texts: List[str], book_id: str, normalize: bool = False, metric: str = "l2") -> Tuple[np.ndarray, faiss.Index]:
    """Metinleri embed eder, FAISS index kurar ve diske kaydeder."""
    vecs = embed_texts(texts, normalize=normalize)
    index = build_faiss_index(vecs, metric=metric)
    save_embeddings(book_id, vecs)
    save_faiss_index(book_id, index)
    return vecs, index


def ensure_index(book_id: str, metric: str = "l2") -> faiss.Index:
    """Diskte FAISS varsa yükler; yoksa embeddings'i yükleyip kurar."""
    try:
        return load_faiss_index(book_id)
    except FileNotFoundError:
        vecs = load_embeddings(book_id)
        index = build_faiss_index(vecs, metric=metric)
        save_faiss_index(book_id, index)
        return index