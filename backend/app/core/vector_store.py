"""
vector_store.py
-----------------
FAISS tabanlý vektör maðazasý (Vector Store) sarmalayýcý.

Özellikler:
- Metinleri embedding'e çevirip FAISS'e ekler
- Diskte indeks + embedding + metadata saklama
- Sorgu için TOP-K benzer chunk döndürme
- Cosine (IP + normalize) ya da L2 metrik seçimi

Baðýmlýlýklar:
 - embeddings.py (get_embedder, embed_texts, build_faiss_index, ensure_index, save_faiss_index, save_embeddings)

Kullaným:
    store = VectorStore(book_id="abc123", metric="cosine")
    store.build_from_texts(texts, metadatas)
    hits = store.search("soru metni", top_k=3)

Metadata Þemasý:
    [
      {"index": 0, "chunk_id": "...", "text": "...", **extra_meta}
      ...
    ]
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import os
import numpy as np
import faiss

from embeddings import (
    embed_texts,
    build_faiss_index,
    ensure_index,
    save_faiss_index,
    save_embeddings,
)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


class VectorStore:
    def __init__(self, book_id: str, metric: str = "cosine") -> None:
        """
        Args:
            book_id: Ýçerik kimliði (kitap vb.)
            metric: "cosine" (IP + normalize) | "l2"
        """
        self.book_id = book_id
        self.metric = metric.lower()
        if self.metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.index: Optional[faiss.Index] = None
        self.metadatas: List[Dict[str, Any]] = []

    # -----------------------------
    # Persist paths
    # -----------------------------
    @property
    def _faiss_path(self) -> Path:
        return PROC_DIR / f"{self.book_id}.faiss"

    @property
    def _emb_path(self) -> Path:
        return PROC_DIR / f"{self.book_id}_embeddings.npy"

    @property
    def _meta_path(self) -> Path:
        return PROC_DIR / f"{self.book_id}_meta.json"

    # -----------------------------
    # Build / Load / Save
    # -----------------------------
    def build_from_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        normalize = self.metric == "cosine"
        vecs = embed_texts(texts, normalize=normalize)
        metric = "ip" if self.metric == "cosine" else "l2"
        self.index = build_faiss_index(vecs, metric=metric)
        save_embeddings(self.book_id, vecs)
        save_faiss_index(self.book_id, self.index)

        # metadata kaydet
        if metadatas is None:
            metadatas = [{"index": i, "text": t} for i, t in enumerate(texts)]
        else:
            # index alaný yoksa ekle
            for i, md in enumerate(metadatas):
                md.setdefault("index", i)
                md.setdefault("text", texts[i])
        self.metadatas = metadatas
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        # FAISS
        try:
            self.index = ensure_index(self.book_id, metric=("ip" if self.metric == "cosine" else "l2"))
        except Exception as e:
            raise FileNotFoundError(f"FAISS index yüklenemedi: {e}")
        # Metadata
        if not self._meta_path.exists():
            raise FileNotFoundError(f"Metadata bulunamadý: {self._meta_path}")
        self.metadatas = json.loads(self._meta_path.read_text(encoding="utf-8"))

    def is_ready(self) -> bool:
        return self.index is not None and len(self.metadatas) > 0

    # -----------------------------
    # Update
    # -----------------------------
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        if self.index is None:
            raise RuntimeError("Önce build_from_texts() ya da load() çaðýrýn")
        normalize = self.metric == "cosine"
        new_vecs = embed_texts(texts, normalize=normalize)
        self.index.add(new_vecs.astype(np.float32))

        start = len(self.metadatas)
        if metadatas is None:
            metadatas = [{"index": start + i, "text": t} for i, t in enumerate(texts)]
        else:
            for i, md in enumerate(metadatas):
                md.setdefault("index", start + i)
                md.setdefault("text", texts[i])
        self.metadatas.extend(metadatas)

        # persist güncelle
        save_faiss_index(self.book_id, self.index)
        if self._meta_path.exists():
            old = json.loads(self._meta_path.read_text(encoding="utf-8"))
        else:
            old = []
        old.extend(metadatas)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(old, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # Query
    # -----------------------------
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index hazýr deðil. load() ya da build_from_texts() çaðýrýn.")
        # embed query
        normalize = self.metric == "cosine"
        qvec = embed_texts([query], normalize=normalize)
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32)
        D, I = self.index.search(qvec, top_k)
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            md = self.metadatas[idx].copy()
            md.update({"distance": float(dist), "faiss_index": int(idx)})
            results.append(md)
        return results