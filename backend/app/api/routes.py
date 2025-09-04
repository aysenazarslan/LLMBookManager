# backend/app/api/routes.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
from uuid import uuid4
from pathlib import Path
import datetime
import os
import json
import re

import numpy as np
import faiss
import fitz  # PyMuPDF

# Embedding & FAISS yardımcıları (tek kaynaktan)
from app.core.embeddings import (
    embed_texts,
    build_faiss_index,
    save_embeddings,
    load_embeddings,
)

# LLM cevabı için merkezi sarmalayıcı
from app.core.llm_responder import LLMResponder

# ---- DB (MSSQL - SQLAlchemy + pyodbc) ----
from sqlalchemy import Table, MetaData, select, insert, text
from sqlalchemy.orm import Session

# database.py'den engine & get_db kullan
from app.core.database import engine, get_db

# =============================
# CONFIG
# =============================
router = APIRouter(prefix="/api", tags=["books", "rag", "status"])

EMBED_MODEL_ID = os.getenv(
    "EMBED_MODEL_ID", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "120"))
TOP_K = int(os.getenv("TOP_K", "3"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# LLM tekil örnek
llm = LLMResponder()

# =============================
# DB TABLE REFLECTION
# =============================
metadata = MetaData(schema="dbo")
Books: Table = Table("Books", metadata, autoload_with=engine)
Embeddings: Table = Table("Embeddings", metadata, autoload_with=engine)
ChatSessions: Table = Table("ChatSessions", metadata, autoload_with=engine)
Messages: Table = Table("Messages", metadata, autoload_with=engine)
Users: Table = Table("Users", metadata, autoload_with=engine)

# =============================
# IN-MEMORY FAISS CACHE
# =============================
# book_id(str) -> (faiss.Index, [segment_text by chunk_index])
faiss_indexes: Dict[str, Tuple[faiss.Index, List[str]]] = {}

# =============================
# SCHEMAS
# =============================
class BookOut(BaseModel):
    id: str
    title: str
    author: Optional[str] = None
    language: Optional[str] = "tr"
    created_at: Optional[datetime.datetime] = None

class AskBody(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # ChatSessions FK gereği zorunlu

class ChatResponse(BaseModel):
    answer: str
    source_chunks: List[int]

# =============================
# HELPERS
# =============================
def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(raw_text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", raw_text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"^\s*Page \d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()

def chunk_text(text: str, max_words: int = CHUNK_WORDS) -> List[Dict[str, Any]]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        w = words[i : i + max_words]
        chunks.append(
            {
                "chunk_id": str(uuid4()),
                "index": i // max_words,
                "chunk_text": " ".join(w),
                "token_count": len(w),
            }
        )
    return chunks

def _deserialize_vec(s: str) -> np.ndarray:
    arr = json.loads(s)
    return np.array(arr, dtype="float32")

def _build_faiss_from_db(db: Session, book_id: str) -> Tuple[faiss.Index, List[str]]:
    # Embeddings'ten bu kitaba ait tüm vektörleri ve metinleri sırayla çek
    rows = (
        db.execute(
            select(
                Embeddings.c.vector, Embeddings.c.segment_text, Embeddings.c.chunk_index
            )
            .where(Embeddings.c.book_id == book_id)
            .order_by(Embeddings.c.chunk_index.asc())
        )
        .all()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No embeddings found for this book.")

    texts: List[str] = []
    vecs: List[np.ndarray] = []
    for v_str, seg_text, _idx in rows:
        if v_str is None:
            continue
        texts.append(seg_text)
        vecs.append(_deserialize_vec(v_str))
    if not vecs:
        raise HTTPException(status_code=404, detail="Embeddings empty.")

    V = np.vstack(vecs).astype("float32")
    index = build_faiss_index(V, metric="l2")
    return index, texts

# =============================
# ROUTES
# =============================

@router.post("/books/upload")
async def upload_book(
    title: str = Form(...),
    author: str = Form(""),
    language: str = Form("tr"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # (Şimdilik) sadece PDF
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Only PDF is supported for now.")

    book_id = str(uuid4())
    raw_pdf_path = RAW_DIR / f"{book_id}.pdf"
    chunks_path = PROC_DIR / f"{book_id}_chunks.json"

    # Dosyayı kaydet
    with open(raw_pdf_path, "wb") as f:
        f.write(await file.read())

    # Metin çıkar + temizle + chunkla
    raw = extract_text_from_pdf(raw_pdf_path)
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Books tablosuna kayıt
    db.execute(
        insert(Books).values(
            id=book_id,
            title=title,
            author=author or None,
            language=language,
            source_file=str(raw_pdf_path),
            created_at=datetime.datetime.utcnow(),
        )
    )
    db.commit()

    return {
        "book_id": book_id,
        "title": title,
        "chunk_count": len(chunks),
        "status": "ok",
    }


@router.post("/books/{book_id}/embed")
async def embed_book(book_id: str, db: Session = Depends(get_db)):
    chunks_path = PROC_DIR / f"{book_id}_chunks.json"
    if not chunks_path.exists():
        raise HTTPException(status_code=404, detail="Chunk file not found")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["chunk_text"] for c in chunks]

    # Embedding üret (tek kaynaktan)
    vecs = embed_texts(texts, normalize=False)  # cosine istiyorsan True + metric="ip"
    dim = int(vecs.shape[1])

    # FAISS index'i belleğe kur ve cache'le
    index = build_faiss_index(vecs, metric="l2")
    faiss_indexes[book_id] = (index, texts)

    # Disk persist (opsiyonel)
    save_embeddings(book_id, vecs)  # data/processed/{book_id}_embeddings.npy

    # DB'ye Embeddings kayıtları (her chunk satırı)
    now = datetime.datetime.utcnow()
    rows = []
    # JSON string'e çevir: NVARCHAR(MAX) 'vector' alanına yazacağız
    for c, v in zip(chunks, vecs):
        rows.append(
            {
                "id": str(uuid4()),
                "book_id": book_id,
                "segment_text": c["chunk_text"],
                "vector": json.dumps([float(x) for x in v.tolist()]),
                "chunk_index": int(c["index"]),
                "created_at": now,
            }
        )

    db.execute(Embeddings.insert(), rows)  # bulk insert
    db.commit()

    return {
        "book_id": book_id,
        "chunk_count": len(chunks),
        "embedding_dim": dim,
        "status": "ok",
    }


@router.post("/chat/ask", response_model=ChatResponse)
async def chat_ask(body: AskBody, book_id: str, db: Session = Depends(get_db)):
    # FAISS cache hazır değilse DB'den yükleyip kur
    if book_id not in faiss_indexes:
        index, texts = _build_faiss_from_db(db, book_id)
        faiss_indexes[book_id] = (index, texts)

    index, texts = faiss_indexes[book_id]

    # Sorgu embed'i
    qvec = embed_texts([body.question], normalize=False)
    D, I = index.search(np.asarray(qvec, dtype=np.float32), k=TOP_K)

    idxs = [int(i) for i in I[0] if 0 <= i < len(texts)]
    ctx_chunks = [texts[i] for i in idxs]

    # LLM cevabı (tek sarmalayıcı)
    answer_text, _meta = llm.generate(
        question=body.question,
        contexts=ctx_chunks,
        source_indices=idxs,
        style={"temperature": 0.4, "max_tokens": 512},
        provider="openai",  # local istersen "local"
        add_citations=False,  # citations dizgesini kendimiz saklıyoruz
    )
    answer = answer_text

    # ChatSessions + Messages kayıtları
    if not body.user_id:
        raise HTTPException(status_code=400, detail="user_id is required (for ChatSessions FK)")

    session_id = body.session_id or str(uuid4())

    # Yeni session ise oluştur
    if body.session_id is None:
        db.execute(
            insert(ChatSessions).values(
                id=session_id,
                user_id=body.user_id,
                book_id=book_id,
                started_at=datetime.datetime.utcnow(),
                ended_at=None,
            )
        )

    # Mesajları kaydet
    db.execute(
        insert(Messages).values(
            id=str(uuid4()),
            session_id=session_id,
            role="user",
            message_text=body.question,
            timestamp=datetime.datetime.utcnow(),
            source_chunks=None,
        )
    )
    db.execute(
        insert(Messages).values(
            id=str(uuid4()),
            session_id=session_id,
            role="assistant",
            message_text=answer,
            timestamp=datetime.datetime.utcnow(),
            source_chunks=json.dumps(idxs),
        )
    )
    db.commit()

    return ChatResponse(answer=answer, source_chunks=idxs)


@router.get("/books", response_model=List[BookOut])
async def list_books(db: Session = Depends(get_db)):
    rows = (
        db.execute(
            select(
                Books.c.id, Books.c.title, Books.c.author, Books.c.language, Books.c.created_at
            ).order_by(Books.c.created_at.desc())
        )
        .all()
    )
    return [
        BookOut(
            id=str(r.id) if not isinstance(r.id, str) else r.id,
            title=r.title,
            author=r.author,
            language=r.language,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/sessions/{session_id}/messages")
async def session_messages(session_id: str, db: Session = Depends(get_db)):
    rows = (
        db.execute(
            select(
                Messages.c.role,
                Messages.c.message_text,
                Messages.c.timestamp,
                Messages.c.source_chunks,
            )
            .where(Messages.c.session_id == session_id)
            .order_by(Messages.c.timestamp.asc())
        )
        .all()
    )
    out = []
    for role, text_, ts, src in rows:
        out.append(
            {
                "role": role,
                "message": text_,
                "timestamp": ts,
                "source_chunks": json.loads(src) if src else None,
            }
        )
    return out


@router.get("/status")
async def status(db: Session = Depends(get_db)):
    books_count = db.execute(text("SELECT COUNT(*) FROM dbo.Books")).scalar_one()
    emb_count = db.execute(text("SELECT COUNT(*) FROM dbo.Embeddings")).scalar_one()
    sessions_count = db.execute(text("SELECT COUNT(*) FROM dbo.ChatSessions")).scalar_one()
    return {
        "books": int(books_count or 0),
        "embeddings": int(emb_count or 0),
        "sessions": int(sessions_count or 0),
        "faiss_loaded": list(faiss_indexes.keys()),
        "embed_model": EMBED_MODEL_ID,
        "llm_ready": True,  # LLMResponder içinde anahtar kontrolü var; burada basitleştiriyoruz
        "db": "mssql",
    }