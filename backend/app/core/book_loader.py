"""
book_loader.py
--------------
Kitap içeriğini (PDF/ePub) okur, temizler, chunk'lara böler ve metadata ile beraber diske yazar.

Çıktılar (DATA_DIR/processed altında):
 - {book_id}_chunks.json  : [ {chunk_id, index, chunk_text, token_count, page_start, page_end} ]
 - {book_id}_meta.json    : { book_id, title, author, language, pages, created_at }

Bağımlılıklar:
 - PyMuPDF (fitz)   -> PDF okuma
 - ebooklib + bs4   -> ePub okuma

Ortam değişkenleri:
 - DATA_DIR     : varsayılan ./data
 - CHUNK_WORDS  : varsayılan 120
 - CHUNK_OVERLAP_WORDS : varsayılan 30 (0 da olabilir)

Kullanım:
    from book_loader import process_book
    chunks, meta = process_book(file_path="data/raw/xxx.pdf", book_id="abc123", title="Küçük Prens", author="...")
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import os
import re
import json
import uuid
import datetime as dt

# PDF
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# EPUB
try:
    from ebooklib import epub
    from bs4 import BeautifulSoup
    HAS_EPUB = True
except Exception:
    HAS_EPUB = False

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "120"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))

# -----------------------------
# Public API
# -----------------------------

def process_book(file_path: str | Path,
                 book_id: str,
                 title: str,
                 author: Optional[str] = None,
                 language: str = "tr") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Kitabı oku, temizle, chunk'lara böl ve diske kaydet.

    Returns: (chunks, meta)
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {p}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        text, page_map = _read_pdf(p)
    elif ext in (".epub",):
        text, page_map = _read_epub(p)
    else:
        raise ValueError(f"Desteklenmeyen format: {ext}")

    cleaned = _clean_text(text)
    chunks = _chunk_text(cleaned, CHUNK_WORDS, CHUNK_OVERLAP, page_map)

    # persist chunks
    chunks_path = PROC_DIR / f"{book_id}_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    meta = {
        "book_id": book_id,
        "title": title,
        "author": author,
        "language": language,
        "pages": page_map.get("pages", 0),
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    meta_path = PROC_DIR / f"{book_id}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return chunks, meta

# -----------------------------
# Readers
# -----------------------------

def _read_pdf(path: Path) -> Tuple[str, Dict[str, Any]]:
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) kurulu değil: pip install pymupdf")
    doc = fitz.open(path)
    texts: List[str] = []
    page_offsets: List[Tuple[int, int]] = []  # (start_char, end_char)

    total = 0
    for page in doc:
        t = page.get_text()
        texts.append(t)
        start = total
        total += len(t)
        page_offsets.append((start, total))

    full = "".join(texts)
    return full, {"pages": len(texts), "offsets": page_offsets}


def _read_epub(path: Path) -> Tuple[str, Dict[str, Any]]:
    if not HAS_EPUB:
        raise RuntimeError("EPUB için ebooklib ve beautifulsoup4 kurulu olmalı: pip install ebooklib bs4")
    book = epub.read_epub(str(path))
    bufs: List[str] = []
    offsets: List[Tuple[int, int]] = []
    total = 0

    for item in book.get_items():
        if item.get_type() == 9:  # DOCUMENT
            soup = BeautifulSoup(item.get_content(), "html.parser")
            t = soup.get_text(separator=" ")
            bufs.append(t)
            start = total
            total += len(t)
            offsets.append((start, total))

    full = "".join(bufs)
    return full, {"pages": len(offsets), "offsets": offsets}

# -----------------------------
# Cleaning & Chunking
# -----------------------------

def _clean_text(raw: str) -> str:
    # fazla boşluk ve satır atlamaları
    text = re.sub(r"\u00a0", " ", raw)  # non-breaking space
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    # sayfa numarası/tek başına kalan "Page x" kalıpları
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _chunk_text(text: str, max_words: int, overlap_words: int, page_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    words = text.split()
    chunks: List[Dict[str, Any]] = []

    def span_to_pages(start_char: int, end_char: int) -> Tuple[int, int]:
        # char aralığını yaklaşık sayfa aralığına çevir
        offs = page_map.get("offsets", [])
        start_page = end_page = 0
        for i, (s, e) in enumerate(offs):
            if start_char >= s:
                start_page = i + 1
            if end_char <= e:
                end_page = i + 1
                break
        if end_page < start_page:
            end_page = start_page
        return start_page, end_page

    i = 0
    char_cursor = 0
    while i < len(words):
        w = words[i:i + max_words]
        chunk_str = " ".join(w)
        # kabaca karakter aralığı: mevcut cursor + chunk_len
        start_char = char_cursor
        end_char = start_char + len(chunk_str)
        ps, pe = span_to_pages(start_char, end_char)

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "index": len(chunks),
            "chunk_text": chunk_str,
            "token_count": len(w),
            "page_start": ps,
            "page_end": pe,
        })
        # overlap ile ilerleme
        step = max_words - max(0, overlap_words)
        i += step if step > 0 else max_words
        char_cursor = end_char + 1

    return chunks

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PDF/ePub kitabı işle ve chunk'la")
    ap.add_argument("file", type=str, help="Girdi PDF/ePub yolu")
    ap.add_argument("--book-id", type=str, default=None)
    ap.add_argument("--title", type=str, default="Untitled")
    ap.add_argument("--author", type=str, default=None)
    args = ap.parse_args()

    bid = args.book_id or uuid.uuid4().hex[:8]
    chunks, meta = process_book(args.file, bid, args.title, args.author)
    print(f"✅ {args.file} işlendi: {len(chunks)} chunk; meta: {meta}")