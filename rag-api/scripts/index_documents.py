#!/usr/bin/env python3
import sys
from pathlib import Path
import logging
import os
from tqdm import tqdm

# So kommst du zu deinen Imports im Projekt
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.retriever import index_document, ensure_index
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCUMENTS_FOLDER = Path(
    os.getenv(
        "DOCUMENTS_FOLDER",
        "/home/ruki-labor/Schreibtisch/BA-Talom/Dokuments"
    )
)


def extract_pdf_pages(pdf_path: Path):
    pages = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((page_num, text))

        logger.info(f"   📄 {len(pages)} Seiten extrahiert")
        return pages
    except Exception as e:
        logger.error(f"   ❌ Fehler: {e}")
        return []


def chunk_text(
    text: str,
    page_num: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):
    chunks = []
    sentences = text.split(". ")
    current_chunk = ""
    chunk_id = 0

    for sentence in sentences:
        # sentence wieder mit Punkt “runden”, weil split(". ") den Punkt verliert
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_with_dot = sentence + "."

        if len(current_chunk) + len(sentence_with_dot) < chunk_size:
            current_chunk += sentence_with_dot + " "
        else:
            if current_chunk.strip():
                chunks.append({
                    "id": f"page_{page_num}_chunk_{chunk_id}",
                    "content": current_chunk.strip(),
                    "page": page_num,
                })
                chunk_id += 1

            # overlap: wir nehmen die letzten overlap-Wörter des aktuellen sentence
            words = sentence.split()
            if len(words) > overlap:
                overlap_text = " ".join(words[-overlap:])
            else:
                overlap_text = sentence

            # neue Chunk-Struktur
            current_chunk = overlap_text + ". " + sentence + ". "

    if current_chunk.strip():
        chunks.append({
            "id": f"page_{page_num}_chunk_{chunk_id}",
            "content": current_chunk.strip(),
            "page": page_num,
        })

    return chunks


def main():
    logger.info("=" * 60)
    logger.info("🚀 RAG DOCUMENT INDEXING")
    logger.info("=" * 60)

    logger.info("\n🔧 Initialisiere Elasticsearch...")
    try:
        ok = ensure_index()
        if not ok:
            logger.error("❌ Index konnte nicht vorbereitet werden.")
            return 1
        logger.info("✅ Index erstellt/verifiziert")
    except Exception as e:
        logger.error(f"❌ Fehler bei Index-Erstellung: {e}")
        return 1

    if not DOCUMENTS_FOLDER.exists():
        logger.error(f"❌ Ordner nicht gefunden: {DOCUMENTS_FOLDER}")
        return 1

    pdf_files = sorted(DOCUMENTS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"⚠️ Keine PDFs in {DOCUMENTS_FOLDER}")
        return 1

    logger.info(f"\n📂 Ordner: {DOCUMENTS_FOLDER}")
    logger.info(f"📊 Gefunden: {len(pdf_files)} PDF(s)\n")

    total_chunks = 0
    errors = 0

    for idx, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"\n[{idx}/{len(pdf_files)}] 📥 {pdf_file.name}")
        try:
            pages = extract_pdf_pages(pdf_file)
            if not pages:
                continue

            for page_num, page_text in pages:
                chunks = chunk_text(page_text, page_num)

                for chunk in tqdm(chunks, desc=f"   Seite {page_num}", leave=False):
                    doc_id = f"{pdf_file.stem}_{chunk['id']}"
                    try:
                        success = index_document(
                            doc_id=doc_id,
                            content=chunk["content"],
                            filename=pdf_file.name,
                            page=chunk["page"],
                        )
                        if success:
                            total_chunks += 1
                    except Exception as e:
                        logger.error(f"   ❌ Fehler bei Chunk {chunk['id']}: {e}")

        except Exception as e:
            logger.error(f"❌ Fehler bei {pdf_file.name}: {e}")
            errors += 1

    logger.info("\n" + "=" * 60)
    logger.info("✅ INDEXIERUNG ABGESCHLOSSEN")
    logger.info("=" * 60)
    logger.info(f"📊 PDFs verarbeitet: {len(pdf_files) - errors}")
    logger.info(f"⚠️ Fehler: {errors}")
    logger.info(f"📈 Insgesamt Chunks: {total_chunks}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

