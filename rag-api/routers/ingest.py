from fastapi import APIRouter, UploadFile, File, HTTPException
from services.retriever import index_document, ensure_index
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
import uuid
import io
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)
router = APIRouter()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Teile Text in Chunks mit Overlap auf"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

@router.on_event("startup")
async def startup():
    await ensure_index()

@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload und Index Dokument"""
    try:
        content_bytes = await file.read()
        text = ""
        
        # PDF vs Text
        if file.filename.endswith(".pdf"):
            try:
                reader = PdfReader(io.BytesIO(content_bytes))
                for page_num, page in enumerate(reader.pages):
                    text_content = page.extract_text() or ""
                    text += f"\n[Page {page_num + 1}]\n{text_content}"
            except Exception as e:
                logger.error(f"PDF Parse Error: {e}")
                raise HTTPException(status_code=400, detail="PDF konnte nicht gelesen werden")
        else:
            text = content_bytes.decode("utf-8", errors="ignore")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Kein Text extrahiert")
        
        # Chunking
        chunks = chunk_text(text)
        
        # Indexieren
        for idx, chunk in enumerate(chunks):
            doc_id = f"{file.filename}_{idx}_{uuid.uuid4()}"
            try:
                await index_document(doc_id, chunk, file.filename)
            except Exception as e:
                logger.error(f"Indexing Error für Chunk {idx}: {e}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"{len(chunks)} Chunks indexiert"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

