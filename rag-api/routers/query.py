from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import time
import re
import logging

from config.settings import (
    AVAILABLE_MODELS, DEFAULT_MODEL, MIN_SCORE
)
from services.retriever import search_hybrid, search_bm25, search_vector
from services.embedder import get_embedding
from services.llm import generate_answer
from services.intent import detect_intent

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Response Models ──
class SourceEntry(BaseModel):
    quelle: int
    datei: str
    seite: int
    score: float
    relevanz: str
    abschnitt: str


class AskResponse(BaseModel):
    antwort: str
    intent: str
    model: str
    retrieval_mode: str
    meta: dict
    quellen: list[SourceEntry]


def build_context(hits: list) -> tuple[str, list]:
    """Erstellt Kontext + Sources für Response"""
    context_parts = []
    sources = []

    for i, hit in enumerate(hits, 1):
        source = hit.get("_source", {})
        score = round(hit.get("_score", 0), 5)
        text = source.get("text", "")[:800]
        datei = source.get("filename", "unbekannt")
        seite = source.get("page", "?")
        highlight = hit.get("highlight", {}).get("text", [])

        clean_highlight = [re.sub(r"<[^>]+>", "", frag).strip() for frag in highlight]
        abschnitt = " [...] ".join(clean_highlight) if clean_highlight else text[:500]

        relevanz = "hoch" if i == 1 else "mittel" if i <= 3 else "niedrig"

        context_parts.append(abschnitt)

        sources.append(
            SourceEntry(
                quelle=i,
                datei=datei,
                seite=seite,
                score=score,
                relevanz=relevanz,
                abschnitt=abschnitt[:300],
            )
        )

    return "\n\n---\n\n".join(context_parts), sources


@router.get("/ask", response_model=AskResponse)
def ask(
    query: str = Query(..., description="Ihre Frage "),
    model: str = Query(DEFAULT_MODEL, enum=AVAILABLE_MODELS),
):
    """Hybrid Search + LLM"""
    model = model.strip()

    # Begrüßung (nur exakte/reine Begrüßungen, keine Fragen die zufällig "hallo" enthalten)
    gruss_woerter = ["hallo", "hi", "guten morgen", "guten tag", "hey", "servus", "moin", "guten abend"]
    query_clean = query.lower().strip().strip("!.,? ")
    if query_clean in gruss_woerter:
        return {
            "antwort": "Hallo! Ich bin dein Assistent. Wie kann ich dir helfen?",
            "intent": "begruessung",
            "model": model,
            "retrieval_mode": "none",
            "meta": {"latency_ms": 0, "score": 0, "hits": 0},
            "quellen": [],
        }

    start = time.time()
    intent = detect_intent(query)
    logger.info(f"🔍 Query: '{query}' | Intent: {intent} | Model: {model}")

    # Hybrid Search
    hits, max_score = search_hybrid(query, intent)

    # Keine Ergebnisse
    if not hits or max_score < MIN_SCORE:
        return {
            "antwort": "Keine relevanten Dokumente gefunden.",
            "intent": intent,
            "model": model,
            "retrieval_mode": "hybrid",
            "meta": {
                "latency_ms": round((time.time() - start) * 1000, 2),
                "score": round(max_score, 5),
                "hits": 0,
            },
            "quellen": [],
        }

    # Kontext + Antwort
    context, sources = build_context(hits)
    answer = generate_answer(query, context, intent, model)
    latency_ms = round((time.time() - start) * 1000, 2)

    logger.info(f"✅ Response | Zeit: {latency_ms}ms | Score: {max_score:.5f}")

    return {
        "antwort": answer,
        "intent": intent,
        "model": model,
        "retrieval_mode": "hybrid (BM25 + Vector RRF)",
        "meta": {
            "latency_ms": latency_ms,
            "score": round(max_score, 5),
            "hits": len(hits),
        },
        "quellen": sources,
    }


@router.get("/compare")
def compare(
    query: str = Query(..., description="Testfrage"),
    model: str = Query(DEFAULT_MODEL, enum=AVAILABLE_MODELS),
):
    """BM25 vs Vector Vergleich inkl. Analyse (gemeinsam / nur / Overlap %)"""

    model = model.strip()
    intent = detect_intent(query)

    def chunk_key(hit: dict) -> str | None:
        src = hit.get("_source", {}) or {}
        filename = src.get("filename")
        page = src.get("page")
        start = src.get("start_offset")
        end = src.get("end_offset")
        if filename is not None and page is not None and start is not None and end is not None:
            return f"{filename}::p{page}::{start}-{end}"
        if filename is not None and page is not None:
            return f"{filename}::p{page}"
        return None

    # BM25
    t0 = time.time()
    bm25_hits = search_bm25(query, intent, size=8)
    bm25_time = round((time.time() - t0) * 1000, 2)
    ctx_bm25, src_bm25 = build_context(bm25_hits)
    ans_bm25 = generate_answer(query, ctx_bm25, intent, model) if bm25_hits else "Keine Ergebnisse"

    # Vector (reine Vectorsuche, ohne BM25)
    t1 = time.time()
    query_embedding = get_embedding(query)
    vector_hits = search_vector(query_embedding, size=8)
    vector_time = round((time.time() - t1) * 1000, 2)
    ctx_vector, src_vector = build_context(vector_hits)
    ans_vector = generate_answer(query, ctx_vector, intent, model) if vector_hits else "Keine Ergebnisse"

    bm25_keys = {chunk_key(h) for h in bm25_hits}
    vector_keys = {chunk_key(h) for h in vector_hits}

    bm25_keys.discard(None)
    vector_keys.discard(None)

    gemeinsam_keys = bm25_keys & vector_keys
    nur_bm25_keys = bm25_keys - vector_keys
    nur_vector_keys = vector_keys - bm25_keys

    den = max(len(bm25_keys | vector_keys), 1)
    overlap_ratio = len(gemeinsam_keys) / den
    overlap_prozent = overlap_ratio * 100

    return {
        "query": query,
        "intent": intent,
        "bm25": {
            "answer": ans_bm25,
            "hits": len(bm25_hits),
            "latency_ms": bm25_time,
            "max_score": max([h.get("_score", 0) for h in bm25_hits], default=0),
            "sources": src_bm25,
        },
        "vector": {
            "answer": ans_vector,
            "hits": len(vector_hits),
            "latency_ms": vector_time,
            "max_score": (
                max([h.get("_score", 0) for h in vector_hits], default=0)
            ),
            "sources": src_vector,
        },
        "analyse": {
            "gemeinsame_dokumente": len(gemeinsam_keys),
            "nur_in_bm25": len(nur_bm25_keys),
            "nur_in_vector": len(nur_vector_keys),
            "overlap_prozent": overlap_prozent,
        },
        "overlap": overlap_ratio,
    }


@router.get("/health")
def health():
    """Health Check"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "modelle": AVAILABLE_MODELS,
    }

