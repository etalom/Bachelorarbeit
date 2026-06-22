"""
Retriever Service - Hybrid Search (BM25 + Vector mit RRF)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch import Elasticsearch
from config.settings import (
    ES_URL, ES_INDEX, MIN_SCORE, 
    HYBRID_BM25_W, HYBRID_VECTOR_W, RRF_K, TOP_K_RESULTS,
    INTENT_BOOST
)
from services.embedder import get_embedding

logger = logging.getLogger(__name__)

# Lazy-load ES Client
_es_client = None

def get_es():
    """Elasticsearch Client (singleton)"""
    global _es_client
    if _es_client is None:
        _es_client = Elasticsearch([ES_URL])
        logger.info(f"✅ ES verbunden: {ES_URL}")
    return _es_client


def ensure_index():
    """Erstelle Index falls nicht vorhanden"""
    try:
        es = get_es()
        
        if es.indices.exists(index=ES_INDEX):
            logger.info(f"✅ Index '{ES_INDEX}' existiert")
            return True
        
        logger.info(f"🔧 Erstelle Index '{ES_INDEX}'...")
        es.indices.create(
            index=ES_INDEX,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "german_analyzer": {
                                "type": "standard",
                                "stopwords": "_german_"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "german_analyzer",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "filename": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "indexed_at": {"type": "date"},
                    }
                }
            }
        )
        logger.info(f"✅ Index erstellt")
        return True
    except Exception as e:
        logger.error(f"❌ Index-Fehler: {e}")
        return False


def index_document(doc_id: str, content: str, filename: str, page: int) -> bool:
    """Indexiere ein Dokument mit Embedding"""
    try:
        es = get_es()
        
        # Embedding generieren
        embedding = get_embedding(content)
        if embedding is None:
            logger.error(f"❌ Embedding fehlgeschlagen: {doc_id}")
            return False
        
        # Indexieren
        es.index(
            index=ES_INDEX,
            id=doc_id,
            body={
                "text": content,
                "filename": filename,
                "page": page,
                "embedding": embedding,
            }
        )
        
        return True
    except Exception as e:
        logger.error(f"❌ Index-Fehler {doc_id}: {e}")
        return False


def search_bm25(query: str, intent: str, size: int = 15) -> list[dict]:
    """BM25-Suche (lexikalisch)"""
    boost_terms = INTENT_BOOST.get(intent, [])
    should_clauses = [
        {
            "multi_match": {
                "query": query,
                "fields": ["text^2", "filename^3"],
                "type": "best_fields",
                "fuzziness": "AUTO",
            }
        }
    ]
    
    for term in boost_terms:
        should_clauses.append({
            "match_phrase": {
                "text": {"query": term, "boost": 3}
            }
        })
    
    es_query = {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
                "must_not": [
                    {"match_phrase": {"filename": "Rundschreiben"}}
                ],
            }
        },
        "highlight": {
            "fields": {"text": {}},
            "fragment_size": 900,
            "number_of_fragments": 1,
        },
        "size": size,
        "_source": ["text", "filename", "page"],
    }
    
    try:
        es = get_es()
        response = es.search(index=ES_INDEX, body=es_query)
        hits = response.get("hits", {}).get("hits", [])
        logger.info(f"📄 BM25 Treffer: {len(hits)}")
        return hits
    except Exception as e:
        logger.error(f"❌ BM25 Fehler: {e}")
        return []


def search_vector(query_embedding: list[float], size: int = 15) -> list[dict]:
    if not query_embedding:
        logger.warning("⚠️ Kein Embedding → Vector-Suche übersprungen")
        return []

    knn_query = {
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": size,
            "num_candidates": 100,
            "filter": {
                "bool": {
                    "must_not": [
                        {"match_phrase": {"filename": "Rundschreiben"}}
                    ]
                }
            },
        },
        "_source": ["text", "filename", "page"],
        "size": size,
    }
    
    try:
        es = get_es()
        response = es.search(index=ES_INDEX, body=knn_query)
        hits = response.get("hits", {}).get("hits", [])
        logger.info(f"🔢 Vector Treffer: {len(hits)}")
        return hits
    except Exception as e:
        logger.error(f"❌ Vector Fehler: {e}")
        return []


def reciprocal_rank_fusion(bm25_hits: list, vector_hits: list, top_n: int = TOP_K_RESULTS) -> tuple[list, float]:
    """RRF: Kombiniere BM25 + Vector Rankings"""
    doc_store = {}
    
    # BM25-Scores
    for rank, hit in enumerate(bm25_hits, 1):
        doc_id = hit["_id"]
        bm25_score = hit.get("_score", 0)
        rrf_score = HYBRID_BM25_W * (1 / (RRF_K + rank))
        
        doc_store[doc_id] = {
            "hit": hit,
            "bm25_rank": rank,
            "bm25_score": bm25_score,
            "bm25_rrf": rrf_score,
            "vec_rank": None,
            "vec_score": None,
            "vec_rrf": 0,
        }
    
    # Vector-Scores
    for rank, hit in enumerate(vector_hits, 1):
        doc_id = hit["_id"]
        vec_score = hit.get("_score", 0)
        rrf_score = HYBRID_VECTOR_W * (1 / (RRF_K + rank))
        
        if doc_id in doc_store:
            doc_store[doc_id]["vec_rank"] = rank
            doc_store[doc_id]["vec_score"] = vec_score
            doc_store[doc_id]["vec_rrf"] = rrf_score
        else:
            doc_store[doc_id] = {
                "hit": hit,
                "bm25_rank": None,
                "bm25_score": None,
                "bm25_rrf": 0,
                "vec_rank": rank,
                "vec_score": vec_score,
                "vec_rrf": rrf_score,
            }
    
    # Sortieren nach kombiniertem RRF-Score
    ranked = sorted(
        doc_store.values(),
        key=lambda x: x["bm25_rrf"] + x["vec_rrf"],
        reverse=True
    )
    
    logger.info(f"🔀 RRF: BM25({len(bm25_hits)}) + Vector({len(vector_hits)}) → Top {top_n}")
    
    # Finale Hits
    final_hits = []
    for item in ranked[:top_n]:
        hit = item["hit"].copy()
        combined_score = item["bm25_rrf"] + item["vec_rrf"]
        hit["_score"] = round(combined_score, 5)
        hit["_rrf_meta"] = {
            "rrf_score": hit["_score"],
            "bm25_rank": item["bm25_rank"],
            "bm25_score": round(item["bm25_score"], 3) if item["bm25_score"] else None,
            "vec_rank": item["vec_rank"],
            "vec_score": round(item["vec_score"], 3) if item["vec_score"] else None,
        }
        final_hits.append(hit)
    
    max_score = final_hits[0]["_score"] if final_hits else 0.0
    return final_hits, max_score


def search_hybrid(query: str, intent: str) -> tuple[list, float]:
    t_start = time.time()
    logger.info(f"🔍 Query: '{query}' | Intent: {intent}")

    bm25_hits = []
    query_embedding = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_bm25 = pool.submit(search_bm25, query, intent, 15)
        future_emb = pool.submit(get_embedding, query)

        for future in as_completed([future_bm25, future_emb]):
            if future == future_bm25:
                bm25_hits = future.result()
            elif future == future_emb:
                query_embedding = future.result()

    vector_hits = search_vector(query_embedding, size=15)

    if not bm25_hits and not vector_hits:
        logger.warning("⚠️ Keine Ergebnisse!")
        return [], 0.0

    if not bm25_hits:
        logger.warning("⚠️ BM25 leer → nur Vector")
        hits = vector_hits[:TOP_K_RESULTS]
        max_score = hits[0].get("_score", 0.0) if hits else 0.0
        return hits, max_score

    if not vector_hits:
        logger.warning("⚠️ Vector leer → nur BM25")
        hits = bm25_hits[:TOP_K_RESULTS]
        max_score = hits[0].get("_score", 0.0) if hits else 0.0
        return hits, max_score

    final_hits, max_score = reciprocal_rank_fusion(bm25_hits, vector_hits)

    t_elapsed = round((time.time() - t_start) * 1000, 2)
    logger.info(f"✅ Hybrid Search | Score: {max_score:.5f} | Zeit: {t_elapsed}ms | Hits: {len(final_hits)}")

    return final_hits, max_score

