from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# ✅ Richtiger Import
from routers.query import router as query_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API - Bachelor Arbeit",
    version="1.0.0",
    description="Hybrid Search + LLM Integration"
)

# CORS für Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router
app.include_router(query_router, prefix="/api", tags=["Query"])

@app.get("/health")
async def health():
    """Health Check"""
    return {
        "status": "healthy",
        "service": "rag-api",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root Endpoint"""
    return {
        "service": "RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )

