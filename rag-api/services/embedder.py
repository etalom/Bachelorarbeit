import requests
import logging
from config.settings import EMBEDDING_URL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

def get_embedding(text: str) -> list[float]:
    """
    🧠 Erzeugt Embedding über Ollama BGE-M3
    
    BGE-M3 Properties:
    - Dimension: 1024
    - Deutsch optimiert ✅
    - State-of-the-art Retrieval
    """
    try:
        response = requests.post(
            EMBEDDING_URL,
            json={
                "model": EMBEDDING_MODEL,
                "prompt": text,
            },
            timeout=60,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        
        if not embedding:
            logger.warning("⚠️ Empty embedding vector")
            return []
        
        logger.debug(f"✅ Embedding generated | Dim: {len(embedding)}")
        return embedding
        
    except requests.exceptions.Timeout:
        logger.error("❌ Embedding Timeout")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("❌ Ollama nicht erreichbar")
        return []
    except Exception as e:
        logger.error(f"❌ Embedding Error: {type(e).__name__}: {e}")
        return []

