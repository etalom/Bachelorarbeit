import logging
from config.settings import INTENT_KEYWORDS

logger = logging.getLogger(__name__)

def detect_intent(query: str) -> str:
    """Erkennt Intent basierend auf Keywords"""
    query_lower = query.lower()
    
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            logger.info(f"🎯 Intent: {intent}")
            return intent
    
    logger.info(f"🎯 Intent: allgemein (default)")
    return "allgemein"

