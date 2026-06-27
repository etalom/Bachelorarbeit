import logging
from config.settings import INTENT_KEYWORDS

logger = logging.getLogger(__name__)

def detect_intent(query: str) -> str:
    """Erkennt Intent basierend auf Keywords (Score-basiert)"""
    query_lower = query.lower()
    
    best_intent = "allgemein"
    best_score = 0
    best_phrases = 0
    
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        phrases = 0
        for kw in keywords:
            if kw in query_lower:
                if len(kw.split()) > 1:
                    score += 3
                    phrases += 1
                else:
                    score += 1
        if score > best_score or (score == best_score and phrases > best_phrases):
            best_score = score
            best_phrases = phrases
            best_intent = intent
    
    logger.info(f"🎯 Intent: {best_intent} (Score: {best_score}, Phrasen: {best_phrases})")
    return best_intent

