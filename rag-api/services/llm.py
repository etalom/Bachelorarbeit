import requests
import logging
from config.settings import INTENT_HINTS, OLLAMA_URL, TOKEN_LIMITS

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "Ich konnte keine passende Antwort in den bereitgestellten Dokumenten finden."

def build_prompt(query: str, context: str, intent: str, model: str) -> str:
    """Erstellt Prompt mit Intent-Hint"""
    system_hint = INTENT_HINTS.get(intent, INTENT_HINTS.get("definition", ""))
    
    return f"""{system_hint}

KONTEXT AUS DOKUMENTEN:
---
{context}
---

FRAGE: {query}

Basiere deine Antwort NUR auf dem obigen Kontext. Falls die Antwort nicht in den gegebenen Dokumenten steht, sag einfach klar, dass die Information nicht im Kontext enthalten ist."""

def get_temperature(model: str) -> float:
    return 0

def check_faithfulness(query: str, context: str, answer: str, model: str) -> str:
    """Prüft per zweitem LLM-Call, ob die Antwort im Kontext gedeckt ist."""
    check_prompt = f"""Du prüfst, ob die folgende Antwort im gegebenen Kontext enthalten ist.

KONTEXT:
---
{context[:2000]}
---

ANTWORT:
{answer}

FRAGE:
{query}

Ist die Antwort vollständig durch den Kontext gedeckt? Antworte nur mit JA oder NEIN."""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": check_prompt,
                "stream": False,
                "num_predict": 5,
                "temperature": 0,
            },
            timeout=15,
        )
        resp.raise_for_status()
        verdict = resp.json().get("response", "").strip().upper()
        if "JA" in verdict:
            return answer
        logger.warning(f"⚠️ Faithfulness-Guard griff ein (Antwort: {verdict})")
        return FALLBACK_ANSWER
    except Exception as e:
        logger.error(f"❌ Faithfulness-Guard Fehler: {e}")
        return answer

def generate_answer(query: str, context: str, intent: str, model: str) -> str:
    """Generiert Antwort mit Ollama + Faithfulness-Guard"""
    prompt = build_prompt(query, context, intent, model)
    num_predict = TOKEN_LIMITS.get(intent, 500)
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "num_predict": num_predict,
                "num_ctx": 4096,
                "temperature": 0,
                "top_p": 0.9,
            },
            timeout=300,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        if not answer:
            return FALLBACK_ANSWER
        
        return check_faithfulness(query, context, answer, model)

    except requests.exceptions.Timeout:
        logger.error("❌ Ollama Timeout")
        return "Zeitüberschreitung"
    except Exception as e:
        logger.error(f"❌ LLM Error: {type(e).__name__}: {e}")
        return "Fehler bei Antwortgenerierung"

