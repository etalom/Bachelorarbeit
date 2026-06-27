import requests
import logging
from config.settings import INTENT_HINTS, OLLAMA_URL, TOKEN_LIMITS

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "Die von Ihnen angefragten Informationen sind in den mir vorliegenden Vergabe-Dokumenten nicht enthalten. Bitte prüfen Sie, ob Ihre Frage den Anwendungsbereich der bereitgestellten Unterlagen betrifft, oder formulieren Sie Ihre Frage genauer."

def build_prompt(query: str, context: str, intent: str, model: str) -> str:
    """Erstellt Prompt mit Intent-Hint"""
    system_hint = INTENT_HINTS.get(intent, INTENT_HINTS.get("definition", ""))
    
    return f"""{system_hint}

KONTEXT AUS DOKUMENTEN:
---
{context}
---

FRAGE: {query}

WICHTIG – Antworte streng nach diesen Regeln:
1. Die Antwort MUSS auf dem obigen Kontext basieren. Keine Informationen erfinden, die nicht im Kontext stehen.
2. Versuche die Frage so gut wie möglich aus dem Kontext zu beantworten. Oft sind die Informationen sinngemäß in den Dokumenten enthalten, auch wenn nicht exakt das gleiche Wort verwendet wird.
3. Wenn die Dokumente Teilinformationen enthalten, gib diese präzise und faktenbasiert weiter – ohne Ausschmückungen.
4. Nur wenn wirklich GAR KEINE relevanten Informationen im Kontext sind: Teile höflich mit, dass die Frage nicht auf Basis der vorliegenden Vergabe-Dokumente beantwortet werden kann. Formuliere natürlich, kein Standard-Satz."""

def get_temperature(model: str) -> float:
    return 0

def check_faithfulness(query: str, context: str, answer: str, model: str) -> str:
    """Prüft per zweitem LLM-Call, ob die Antwort im Kontext gedeckt ist (nur Logging, blockiert nicht mehr)."""
    max_ctx = 4000 if len(context) > 4000 else len(context)
    check_prompt = f"""Du prüfst, ob die folgende Antwort im gegebenen Kontext enthalten ist.

KONTEXT:
---
{context[:max_ctx]}
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
                "num_predict": 10,
                "temperature": 0.1,
            },
            timeout=15,
        )
        resp.raise_for_status()
        verdict = resp.json().get("response", "").strip()
        logger.info(f"🔎 Faithfulness-Verdict: {verdict}")
        if "JA" not in verdict.upper().split()[:3]:
            logger.warning(f"⚠️ Faithfulness-Guard würde eingreifen (Verdict: {verdict}), lasse Antwort aber trotzdem durch")
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

