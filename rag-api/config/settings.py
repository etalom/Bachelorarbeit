import os
from dotenv import load_dotenv

load_dotenv()

# ── Document Chunking ──
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
# ── Elasticsearch ──
ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX", "pdf_index")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.005"))

# ── Ollama ──
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://ollama:11434/api/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")

AVAILABLE_MODELS = ["gpt-oss:20b", "mistral:latest", "mistral:7b", "gemma3:4b"]
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-oss:20b")

# ── Hybrid Search Gewichte ──
HYBRID_BM25_W = float(os.getenv("HYBRID_BM25_W", "0.7"))
HYBRID_VECTOR_W = float(os.getenv("HYBRID_VECTOR_W", "0.3"))
RRF_K = int(os.getenv("RRF_K", "30"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "15"))

# ── Token Limits pro Intent ──
TOKEN_LIMITS = {
    "schwellenwert": 500,
    "verfahrensart": 450,
    "unterlagen": 450,
    "fristen": 450,
    "konsequenz": 400,
    "angebote": 400,
    "vergleichsangebote": 400,
    "rahmenvertrag": 400,
    "bieter": 400,
    "drittmittel": 350,
    "nachhaltigkeit": 350,
    "papierangebote": 350,
    "ansprechpartner": 350,
    "definition": 350,
    "zweck": 350,
    "brutto_netto": 350,
    "direktkauf": 350,
    "allgemein": 350,
}

# ── Intent Keywords ──
INTENT_KEYWORDS = {
    "schwellenwert": [
        "schwellenwert", "grenzwert", "ab wann", "wann muss",
        "freibetrag", "ab euro", "obergrenze", "unterhalb",
        "oberhalb", "limit", "wertgrenze", "schwellen", "eu-schwelle",
        "europaweite", "oberhalb der eu", "unterhalb der eu",
    ],
    "verfahrensart": [
        "verfahrensart", "welches verfahren", "welches vergabeverfahren",
        "öffentlich", "beschränkt", "verhandlung", "verhandlungsvergabe",
        "direktauftrag", "ausschreibung", "verfahren", "vergabeverfahren",
        "beschaffung", "oeffentliche ausschreibung", "vergabeart",
    ],
    "definition": [
        "definition", "was ist", "bedeutet", "erkläre", "erläutere",
        "unterschied", "vergabe", "beschaffung", "verfahren",
        "vergabeverfahren", "kontext", "im kontext", "genau bedeutet",
    ],
    "zweck": [
        "zweck", "warum", "warum muss", "wieso", "motivation",
        "grund", "grundsätze", "grundsatz", "wettbewerb",
        "transparenz", "gleichbehandlung", "wirtschaftlichkeit",
        "spars", "sicherheit", "korruption", "vetternwirtschaft",
    ],
    "brutto_netto": [
        "brutto", "netto", "brutto oder netto", "welche anzugeben",
        "umsatzsteuer", "vorsteuer", "steuer", "kostenschätzung",
        "geschätzter auftragswert", "wert", "mehrwertsteuer","ohne umsatzsteuer",
    ],
    "direktkauf": [
        "direktkauf", "direkt beauftragen", "direkt beauftragt",
        "bis zu welchem betrag", "höchstbetrag", "budget",
        "selbst beauftragen", "ohne ausschreibung", "unmittelbar",
        "direktauftrag", "freihändige",  # falls du das im Kontext nutzt
    ],
    "angebote": [
        "angebote", "angebotseinholung", "selbst einholen",
        "darf ich angebote einholen", "eingeholt werden",
        "anzahl", "wie viele angebote", "anbieter", "lieferanten",
    ],
    "vergleichsangebote": [
        "vergleichsangebote", "vergleichs angebot", "mindestens drei",
        "drei angebote", "mindest", "begründung", "weniger angebote",
    ],
    "unterlagen": [
        "unterlagen", "dokumente", "vorlegen", "einreichen",
        "bedarfsermittlung", "bedarfsmeldung",
        "unternehmensliste", "eignungsanforderungen",
        "referenzen", "lv", "leistungsbeschreibung",
    ],
    "fristen": [
        "fristen", "frist", "abgabe", "zuschlag",
        "wie lange", "offen sein", "angemessen", "vorlaufzeit",
        "kalendertage", "elektronisch", "e-vergabe",
    ],
    "konsequenz": [
        "konsequenzen", "folgen", "drohen", "unzulässig",
        "nichtanerkennung", "rückforderung", "zuschlag unwirksam",
        "disziplinar", "heilung", "ablehnung",
    ],
    "bieter": [
        "bieter", "unternehmen", "zuständig", "anforderungen bieter",
        "teilnahme", "eignung", "zugelassen", "ausgeschlossen",
    ],
    "nachhaltigkeit": [
        "nachhaltigkeit", "soziale", "ökologische", "energieeffizienz",
        "reparierbarkeit", "faire lieferketten", "kriterien",
        "umwelt", "sozial",
    ],
    "papierangebote": [
        "papierangebote", "papier angebot", "schriftlich",
        "ausgeschlossen", "elektronisch", "form- und fristgerecht",
    ],
    "ansprechpartner": [
        "ansprechpartner", "zuständig", "wer ist zuständig",
        "zentrale vergabestelle", "dezentrale beschaffung",
        "kontakt", "stelle", "inventarisierung",
    ],
    "drittmittel": [
        "drittmittel", "fördermittel", "fördermittelgeber",
        "dfg", "bmbf", "zusätzlich", "besondere vorgaben",
    ],
    "rahmenvertrag": [
        "rahmenvertrag", "rahmenvertrag besteht", "muss genutzt",
        "vertrag deckt", "ähnliche leistungen", "nutzen",
    ],
    # fallback
    "allgemein": [
        "vergaberecht", "vergaberegeln", "beschaffung", "vergabe",
        "verstehen", "erklären", "übersicht", "wie funktioniert",
    ],
}

# ── Intent Boost (Trigger / Überschriften-Favoriten) ──
INTENT_BOOST = {
    "schwellenwert": [
        "Schwellenwert", "Wertgrenze", "Grenzwert",
        "ab welchem Betrag", "oberhalb der EU", "unterhalb der EU",
        "EU-Schwellenwert", "EU-weite Ausschreibung", "ca. 215.000"
    ],

    "verfahrensart": [
        "Vergabeverfahren", "Verfahren", "Vergabe",
        "Direktauftrag", "Direktauftrag bis",
        "Verhandlungsvergabe", "Beschränkte Ausschreibung",
        "öffentliche Ausschreibung", "europaweite Ausschreibung","ohne Umsatzsteuer"
    ],

    "definition": [
        "Definition", "Was ist", "Bedeutet", "Unterschied",
        "Vergabeverfahren", "Beschaffung",
        "formal geregelte Abläufe", "gesamter Prozess"
    ],

    "zweck": [
        "Zweck", "Grundsätze",
        "Wettbewerb", "Transparenz", "Gleichbehandlung",
        "Wirtschaftlichkeit", "Sparsamkeit",
        "rechtskonform", "Korruption", "Vetternwirtschaft"
    ],

    "brutto_netto": [
        "Brutto", "Netto", "Umsatzsteuer", "Vorsteuer",
        "ohne Umsatzsteuer", "geschätzter Auftragswert",
        "Gesamtwert", "inklusive aller Optionen", "Kostenübernahme"
    ],

    "direktkauf": [
        "Direktkauf", "Direktbeauftragung", "Direktauftrag",
        "bis zu welchem Betrag", "Höchstbetrag", "Budget",
        "Direktkauf zu beachten", "eigenständig Beschaffungen"
    ],

    "angebote": [
        "Angebote", "Angebotseinholung",
        "Angebote selbst einholen", "zentrale Beschaffungsstelle",
        "befugte Person", "genehmigte Ausnahmen",
        "Direktaufträgen unter", "zulässig"
    ],

    "vergleichsangebote": [
        "Vergleichsangebote",
        "mindestens drei", "drei Angebote",
        "im Wettbewerb eingeholt",
        "Begründung erforderlich", "wenn weniger"
    ],

    "unterlagen": [
        "Unterlagen", "Dokumente", "vorzulegen",
        "Bedarfsanforderung", "Warengruppe", "Kostenstelle", "PSP",
        "Produktbeschreibung", "LV", "Leistungsbeschreibung",
        "Angebot und eine Begründung",
        "Unternehmensliste", "Eignungsanforderungen", "Referenzen"
    ],

    "fristen": [
        "Fristen", "Vorlaufzeiten", "angemessene Fristen",
        "angemessen", "Angebotsabgabe", "Zuschlag",
        "10-14 Tagen", "10 bis 14", "10-14",
        "35 Kalendertage", "30 Kalendertage",
        "elektronische Vergabe", "e-vergabe",
        "§ 13 UVgO", "§ 15 VgV"
    ],

    "konsequenz": [
        "Konsequenzen", "Folgen",
        "nicht genehmigt", "unzulässig",
        "Nichtanerkennung der Ausgaben", "Nichtanerkennung",
        "Rückforderung", "Disziplinarmaßnahmen",
        "Zuschlag unwirksam", "unwirksam", "Heilung"
    ],

    "nachhaltigkeit": [
        "Nachhaltigkeit", "soziale und ökologische Kriterien",
        "Energieeffizienz", "Reparierbarkeit",
        "faire Lieferketten", "Lieferkettennachweise",
        "Nachhaltige Beschaffung"
    ],

    "papierangebote": [
        "Papierangebote", "Papierangebot", "schriftlich",
        "gemäß § 38 UVgO",
        "elektronische Angebotseinreichung",
        "vom Verfahren auszuschließen", "ausschließen",
        "nicht gültig", "form- und fristgerecht"
    ],

    "ansprechpartner": [
        "Ansprechpartner", "Ansprechpersonen",
        "zentrale Vergabestelle", "dezentral",
        "dezentrale Beschaffung", "Zuständigkeit",
        "Grundsätzlich erfolgt die Beschaffung", "DM-MA"
    ],

    "drittmittel": [
        "Drittmittel", "drittmittelfinanziert",
        "Fördermittelgeber", "DFG", "BMBF",
        "zusätzliche Vorgaben", "strengere Dokumentationspflichten",
        "DM-MA", "zuständige"
    ],

    "rahmenvertrag": [
        "Rahmenvertrag",
        "muss genutzt werden", "Rahmenvertrag besteht",
        "Eigenständige Beschaffungen außerhalb",
        "unzulässig", "deckt die gewünschte Leistung nicht ab"
    ],

    # fallback
    "allgemein": [
        "Vergabe", "Beschaffung", "Vergaberecht",
        "Übersicht", "Erklären", "Verstehen"
    ],
}

# ── Intent Hints (System Prompts) ──
# Hinweis: Regel „nur Verfahrensarten nennen, wenn sie in den Quellen stehen“
INTENT_HINTS = {
    "definition": """Du bist Experte für deutsches Vergaberecht, spezialisiert auf Rheinland-Pfalz.
Erkläre präzise, was ein Vergabeverfahren ist und was eine Beschaffung bedeutet.
Unterscheide klar:
- Vergabeverfahren = formal geregelter Ablauf zur Vergabe
- Beschaffung = gesamter Prozess (z.B. Bedarfsermittlung, Leistungserbringung, Kostenübernahme) und ist oft hochschulspezifisch

Wichtig (strenge Quellenregel):
Verfahrensarten: Nenne nur dann eine Verfahrensart, wenn sie in den Quellen ausdrücklich steht:
- Öffentliche Ausschreibung
- Beschränkte Ausschreibung
- Verhandlungsvergabe
- Direktauftrag

Keine Erfindungen. Rechtsgrundlagen nur soweit aus dem Kontext ersichtlich (GWB, VgV, UVgO, VV ÖA RLP).""",

    "schwellenwert": """Du bist Experte für Vergaberecht (Rheinland-Pfalz).
Beantworte Fragen zum Schwellenwert / Wertgrenze: ab welchem Betrag welches Vorgehen erforderlich ist.

Pflichten:
- Werte nur als Netto angeben, wenn es im Kontext so steht.
- Nenne nur konkrete Zahlen/Schwellen, wenn sie im Kontext vorkommen.
- Wenn der Betrag/Schwellenwert im Kontext fehlt: sag, dass es im Kontext nicht enthalten ist.

Rechts-/Bezugshinweise nur soweit aus dem Kontext ersichtlich (UVgO, VgV; ggf. VV ÖA RLP).""",

    "zweck": """Du erklärst den Zweck/Grund, warum Vergabeverfahren/Verfahren/Beschaffungen durchgeführt werden müssen.

Nutze nur die im Kontext genannten Grundsätze, z.B.:
Wettbewerb, Transparenz, Gleichbehandlung, Wirtschaftlichkeit, Sparsamkeit,
und wenn erwähnt: Vorbeugung gegen Korruption und Vetternwirtschaft.

Keine Erfindungen. Wenn ein Punkt nicht im Kontext steht: kurz vermerken, dass er dort nicht enthalten ist.
Rechtsgrundlagen nur soweit aus dem Kontext ersichtlich.""",

    "brutto_netto": """Du beantwortest Fragen zu Brutto/Netto im Vergabekontext.

Regeln:
- Erkläre die im Kontext genannte Logik zum geschätzten Auftragswert (z.B. „ohne Umsatzsteuer“, Gesamtwert inkl. Optionen/Verlängerungen – *nur wenn im Kontext so beschrieben*).
- Keine Umrechnung von Brutto/Netto vornehmen.
- Keine Erfindungen: wenn im Kontext steht „netto“, dann so angeben; wenn nicht: sagen, dass es fehlt.

Rechtsgrundlagen nur soweit aus dem Kontext ersichtlich (z.B. UVgO, VV ÖA RLP).""",

    "direktkauf": """Du beantwortest Fragen zu Direktkauf/Direktauftrag.

Pflichten:
- Nenne Höchstbeträge/Grenzen nur, wenn sie im Kontext ausdrücklich stehen.
- Erkläre nur das, was im Kontext zur Dokumentation/Beantragung erwähnt ist.
- Wenn „hochschulspezifisch“ erwähnt ist: sage, dass interne Prozessdetails variieren und im Kontext entsprechend benannt sein müssen.

Keine Verfahrensarten nennen, die nicht im Kontext vorkommen.
Rechtsgrundlagen nur soweit aus dem Kontext ersichtlich.""",

    "angebote": """Du erklärst, ob und wann Angebote eingeholt werden müssen und wer das darf.

Regeln:
- Nenne Regeln/Erlaubnisse nur, wenn sie im Kontext stehen (z.B. zentrale Beschaffung/ befugte Person / genehmigte Ausnahmen).
- Nenne nur zulässige Situationen, die im Kontext erwähnt sind.
- Keine Annahmen.

Wenn im Kontext „Direktaufträge unter 10.000 €“ als zulässig erwähnt sind: genau so übernehmen.""",

    "vergleichsangebote": """Du beantwortest Fragen zu Vergleichsangeboten.

Pflichten:
- Nenne Mindestanzahl nur, wenn sie im Kontext steht (z.B. „mindestens drei Angebote“).
- Nenne Begründungspflichten nur, wenn sie im Kontext vorkommen (z.B. „Begründung erforderlich, wenn weniger Angebote vorliegen“).
- Keine Erfindungen.""",

    "unterlagen": """Du erklärst, welche Unterlagen/Dokumente in welcher Situation vorgelegt werden müssen.

Regeln:
- Liste nur Unterlagen, die im Kontext benannt sind (z.B. Bedarfsanforderung inkl. Warengruppe, Kostenstelle, PSP, Produktbeschreibung/LV, ggf. Angebot und Begründung).
- Wenn im Kontext erwähnt: Unternehmensliste/Eignungsanforderungen/Referenzen nur dann nennen, wenn diese Bedingungen vorkommen (z.B. bei Ausschreibungen über 10.000 EUR – nur wenn im Kontext so steht).
- Keine Vermutungen.""",

    "fristen": """Du erklärst Fristen und Zeiträume im Vergabeprozess.

Regeln:
- Nenne konkrete Fristen/Zeitspannen nur, wenn sie im Kontext vorkommen.
- Wenn „angemessene Fristen“ oder „Vorlaufzeiten“ im Kontext vorkommen: übernimm genau diese Aussage.
- Wenn explizite Regeln genannt sind (z.B. § 13 UVgO, § 15 VgV, 10-14 Tage, 35/30 Kalendertage, E-Vergabe): nur diese Werte übernehmen.

Keine Erfindungen. Rechtsgrundlagen nur soweit aus dem Kontext ersichtlich.""",

    "konsequenz": """Du erklärst Konsequenzen, wenn Direktkäufe/Abweichungen nicht regelkonform sind.

Pflichten:
- Nenne nur Konsequenzen, die im Kontext vorkommen, z.B.:
  Nichtanerkennung der Ausgaben, Rückforderung von Drittmitteln, Disziplinarmaßnahmen
  sowie ggf. Aussage zur Vergabeprüfstelle/Unwirksamkeit/Heilung/ Ablehnung (nur wenn im Kontext so genannt).
- Keine zusätzlichen Sanktionen erfinden.""",

    "nachhaltigkeit": """Du erklärst, wie Nachhaltigkeitskriterien vergabekonform berücksichtigt werden.

Pflichten:
- Nenne nur Kriterien/Beispiele aus dem Kontext (z.B. Energieeffizienz, Reparierbarkeit, faire Lieferkettennachweise).
- Wenn im Kontext eine konkrete Rechtsverpflichtung genannt ist (z.B. LTTG RP), dann nur so wiedergeben wie im Kontext.
- Keine Erfindungen.""",

    "papierangebote": """Du beantwortest Fragen zu Papierangeboten vs. elektronischer Angebotsabgabe.

Regeln (streng aus Kontext):
- Wenn im Kontext steht, dass Papierangebote im Vergabeverfahren nicht zulässig sind:
  dann klar sagen: Papierangebote sind auszuschließen und ersetzen nicht die elektronische Angebotsabgabe.
- Wenn im Kontext § 38 UVgO genannt ist, nenne § 38 UVgO und die Pflicht zur elektronischen Einreichung nur dann, wenn es im Kontext vorkommt.
- Erkläre den Unterschied elektronisch vs. Papier nur im Rahmen des Kontextes.

Keine Vermutungen.""",

    "ansprechpartner": """Du erklärst, welche Stelle/Zuständigkeit gilt (zentrale vs. dezentrale Beschaffung).

Regeln:
- Nenne nur die im Kontext benannten Aussagen, z.B.:
  grundsätzlich zentrale Vergabestelle,
  dezentrale Beschaffung nur bei geringen Auftragswerten und mit vorheriger Genehmigung,
  außerdem: Ansprechpartnerrollen/ Zuständigkeiten nur soweit im Kontext benannt.
- Keine Annahmen.""",

    "drittmittel": """Du erklärst zusätzliche vergaberechtliche Anforderungen bei drittmittelfinanzierten Vorhaben.

Pflichten:
- Nenne nur die im Kontext genannten Dinge, z.B. dass Fördermittelgeber strengere Dokumentationspflichten und ggf. Schwellenwerte vorsehen können.
- Nenne den Hinweis auf den zuständigen DM-MA / Kontakt nur, wenn er im Kontext vorkommt.
- Keine Erfindungen.""",

    "rahmenvertrag": """Du erklärst, wann Rahmenverträge genutzt werden müssen und wann Ausnahmen gelten.

Regeln:
- Nenne nur die im Kontext enthaltenen Anforderungen:
  „Wenn ein Rahmenvertrag besteht, muss er genutzt werden“,
  „eigenständige Beschaffungen außerhalb sind unzulässig“ (sofern im Kontext genannt),
  „Ausnahme nur, wenn der Vertrag die gewünschte Leistung nicht abdeckt“ (nur wenn im Kontext so genannt).
- Keine Erfindungen.""",

    "allgemein": """Du bist Experte für deutsches Vergaberecht, spezialisiert auf Rheinland-Pfalz.
Gib eine hilfreiche, präzise Antwort basierend auf dem Kontext.
Wenn Details im Kontext fehlen: sag das kurz.

Zahlen/konkrete Schwellen und Verfahrensarten nur nennen, wenn sie im Kontext ausdrücklich stehen.
Rechtsgrundlagen nur soweit aus dem Kontext ersichtlich: GWB, VgV, UVgO, VV ÖA RLP.""",
}

print(f"✅ Config loaded: ES={ES_URL} | Ollama={OLLAMA_URL} | Model={DEFAULT_MODEL}")

