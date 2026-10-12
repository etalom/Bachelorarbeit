**RAG-Expertensystem – Retrieval-Augmented Generation**

Entwicklung eines RAG-Expertensystems mit hybrider Suche, lokalem LLM und Evaluation – im Rahmen meiner Bachelorarbeit an der Hochschule Koblenz.

**Überblick**

Dieses Projekt entstand im Rahmen meiner Bachelorarbeit im Studiengang Informationstechnik an der Hochschule Koblenz (März 2026 – Juni 2026).

Ziel war es, ein Retrieval-Augmented-Generation-(RAG)-System zu entwerfen und umzusetzen, das Dokumentensuche mit Large Language Models (LLMs) kombiniert und kontextbasierte Antworten auf Nutzerfragen liefert. Das System verbindet hybride Suche, ein API-Backend und eine Benutzeroberfläche für wissensintensive Frage-Antwort-Szenarien.

**Ziele**
Entwicklung einer vollständigen (End-to-End-)RAG-Pipeline
Kombination von schlagwortbasierter und vektorbasierter Suche
Verbesserung der Retrieval-Qualität durch Reciprocal Rank Fusion (RRF)
Bewertung der Retrieval-Leistung mit quantitativen Metriken
Umgang mit themenfremden Fragen und Reduzierung nicht belegter Antworten
Architektur

**Die Anwendung ist modular aufgebaut:**

Schritt	Komponente	Beschreibung
1	Dokumentenverarbeitung	Zerlegung der Dokumente in Chunks mit Überlappung (LangChain)
2	Informationssuche	BM25-Schlagwortsuche und Vektorsuche in Elasticsearch
3	Ergebnisfusion	Zusammenführung der Suchergebnisse per Reciprocal Rank Fusion (RRF)
4	Antwortgenerierung	Lokal betriebenes LLM (Ollama) erzeugt Antworten auf Basis des gefundenen Kontexts
5	Backend	FastAPI verbindet Retrieval und Generierung
6	Benutzeroberfläche	Streamlit zum Stellen von Fragen und Anzeigen der Antworten
Tech-Stack
Bereich	            Technologie
Programmiersprache	Python
LLM-Orchestrierung	LangChain
Suchmaschine	      Elasticsearch
Backend	            FastAPI
Frontend	          Streamlit
LLM-Runtime	        Ollama
Containerisierung	  Docker Compose
Versionskontrolle	  Git

**Zentrale Funktionen**
Hybrides Retrieval aus BM25 und Vektorsuche
Reciprocal Rank Fusion zur Zusammenführung der Suchergebnisse
Kontextbasierte Antwortgenerierung mit einem lokalen LLM
Chunking mit Überlappung, um zusammenhängenden Kontext zu erhalten
Erkennung und Behandlung themenfremder Fragen
Bewertung von Retrieval- und Antwortqualität
Evaluation

**Das System wurde mit einem Testset von 42 Fällen evaluiert. Die Evaluation umfasste:**

Precision@k
Recall@k
LLM-as-a-Judge
Faithfulness-Checks

**Ergebnisse für die getesteten Fragen mit Kontext (In-Context-Fragen):
**
Metrik	Ergebnis
Precision@k	1,00
Recall@15	1,00
Testfälle	42

**Hinweis**: Diese Ergebnisse gelten für die evaluierten Testfälle. Sie sind keine Garantie für gleiche Leistung bei unbekannten Fragen oder Dokumenten.

Erste Schritte
Voraussetzungen
Python
Docker und Docker Compose
Laufende Elasticsearch- und Ollama-Dienste (werden über Docker Compose bereitgestellt)
Installation
bash
git clone https://github.com/etalom/Bachelorarbeit.git
cd Bachelorarbeit
Konfiguration

Lege die benötigten Umgebungsvariablen (z. B. in einer .env-Datei) gemäß der Projektkonfiguration an.

**Start**
**bash**
docker compose up --build
Gelernte Kompetenzen
Aufbau von End-to-End-RAG-Anwendungen
Entwurf hybrider Information-Retrieval-Pipelines
Integration von LLMs mit Suchmaschinen und APIs
Bewertung von Retrieval-Qualität und generierten Antworten
Entwicklung modularer Python-Anwendungen
Autor

Emmanuel Talom Goumlepoue GitHub: github.com/etalom
