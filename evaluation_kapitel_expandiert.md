# 6 Evaluation

Ziel dieses Kapitels ist die quantitative und qualitative Bewertung des entwickelten
Retrieval-Augmented-Generation-(RAG)-Systems. Die Evaluation umfasst sowohl
die Analyse der Retrieval-Qualität als auch die Bewertung der generierten Antworten.
Hierzu wurde ein Katalog aus 17 Evaluationsfragen erstellt, der sowohl fachspezifische als auch fachfremde Anfragen abdeckt.

---

## 6.1 Evaluationsmethodik

### 6.1.1 Fragendesign

Die Evaluation basiert auf einem Datensatz aus 17 Nutzeranfragen, die in zwei Kategorien unterteilt sind:

**Kontextfragen (12):** Fragen, die aus dem Dokumentenkorpus beantwortbar sind. Sie wurden aus den User Stories (Kapitel 4.4) und den realen Vergaberechtsdokumenten abgeleitet und decken die wichtigsten Nutzungsszenarien ab:

- Schwellenwerte und Beträge (UVgO, VV ÖA, LHO)
- Fristen und Verfahrensdauer (UVgO, VgV)
- Brutto/Netto-Angaben
- Dokumentenanforderungen
- Verfahrensarten und -begründungen
- Rollenspezifische Fragen (Mitarbeitende, Sekretariat, Projektleitung)

**Out-of-Context-Fragen (5):** Fragen, die nicht aus dem Korpus beantwortet werden können. Sie testen die Robustheit des Systems gegenüber fachfremden Anfragen und die Fähigkeit, Halluzinationen zu vermeiden:

- Allgemeinwissen (Mondlandung, Relativitätstheorie)
- Alltagsthemen (Pasta kochen, Laptop-Reparatur)
- Grenzfälle (Laptop-Beschaffung -- nahe an der Domäne, aber nicht durch konkrete Dokumente gedeckt)

\begin{table}[h]
\centering
\caption{Kategorisierung der Evaluationsfragen}
\label{tab:fragenkatalog}
\begin{tabular}{p{0.3cm}p{1.5cm}p{10cm}}
\toprule
 & \textbf{Kategorie} & \textbf{Beispielfragen} \\
\midrule
1--12 & Kontextfragen & "Ab welchem Betrag ist eine Vergabe durchzuführen?" \\
& & "Welche Dokumente müssen vorgelegt werden?" \\
& & "Wie lange dauert ein Vergabeverfahren?" \\
\midrule
13--17 & Out-of-Context & "Wie mache ich eine Mondlandung?" \\
& & "Wie koche ich Pasta?" \\
& & "Wer hat die Relativitätstheorie erfunden?" \\
\bottomrule
\end{tabular}
\end{table}

### 6.1.2 Systemkonfiguration

Alle Evaluationen wurden mit folgender Konfiguration durchgeführt:

- **LLM**: `gpt-oss:20b` (20,9 Mrd. Parameter, MXFP4-Quantisierung, 13 GB)
- **Embedding-Modell**: `bge-m3:latest`
- **Retrieval-Modus**: Hybrid (BM25 + vector search, gewichtet)
- **Top-k Chunks**: 5
- **Faithfulness-Check**: Aktiv (zweiter LLM-Durchlauf zur Validierung)
- **Hardware**: NVIDIA RTX 5090 (32 GB VRAM), CPU-only-Fallback deaktiviert
- **Durchschnittliche Latenz**: 6,4 Sekunden pro Query

### 6.1.3 Bewertungsmetriken

Zur Bewertung wurden zwei zentrale Ansätze verwendet:

1. **Klassische Information-Retrieval-Metriken (Precision@k, Recall@k)** \\
   Diese Metriken messen die Qualität der Dokumentensuche. Precision@k gibt an, wie viele der Top-k zurückgegebenen Dokumente relevant sind, während Recall@k den Anteil der insgesamt relevanten Dokumente misst, der in den Top-k enthalten ist. Die Relevanz wurde zweistufig definiert:
   - \textit{Mode A (streng)}: Nur Dokumente mit \texttt{relevanz=hoch} gelten als relevant.
   - \textit{Mode B (moderat)}: Dokumente mit \texttt{relevanz=hoch} oder \texttt{relevanz=mittel} gelten als relevant.

2. **LLM-basierte Bewertung der Antwortqualität (LLM-as-a-Judge)** \\
   Für die Bewertung der Antwortqualität wurde ein lokal ausgeführtes LLM der LLaMA-Familie (\texttt{mistral:latest}) verwendet. Das Modell fungiert als unabhängiger Gutachter und bewertet die generierten Antworten auf Basis eines strukturierten Prompts anhand von vier Kriterien auf einer Skala von 1 (schlecht) bis 5 (sehr gut):

   \begin{description}
     \item[Correctness (Korrektheit):] Ist die Antwort fachlich korrekt basierend auf den Quellen?
     \item[Completeness (Vollständigkeit):] Beantwortet die Antwort die Frage vollständig?
     \item[Faithfulness (Faktentreue):] Bleibt die Antwort treu zu den gegebenen Quellen (keine Halluzinationen)?
     \item[Clarity (Klarheit):] Ist die Antwort klar, strukturiert und verständlich?
   \end{description}

   Durch den Einsatz eines separaten Modells wird eine objektivere Einschätzung der Antwortqualität ermöglicht, da die Bewertung unabhängig vom generierenden System erfolgt.

Die Ergebnisse wurden in JSON-Dateien gespeichert und anschließend automatisiert ausgewertet. Die Auswertung erfolgt getrennt für Kontextfragen und Out-of-Context-Fragen, um ein differenziertes Bild der Systemleistung zu erhalten.

---

## 6.2 Retrieval-Evaluation

### 6.2.1 Ergebnisse der Kontextfragen

Zur Bewertung der Suchqualität wurden die Metriken Precision@k und Recall@k mit k=3 verwendet. Die Berechnung erfolgt gemäß:

- Precision@k = relevante Treffer in Top-k / k
- Recall@k = relevante Treffer in Top-k / alle relevanten Treffer

Die Implementierung der Metriken ist in Listing 10 dargestellt.

\begin{table}[h]
\centering
\caption{Precision@3 und Recall@3 (Kontextfragen, N=12)}
\label{tab:retrieval}
\begin{tabular}{lcc}
\toprule
\textbf{Relevanz-Definition} & \textbf{Precision@3} & \textbf{Recall@3} \\
\midrule
Mode A (nur \texttt{hoch}) & 0,00 & 0,00 \\
Mode B (\texttt{hoch} + \texttt{mittel}) & 1,00 & 1,00 \\
\bottomrule
\end{tabular}
\end{table}

Die Ergebnisse zeigen ein zweigeteiltes Bild: Jede Kontextfrage liefert genau fünf Quellen, wobei die erste Quelle stets mit \texttt{relevanz=hoch} und die Quellen zwei und drei mit \texttt{relevanz=mittel} bewertet sind. Bei Mode A (streng) wird das einzige \texttt{hoch}-Dokument durch die Sortierung nach Score nicht in den Top-3 positioniert, was zu Precision=0 führt. Bei Mode B (moderat) werden alle relevanten Dokumente in den Top-3 gefunden, was einer perfekten Precision und Recall von 1,0 entspricht.

Dies deutet darauf hin, dass die Relevanzbewertung der Quellen in der Praxis zwischen \texttt{hoch} und \texttt{mittel} innerhalb der Top-3 kategorial unterscheidet, aber nicht beabsichtigt, mehrere \texttt{hoch}-Dokumente pro Query zu liefern.

### 6.2.2 Retrieval bei Out-of-Context-Fragen

Bei fachfremden Fragen liefert das hybride Retrieval ebenfalls fünf Chunks zurück. Diese weisen jedoch niedrige Scores auf und sind lediglich formal korrekt (höchste lexikalische Ähnlichkeit), ohne inhaltlich hilfreich zu sein. Dieses Verhalten ist systembedingt erwartbar, da die BM25-Komponente stets Ergebnisse auf Basis von Wortüberlappungen liefert, selbst wenn keine semantische Relevanz vorliegt. Die Out-of-Context-Fragen werden daher nicht in die Precision/Recall-Berechnung einbezogen, da eine relevante Grundwahrheit (\texttt{relevanz}) im Sinne des Retrievals nicht definiert werden kann.

---

## 6.3 Ergebnisse der Antwortbewertung

Die nachfolgenden Ergebnisse basieren auf der Bewertung durch das \texttt{mistral:latest}-Modell als LLM-Judge. Zur Veranschaulichung zeigt Listing 11 ein exemplarisches Bewertungsergebnis für eine einzelne Anfrage.

\lstinputlisting[caption=Beispielhafte Bewertung durch LLM-as-a-Judge, label=lst:judge]{listings/judge_example.json}

Das Beispiel zeigt, dass die Antwort in Bezug auf Vollständigkeit, Faktentreue und Verständlichkeit sehr gut bewertet wird, während Defizite in der Korrektheit bestehen.

### 6.3.1 Antwortqualität bei Kontextfragen

\begin{table}[h]
\centering
\caption{LLM-as-a-Judge-Ergebnisse für Kontextfragen (N=12)}
\label{tab:judge_ctx}
\begin{tabular}{lcc}
\toprule
\textbf{Kriterium} & \textbf{∅ / 5,0} & \textbf{Interpretation} \\
\midrule
Correctness & 2,58 & 6 von 12 Antworten verweigert ("keine Antwort gefunden") \\
Completeness & 4,08 & Bei beantworteten Fragen meist vollständig \\
Faithfulness & 5,00 & Keine Halluzinationen -- perfekte Kontexttreue \\
Clarity & 4,75 & Klare Strukturierung (Tabellen, Aufzählungen) \\
\bottomrule
\end{tabular}
\end{table}

Besonders auffällig ist die hohe Correctness-Spanne: Vier Fragen erreichen 5,0 Punkte, während sechs Fragen nur 1,0 Punkte erhalten. Die Ursache liegt in der Antwortverweigerung des Systems: In sechs Fällen generiert das System die Antwort "Ich konnte keine passende Antwort in den bereitgestellten Dokumenten finden", obwohl relevante Quellen im Retrieval-Ergebnis vorhanden waren. Der Faithfulness-Check, der als zweiter LLM-Durchlauf die Quellenbasierung der Antwort prüft, klassifiziert die Quellen in diesen Fällen als unzureichend und verwirft die Antwort.

### 6.3.2 Robustheit bei Out-of-Context-Fragen

\begin{table}[h]
\centering
\caption{LLM-as-a-Judge-Ergebnisse für Out-of-Context-Fragen (N=5)}
\label{tab:judge_ooc}
\begin{tabular}{lcc}
\toprule
\textbf{Kriterium} & \textbf{∅ / 5,0} & \textbf{Interpretation} \\
\midrule
Correctness & 1,40 & System verweigert korrekt -- aber Judge wertet "keine Antwort" als falsch \\
Completeness & 4,40 & Ablehnung als vollständig bewertet \\
Faithfulness & 5,00 & Keine Halluzination bei fachfremden Fragen \\
Clarity & 4,80 & Klare, höfliche Ablehnung \\
\bottomrule
\end{tabular}
\end{table}

Das System verhält sich bei fachfremden Fragen robust: Es erfindet keine Antworten, sondern verweigert konsistent die Beantwortung. Die vergleichsweise niedrige Correctness von 1,40 ist als Artefakt der Bewertungsmethodik zu interpretieren -- der LLM-Judge bewertet eine Antwortverweigerung als "fachlich inkorrekt", was in diesem Kontext jedoch das gewünschte Verhalten darstellt. Die Frage "Ich brauche einen Laptop -- wie beschaffe ich diesen?" (Nr. 17) bildet eine Ausnahme: Sie erhält Correctness=3,0, da die Antwort teilweise auf allgemeine Beschaffungsprozesse eingeht, ohne konkret zu werden.

### 6.3.3 Gesamtergebnis

\begin{table}[h]
\centering
\caption{Gesamtergebnis LLM-as-a-Judge (N=17)}
\label{tab:judge_all}
\begin{tabular}{lc}
\toprule
\textbf{Kriterium} & \textbf{∅ / 5,0} \\
\midrule
Correctness & 2,24 \\
Completeness & 4,18 \\
Faithfulness & 5,00 \\
Clarity & 4,76 \\
\midrule
\textbf{Overall} & \textbf{4,04} \\
\bottomrule
\end{tabular}
\end{table}

Die Evaluation zeigt, dass die generierten Antworten insbesondere in den Kategorien Faktentreue (5,0) und Klarheit (4,76) sehr hohe Werte erreichen. Die Vollständigkeit (4,18) ist gut, wird aber durch die teilweise Antwortverweigerung gedrückt. Die Korrektheit (2,24) stellt die größte Schwachstelle dar. Insgesamt ergibt sich ein durchschnittlicher Score von 4,04, was auf eine solide Gesamtqualität hinweist.

---

## 6.4 Diskussion

Die Kombination aus Retrieval-Metriken und LLM-basierter Bewertung ermöglicht eine umfassende Analyse des Systems. Während Precision und Recall die Qualität der Dokumentensuche messen, bewertet der LLM-as-a-Judge die tatsächliche Nutzbarkeit der generierten Antworten.

### 6.4.1 Stärken des Systems

1. **Halluzinationsfreiheit (Faithfulness 5,0):** Das System erfindet keine Fakten. Dies ist für den Einsatz im Vergaberecht, wo rechtliche Konsequenzen aus Fehlinformationen entstehen können, von zentraler Bedeutung.

2. **Retrieval-Qualität (P@3=1,00 in Mode B):** Die hybride Suche findet zuverlässig relevante Dokumente. Alle Top-3-Quellen sind zumindest mittelrelevant.

3. **Robustheit:** Bei fachfremden Fragen (Mondlandung, Pasta kochen) verweigert das System zuverlässig die Antwort -- es kommt zu keinem "Ausbrechen" oder Halluzinieren.

4. **Klarheit (4,76):** Die Antworten sind gut strukturiert, verwenden Tabellen und Aufzählungen und sind für Nutzer ohne juristische Vorbildung verständlich.

### 6.4.2 Schwächen und Verbesserungspotenzial

1. **Correctness (2,24):** Die Hauptschwachstelle ist die übervorsichtige Antwortgenerierung. Der Faithfulness-Check ist zu restriktiv konfiguriert: In 6 von 12 Kontextfragen werden vorhandene Quellen als unzureichend eingestuft, woraufhin das System die Antwort verwirft.

2. **Trade-off Correctness vs. Faithfulness:** Der Faithfulness-Check führt einen zweiten LLM-Durchlauf durch, der prüft, ob die generierte Antwort durch die Quellen gedeckt ist. Bei Unsicherheit wird die Antwort verworfen. Dieses binäre Verhalten (Antwort oder Verweigerung) ist für das Vergaberecht grundsätzlich sinnvoll, führt aber zu einer Correctness-Einbuße von ca. 50 Prozentpunkten.

3. **Optimierungsmöglichkeiten:**
   - \textit{Abgestufter Schwellwert}: Statt einer binären Verweigerung könnte das System eine abgestufte Konfidenz ausgeben (z. B. "Die Antwort basiert auf folgenden Quellen, jedoch mit eingeschränkter Sicherheit").
   - \textit{Prompt-Optimierung}: Der Prompt des Faithfulness-Checks könnte toleranter formuliert werden, ohne die Halluzinationsfreiheit aufzugeben.
   - \textit{Multi-Modell-Judge}: Die Verwendung mehrerer Bewertungsmodelle könnte die Objektivität des LLM-Judges erhöhen.

### 6.4.3 Grenzen der Evaluation

1. **Subjektiver LLM-Judge:** Die Bewertung durch \texttt{mistral:latest} ist nicht vollständig objektiv. Das Modell kann eigene Verzerrungen in die Bewertung einbringen. Eine Validierung durch menschliche Experten wäre wünschenswert.

2. **Stichprobengröße:** Mit 17 Fragen ist der Datensatz relativ klein. Eine größere Stichprobe (50--100 Fragen) würde statistisch robustere Aussagen ermöglichen.

3. **Einmodell-Bewertung:** Es wurde nur \texttt{mistral:latest} als Judge verwendet. Der Einsatz mehrerer Modelle (z. B. \texttt{gemma3:4b}, \texttt{gpt-oss:20b}) könnte die Reliabilität erhöhen.

4. **Fehlende Nutzerstudie:** Die Evaluation misst technische Metriken, aber nicht die tatsächliche Nutzerzufriedenheit. Eine qualitative Studie mit Fachanwendern aus dem Vergaberecht würde zusätzliche Erkenntnisse liefern.

---

## 6.5 Zusammenfassung

Zusammenfassend zeigt die Evaluation, dass das entwickelte System sowohl relevante Informationen zuverlässig identifizieren als auch qualitativ hochwertige Antworten generieren kann. Die Retrieval-Qualität ist mit Precision@3=1,00 (Mode B) exzellent. Das System halluziniert nicht (Faithfulness 5,0) und verweigert bei fachfremden Fragen zuverlässig.

Die größte Herausforderung besteht im Trade-off zwischen Korrektheit und Faktentreue: Der Faithfulness-Check verhindert zwar zuverlässig Halluzinationen, führt aber in 50 % der Kontextfragen zur Antwortverweigerung. Dies senkt die Correctness auf 2,24, während die Faithfulness bei perfekten 5,0 liegt.

Die Kombination aus quantitativer und qualitativer Bewertung stellt eine fundierte Grundlage für die Beurteilung der Systemleistung dar. Trotz der genannten Einschränkungen bestätigen die Ergebnisse die Eignung des RAG-Ansatzes für das Vergaberecht: Das System liefert rechtssichere, nachvollziehbare und halluzinationsfreie Antworten, wenn der Faithfulness-Check entsprechend konfiguriert ist.
