#!/usr/bin/env python3
"""
Liest die korrigierte relevanz_korrektur.json und schreibt die Werte
zurück in die JSON-Dateien (best_chunks_and_answer.json + out_kontext).
"""
import json, sys
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
KORREKTUR = HERE / "relevanz_korrektur.json"

with open(KORREKTUR, encoding="utf-8") as f:
    eintraege = json.load(f)

# Prüfen ob sich was geändert hat
aenderungen = [e for e in eintraege if e["relevanz_neu"] != e["relevanz_alt"]]
if not aenderungen:
    print("⚠  Keine Änderungen gefunden – 'relevanz_neu' entspricht überall 'relevanz_alt'.")
    ok = input("Trotzdem fortfahren? (j/N): ").strip().lower()
    if ok != "j":
        print("Abbruch.")
        sys.exit(0)

# Gruppieren nach datei → frage_nr → rang
updates = defaultdict(lambda: defaultdict(dict))
for e in eintraege:
    neu = e["relevanz_neu"].strip().lower()
    if neu not in ("hoch", "mittel", "niedrig"):
        print(f"⚠  Ungültig: Frage {e['frage_nr']}, Rang {e['rang']}: '{neu}' – übersprungen")
        continue
    updates[e["datei"]][e["frage_nr"]][e["rang"]] = neu

# In JSON-Dateien schreiben
for datei, fragen in updates.items():
    path = HERE / datei
    if not path.exists():
        print(f"❌ Nicht gefunden: {path}")
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for qi, entry in enumerate(data["entries"], 1):
        if qi not in fragen:
            continue
        for q in entry["quellen"]:
            rang = q["quelle"]
            if rang in fragen[qi]:
                alt = q["relevanz"]
                neu = fragen[qi][rang]
                if alt != neu:
                    q["relevanz"] = neu
                    print(f"  {datei} | Frage {qi} | Rang {rang}: {alt} → {neu}")
                    count += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ {datei}: {count} Änderungen")

print("\nFertig. Jetzt generate_plots.py neu laufen lassen.")
