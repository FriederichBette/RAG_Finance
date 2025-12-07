# Finance RAG Bot with GPT-2

Dieses Projekt demonstriert ein **lokales Retrieval-Augmented-Generation (RAG) System** für Unternehmensfinanzdaten.  

Der Bot kombiniert:

- **FAISS** für schnelles Retrieval relevanter Dokumente
- **Sentence-Transformers** für semantische Embeddings
- **GPT-2-medium** zur Generierung kurzer, konservativer Antworten

Das Ziel ist, Fragen zu Finanzkennzahlen wie **Revenue, Profit oder Growth** von Unternehmen präzise zu beantworten. 
---

## Features

- Fest kodierte Unternehmensdaten:
  - AlphaCorp, BetaInc, GammaLLC, DeltaSolutions, EpsilonTech
- Interaktive Abfrage im Terminal
- GPT-2 antwortet kurz, konservativ und faktenbasiert
- Komplett lokal ausführbar nach einmaligem Modell-Download
- Einfach erweiterbar um neue Daten oder Modelle

---

## Installation

1.) **Repository klonen**


git clone https://github.com/FriederichBette/finance-rag-gpt2.git
cd finance-rag-gpt2


2.) Virtuelle Umgebung (optional) erstellen

python -m venv env
Linux/Mac:
source env/bin/activate
Windows:
env\Scripts\activate


3.) Dependencies installieren

pip install -r requirements.txt


4.) Script starten

python finance_rag_gpt2.py

Beispiel
Ask a finance question (or type 'exit'): Which company has the highest revenue?
FinanceBot says: DeltaSolutions: Revenue $400M

Ask a finance question (or type 'exit'): Which company has the fastest growth?
FinanceBot says: BetaInc: Growth 12.0%

