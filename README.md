# Fanta-Analyst Trade Analyzer

## English
This project is built for Fantacalcio managers who want a quick, visual way to evaluate trades. You paste player season URLs from fantacalcio.it, set roles, and get a clean season-long verdict with charts and player-level context. The UI is **in Italian** because it targets Italian Fantacalcio users. Player URLs must be **fantacalcio.it season player pages**.


Example URL:
`https://www.fantacalcio.it/serie-a/squadre/sassuolo/pinamonti/2038/2025-26`

### Backend (FastAPI)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Test script:
```bash
python backend/test_trade.py
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

Optional API override:
```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

### Trade validity
Valid trades require **the same number of players** on both sides and **the same role counts** (GK/DEF/MID/FWD).

### Notes
- Charts use fixed y-axis tick intervals (0.5 for FV-related charts and 10 for 0-100 scores).
- Backend validates trade constraints (same number of players and role counts).
- The UI focuses on season-long value for the trade verdict.


---

## Italiano
Questo progetto è pensato per chi gioca al Fantacalcio e vuole valutare rapidamente gli scambi. Inserisci gli URL stagione dei giocatori da fantacalcio.it, imposta i ruoli e ottieni un verdetto stagionale con grafici e dettagli per giocatore. L'interfaccia è **in italiano** perché l'app è pensata per utenti Fantacalcio in Italia. Gli URL dei giocatori devono essere **pagine stagione di fantacalcio.it**.


URL di esempio:
`https://www.fantacalcio.it/serie-a/squadre/sassuolo/pinamonti/2038/2025-26`

### Backend (FastAPI)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Script di test:
```bash
python backend/test_trade.py
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

Override opzionale API:
```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

### Validità scambio
Lo scambio è valido solo se **il numero di giocatori è uguale** tra i due lati e **i ruoli coincidono** (GK/DEF/MID/FWD).

### Note
- I grafici usano intervalli fissi sull'asse Y (0.5 per FV e 10 per gli score 0-100).
- Il backend valida la correttezza dello scambio (numero giocatori e ruoli).
- L'interfaccia mostra il valore stagione per il verdetto dello scambio.
