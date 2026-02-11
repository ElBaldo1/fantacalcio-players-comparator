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

### How the algorithm works
1. Data collection
- The backend scrapes match-by-match player data from the Fantacalcio season page: `giornata`, `voto`, `fv`, plus `Quotazione Classic` and `FVM Classic` when available.
- It can also fetch up to two previous seasons to estimate historical trajectory.

2. Core metrics per player
- `avg_voto`: average match rating.
- `avg_fv`: average fantasy rating.
- `bonus_impact`: average `fv - voto`.
- `consistency_raw`: standard deviation of FV (lower is better).
- `recent_trend_raw`: difference between last 5 FV average and full-season FV average.
- `availability_ratio`: matches with FV / max matchday.
- `value_efficiency`: `FVM Classic / Quotazione Classic` (if available).

3. Component scores (0-100)
- Metrics are normalized to 0-100 with role-agnostic scaling functions.
- Main components: `Avg Voto`, `Bonus Impact`, `Consistency`, `Recent Trend`, `Availability`, `FVM`, `Efficienza Valore`.

4. Role-based weighted score
- Each role (`GK`, `DEF`, `MID`, `FWD`) has different weights.
- If FV looks unreliable (FV almost always equal to Voto), `Bonus Impact` weight is reduced and redistributed.
- The model computes:
- `seasonal_score` from full season.
- `recent_score` from last 5 valid FV rows.
- `smoothed_score = 0.7 * seasonal_score + 0.3 * recent_score`.
- `reliability = min(1, matches_n / 10)`.
- `final_score = reliability * smoothed_score + (1 - reliability) * 50`.

5. Prediction and uncertainty
- Prediction uses weighted moving average on last up to 5 FV values (recent matches have higher weight).
- Next-3 forecast:
- `predicted_next3 = clamp(predicted_fv + 0.5 * recent_trend, 4.0, 9.0)`.
- Uncertainty ranges:
- `range = prediction ± 0.84 * std_fv` (same logic for next-3).

6. Trade value metrics
- Season value:
- `season_value = final_score + 10*(availability-0.80) - 5*max(0, std_fv-1.2) + 0.5*trajectory_modifier`.
- Next-3 value:
- `next3_value = scale(predicted_next3) + 8*(availability-0.80) - 8*max(0, std_fv-1.1) + 0.3*trajectory_modifier`.
- Both values are clamped to `0-100`.

7. Trade verdict
- Totals are computed by summing player values per side.
- The side with the higher total value is favored; if totals are close, verdict is balanced.

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
Lo scambio è valido solo se **il numero di giocatori è uguale** tra i due lati e **i ruoli coincidono** (POR/DF/CC/ATT).

### Come funziona l'algoritmo
1. Raccolta dati
- Il backend estrae dalla pagina stagione di Fantacalcio i dati partita per partita: `giornata`, `voto`, `fv`, più `Quotazione Classic` e `FVM Classic` quando disponibili.
- Può anche scaricare fino a due stagioni precedenti per stimare la traiettoria storica.

2. Metriche base per giocatore
- `avg_voto`: media voto.
- `avg_fv`: media fantavoto.
- `bonus_impact`: media di `fv - voto`.
- `consistency_raw`: deviazione standard del FV (più bassa è meglio).
- `recent_trend_raw`: differenza tra media FV ultime 5 e media FV stagionale.
- `availability_ratio`: partite con FV / giornata massima.
- `value_efficiency`: `FVM Classic / Quotazione Classic` (se disponibile).

3. Score componenti (0-100)
- Le metriche vengono normalizzate su scala 0-100 con funzioni di scaling comuni.
- Componenti principali: `Avg Voto`, `Bonus Impact`, `Consistency`, `Recent Trend`, `Availability`, `FVM`, `Efficienza Valore`.

4. Punteggio pesato per ruolo
- Ogni ruolo (`GK`, `DEF`, `MID`, `FWD`) ha pesi differenti.
- Se il FV risulta poco affidabile (FV quasi sempre uguale al Voto), il peso di `Bonus Impact` viene ridotto e redistribuito.
- Il modello calcola:
- `seasonal_score` sulla stagione completa.
- `recent_score` sulle ultime 5 righe valide di FV.
- `smoothed_score = 0.7 * seasonal_score + 0.3 * recent_score`.
- `reliability = min(1, matches_n / 10)`.
- `final_score = reliability * smoothed_score + (1 - reliability) * 50`.

5. Previsione e incertezza
- La previsione usa una media mobile pesata sugli ultimi massimo 5 FV (più peso alle partite recenti).
- Previsione prossime 3:
- `predicted_next3 = clamp(predicted_fv + 0.5 * recent_trend, 4.0, 9.0)`.
- Intervalli di incertezza:
- `range = previsione ± 0.84 * std_fv` (stessa logica per il next-3).

6. Metriche valore scambio
- Valore stagione:
- `season_value = final_score + 10*(availability-0.80) - 5*max(0, std_fv-1.2) + 0.5*trajectory_modifier`.
- Valore prossime 3:
- `next3_value = scale(predicted_next3) + 8*(availability-0.80) - 8*max(0, std_fv-1.1) + 0.3*trajectory_modifier`.
- Entrambi i valori sono limitati a `0-100`.

7. Verdetto scambio
- I totali vengono calcolati sommando i valori dei giocatori per ciascun lato.
- Il lato con valore totale più alto è favorito; se i totali sono vicini, il verdetto è equilibrato.

### Note
- I grafici usano intervalli fissi sull'asse Y (0.5 per FV e 10 per gli score 0-100).
- Il backend valida la correttezza dello scambio (numero giocatori e ruoli).
- L'interfaccia mostra il valore stagione per il verdetto dello scambio.
