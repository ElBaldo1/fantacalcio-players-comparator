# Fanta-Analyst Trade Analyzer

Full-stack trade analyzer for Fantacalcio players. The backend scrapes match stats and scores players; the frontend compares two trade sides and renders verdicts and charts.

## File Tree
```
backend/
  main.py
  scraper.py
  scoring.py
  test_trade.py
  requirements.txt
frontend/
  index.html
  package.json
  postcss.config.cjs
  tailwind.config.js
  tsconfig.json
  vite.config.ts
  src/
    App.tsx
    index.css
    main.tsx
    types.ts
    components/
      SelectionScreen.tsx
      ResultsScreen.tsx
      charts/
        TradeTotalsChart.tsx
        PlayerValueChart.tsx
        PlayerTrendChart.tsx
```

## Backend (FastAPI)
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

## Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

Optional API override:
```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Notes
- Charts use fixed y-axis tick intervals (0.5 for FV-related charts and 10 for 0-100 scores).
- Backend validates trade constraints (same number of players and role counts).
- Trade value uses season and next-3 horizons with explicit formulas.
