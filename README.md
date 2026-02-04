# Fanta-Analyst Pro

Full-stack Fantacalcio player comparison with a FastAPI backend and React (Vite) frontend.

## File Tree
```
backend/
  main.py
  scraper.py
  scoring.py
  test_compare.py
  requirements.txt
frontend/
  index.html
  package.json
  tsconfig.json
  vite.config.ts
  src/
    App.tsx
    index.css
    main.tsx
app.py
README.md
pyproject.toml
```

## Backend (FastAPI)
### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### Run
```bash
uvicorn backend.main:app --reload --port 8000
```

### Test Script
```bash
python backend/test_compare.py
```

You can override URLs and API endpoint:
```bash
API_URL=http://localhost:8000 URL1=<url1> URL2=<url2> python backend/test_compare.py
```

## Frontend (React + Vite)
### Install
```bash
cd frontend
npm install
```

### Run
```bash
npm run dev
```

Optional API URL override:
```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Notes
- The backend uses requests + BeautifulSoup + pandas.read_html for scraping.
- If the page structure changes or the table is missing, the API returns a 422 error.
- Goal involvement uses goals+assists if available; otherwise a bonus/malus proxy per match.
