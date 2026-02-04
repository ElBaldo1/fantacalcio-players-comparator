from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from scraper import FantacalcioScraper
from scoring import build_player_payload


class CompareRequest(BaseModel):
    url1: HttpUrl
    url2: HttpUrl


app = FastAPI(title="Fanta-Analyst API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/compare")
async def compare_players(payload: CompareRequest) -> Dict[str, dict]:
    try:
        name1, df1, opt1 = FantacalcioScraper(str(payload.url1)).fetch()
        name2, df2, opt2 = FantacalcioScraper(str(payload.url2)).fetch()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error.") from exc

    if df1.empty or df2.empty:
        raise HTTPException(
            status_code=422, detail="One of the players has no match data."
        )

    return {
        "player1": build_player_payload(name1, df1, opt1),
        "player2": build_player_payload(name2, df2, opt2),
    }
