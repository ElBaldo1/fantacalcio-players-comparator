from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from scraper import FantacalcioScraper
from scoring import build_player_payload


class PlayerInput(BaseModel):
    url: HttpUrl
    role: str


class TradeRequest(BaseModel):
    left: List[PlayerInput]
    right: List[PlayerInput]


app = FastAPI(title="Fanta-Analyst Trade API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE: Dict[str, tuple] = {}


def role_counts(players: List[PlayerInput]) -> Dict[str, int]:
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in players:
        if p.role in counts:
            counts[p.role] += 1
    return counts


def validate_trade(left: List[PlayerInput], right: List[PlayerInput]) -> None:
    if len(left) != len(right):
        raise HTTPException(
            status_code=422,
            detail=f"Player count mismatch: Left={len(left)}, Right={len(right)}",
        )
    left_counts = role_counts(left)
    right_counts = role_counts(right)
    if left_counts != right_counts:
        raise HTTPException(
            status_code=422,
            detail=f"Role counts mismatch. Left: {left_counts} | Right: {right_counts}",
        )


def fetch_cached(url: str) -> tuple:
    if url in CACHE:
        return CACHE[url]
    name, df = FantacalcioScraper(url).fetch()
    CACHE[url] = (name, df)
    return name, df


def driver_contributions(left: list, right: list, value_key: str) -> List[Dict[str, float]]:
    left_role_avgs = {}
    right_role_avgs = {}
    for role in ["GK", "DEF", "MID", "FWD"]:
        left_vals = [p[value_key] for p in left if p["role"] == role]
        right_vals = [p[value_key] for p in right if p["role"] == role]
        left_role_avgs[role] = sum(left_vals) / len(left_vals) if left_vals else None
        right_role_avgs[role] = sum(right_vals) / len(right_vals) if right_vals else None

    contributions = []
    for p in right:
        baseline = left_role_avgs.get(p["role"])
        if baseline is None:
            continue
        contributions.append({"player": f"{p['name']} (Right)", "delta": p[value_key] - baseline})
    for p in left:
        baseline = right_role_avgs.get(p["role"])
        if baseline is None:
            continue
        contributions.append({"player": f"{p['name']} (Left)", "delta": p[value_key] - baseline})

    contributions.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return contributions[:3]


@app.post("/api/trade/evaluate")
async def evaluate_trade(payload: TradeRequest) -> Dict[str, object]:
    validate_trade(payload.left, payload.right)

    left_players = []
    right_players = []

    try:
        for p in payload.left:
            name, df = fetch_cached(str(p.url))
            left_players.append(build_player_payload(name, df, p.role))
        for p in payload.right:
            name, df = fetch_cached(str(p.url))
            right_players.append(build_player_payload(name, df, p.role))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    left_total_season = sum(p["season_value"] for p in left_players)
    right_total_season = sum(p["season_value"] for p in right_players)
    delta_season = right_total_season - left_total_season

    left_total_next3 = sum(p["next3_value"] for p in left_players)
    right_total_next3 = sum(p["next3_value"] for p in right_players)
    delta_next3 = right_total_next3 - left_total_next3

    threshold = 5.0
    season_verdict = (
        "Right side wins" if delta_season > threshold else "Left side wins" if delta_season < -threshold else "Roughly balanced"
    )
    next3_verdict = (
        "Right side wins" if delta_next3 > threshold else "Left side wins" if delta_next3 < -threshold else "Roughly balanced"
    )

    return {
        "left": left_players,
        "right": right_players,
        "totals": {
            "season": {
                "left": left_total_season,
                "right": right_total_season,
                "delta": delta_season,
                "verdict": season_verdict,
                "drivers": driver_contributions(left_players, right_players, "season_value"),
            },
            "next3": {
                "left": left_total_next3,
                "right": right_total_next3,
                "delta": delta_next3,
                "verdict": next3_verdict,
                "drivers": driver_contributions(left_players, right_players, "next3_value"),
            },
        },
    }
