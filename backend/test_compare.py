import os
import sys
from typing import Any, Dict, List

import requests


def assert_keys(obj: Dict[str, Any], keys: List[str], path: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise AssertionError(f"Missing keys at {path}: {missing}")


def validate_schema(data: Dict[str, Any]) -> None:
    assert_keys(data, ["player1", "player2"], "root")
    for idx, key in enumerate(["player1", "player2"], start=1):
        player = data[key]
        assert_keys(player, ["name", "metrics", "final_score", "components", "matches"], f"{key}")
        assert_keys(
            player["metrics"],
            ["avg_voto", "avg_fv", "bonus_impact", "consistency", "recent_trend", "matches_n"],
            f"{key}.metrics",
        )
        assert_keys(
            player["components"],
            ["Avg Voto", "Bonus Impact", "Consistency", "Recent Trend", "Goal Involvement"],
            f"{key}.components",
        )
        if not isinstance(player["matches"], list):
            raise AssertionError(f"{key}.matches must be a list")


def main() -> None:
    base_url = os.getenv("API_URL", "http://localhost:8000")
    url1 = os.getenv(
        "URL1",
        "https://www.fantacalcio.it/serie-a/squadre/napoli/di-lorenzo/2816",
    )
    url2 = os.getenv(
        "URL2",
        "https://www.fantacalcio.it/serie-a/squadre/inter/lautaro-martinez/2934",
    )

    payload = {"url1": url1, "url2": url2}
    response = requests.post(f"{base_url}/compare", json=payload, timeout=30)
    if response.status_code != 200:
        print(f"Request failed: {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json()
    validate_schema(data)
    print("Schema validation passed")


if __name__ == "__main__":
    main()
