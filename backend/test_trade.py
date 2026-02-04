import os
import sys
from typing import Any, Dict

import requests


def assert_keys(obj: Dict[str, Any], keys: list, path: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise AssertionError(f"Missing keys at {path}: {missing}")


def validate_schema(data: Dict[str, Any]) -> None:
    assert_keys(data, ["left", "right", "totals"], "root")
    assert_keys(data["totals"], ["season", "next3"], "totals")
    for side in ["left", "right"]:
        if not isinstance(data[side], list):
            raise AssertionError(f"{side} must be a list")
        for player in data[side]:
            assert_keys(
                player,
                [
                    "name",
                    "role",
                    "final_score",
                    "predicted_next_fv",
                    "predicted_next3_fv",
                    "season_value",
                    "next3_value",
                    "series",
                ],
                f"{side}.player",
            )


def main() -> None:
    base_url = os.getenv("API_URL", "http://localhost:8000")
    payload = {
        "left": [
            {
                "url": "https://www.fantacalcio.it/serie-a/squadre/napoli/di-lorenzo/2816",
                "role": "DEF",
            }
        ],
        "right": [
            {
                "url": "https://www.fantacalcio.it/serie-a/squadre/inter/frattesi/2848",
                "role": "MID",
            }
        ],
    }

    response = requests.post(f"{base_url}/api/trade/evaluate", json=payload, timeout=30)
    if response.status_code != 200:
        print(f"Request failed: {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json()
    validate_schema(data)
    print("Schema validation passed")


if __name__ == "__main__":
    main()
