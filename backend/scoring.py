import math
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def safe_mean(series: pd.Series) -> float:
    return float(series.dropna().mean()) if series.dropna().size > 0 else float("nan")


def scale_positive(value: float, low: float, high: float) -> float:
    if np.isnan(value):
        return float("nan")
    if high == low:
        return 50.0
    clipped = max(low, min(high, value))
    return 100.0 * (clipped - low) / (high - low)


def scale_inverted(value: float, low: float, high: float) -> float:
    if np.isnan(value):
        return float("nan")
    if high == low:
        return 50.0
    clipped = max(low, min(high, value))
    return 100.0 * (1.0 - (clipped - low) / (high - low))


def scale_trend(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if np.isnan(value):
        return float("nan")
    clipped = max(low, min(high, value))
    return 100.0 * (clipped - low) / (high - low)


def parse_bonus_events(value: Any) -> int:
    """
    Proxy bonus/malus count per match:
    - Count non-zero numeric tokens in the bonus/malus field (e.g. '+3', '-0.5').
    - If no numeric tokens but text exists, count 1 as a weak signal.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    text = str(value).strip().lower()
    if text in {"", "-", "s.v.", "s.v"}:
        return 0
    tokens = re.findall(r"[+-]?\d+(?:[.,]\d+)?", text)
    if tokens:
        count = 0
        for token in tokens:
            try:
                if float(token.replace(",", ".")) != 0:
                    count += 1
            except ValueError:
                continue
        return count
    return 1


def calculate_player_score(
    df: pd.DataFrame, optional_stats: Dict[str, Optional[int]]
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Compute score components and final score with reliability adjustment.
    Returns: final_score, components (0-100), debug values.
    """
    fv_series = df["fv"]
    voto_series = df["voto"]

    avg_voto = safe_mean(voto_series)
    avg_fv = safe_mean(fv_series)
    bonus_impact = safe_mean(fv_series - voto_series)
    consistency_raw = (
        float(fv_series.dropna().std()) if fv_series.dropna().size > 0 else float("nan")
    )

    last5 = fv_series.dropna().tail(5)
    recent_trend_raw = (
        float(last5.mean() - fv_series.dropna().mean()) if last5.size > 0 else float("nan")
    )

    matches_n = int(fv_series.dropna().shape[0])

    involvement_raw = float("nan")
    goals = optional_stats.get("gol") if optional_stats else None
    assists = optional_stats.get("assist") if optional_stats else None
    if goals is not None or assists is not None:
        total_actions = (goals or 0) + (assists or 0)
        involvement_raw = total_actions / matches_n if matches_n > 0 else float("nan")
    elif "bonus_malus" in df.columns:
        total_events = int(df["bonus_malus"].apply(parse_bonus_events).sum())
        involvement_raw = total_events / matches_n if matches_n > 0 else float("nan")

    components = {
        "Avg Voto": scale_positive(avg_voto, 5.0, 7.5),
        "Bonus Impact": scale_positive(bonus_impact, -0.2, 2.0),
        "Consistency": scale_inverted(consistency_raw, 0.2, 2.0),
        "Recent Trend": scale_trend(recent_trend_raw, -1.0, 1.0),
        "Goal Involvement": scale_positive(involvement_raw, 0.0, 1.0),
    }

    weights = {
        "Avg Voto": 0.35,
        "Bonus Impact": 0.30,
        "Consistency": 0.20,
        "Recent Trend": 0.10,
        "Goal Involvement": 0.05,
    }

    if np.isnan(components["Goal Involvement"]):
        removed = weights.pop("Goal Involvement")
        remaining_total = 1.0 - removed
        for key in list(weights.keys()):
            weights[key] = weights[key] / remaining_total

    valid_items = {k: v for k, v in components.items() if not np.isnan(v)}
    if valid_items:
        used_weights = {k: weights.get(k, 0.0) for k in valid_items}
        weight_sum = sum(used_weights.values())
        if weight_sum > 0:
            used_weights = {k: v / weight_sum for k, v in used_weights.items()}
        weighted_sum = sum(valid_items[k] * used_weights[k] for k in valid_items)
    else:
        weighted_sum = 50.0

    reliability = min(1.0, matches_n / 10.0)
    final_score = reliability * weighted_sum + (1.0 - reliability) * 50.0

    debug = {
        "avg_voto": avg_voto,
        "avg_fv": avg_fv,
        "bonus_impact": bonus_impact,
        "consistency_raw": consistency_raw,
        "recent_trend_raw": recent_trend_raw,
        "involvement_raw": involvement_raw,
        "matches_n": matches_n,
        "reliability": reliability,
        "weighted_sum": weighted_sum,
    }

    return final_score, components, debug


def to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def build_player_payload(
    name: str, df: pd.DataFrame, optional_stats: Dict[str, Optional[int]]
) -> Dict[str, Any]:
    avg_voto = safe_mean(df["voto"])
    avg_fv = safe_mean(df["fv"])
    bonus_impact = safe_mean(df["fv"] - df["voto"])
    consistency = (
        float(df["fv"].dropna().std()) if df["fv"].dropna().size > 0 else float("nan")
    )
    last5 = df["fv"].dropna().tail(5)
    recent_trend = (
        float(last5.mean() - df["fv"].dropna().mean()) if last5.size > 0 else float("nan")
    )
    matches_n = int(df["fv"].dropna().shape[0])

    final_score, components, debug = calculate_player_score(df, optional_stats)

    matches = [
        {
            "giornata": to_json_safe(row.giornata),
            "fv": to_json_safe(row.fv),
        }
        for row in df[["giornata", "fv"]].itertuples(index=False)
        if not (pd.isna(row.giornata) or pd.isna(row.fv))
    ]

    payload = {
        "name": name,
        "metrics": {
            "avg_voto": to_json_safe(avg_voto),
            "avg_fv": to_json_safe(avg_fv),
            "bonus_impact": to_json_safe(bonus_impact),
            "consistency": to_json_safe(consistency),
            "recent_trend": to_json_safe(recent_trend),
            "matches_n": matches_n,
        },
        "final_score": to_json_safe(final_score),
        "components": {k: to_json_safe(v) for k, v in components.items()},
        "matches": matches,
        "debug": {k: to_json_safe(v) for k, v in debug.items()},
    }
    return payload
