from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scraper import FantacalcioScraper


ROLE_WEIGHTS = {
    "GK": {
        "Avg Voto": 0.45,
        "Bonus Impact": 0.00,
        "Consistency": 0.25,
        "Recent Trend": 0.10,
        "Availability": 0.20,
    },
    "DEF": {
        "Avg Voto": 0.35,
        "Bonus Impact": 0.15,
        "Consistency": 0.25,
        "Recent Trend": 0.10,
        "Availability": 0.15,
    },
    "MID": {
        "Avg Voto": 0.30,
        "Bonus Impact": 0.30,
        "Consistency": 0.15,
        "Recent Trend": 0.15,
        "Availability": 0.10,
    },
    "FWD": {
        "Avg Voto": 0.25,
        "Bonus Impact": 0.40,
        "Consistency": 0.10,
        "Recent Trend": 0.15,
        "Availability": 0.10,
    },
}


def safe_mean(series: pd.Series) -> float:
    return float(series.dropna().mean()) if series.dropna().size > 0 else float("nan")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scale_positive(value: float, low: float, high: float) -> float:
    if np.isnan(value):
        return float("nan")
    if high == low:
        return 50.0
    clipped = clamp(value, low, high)
    return 100.0 * (clipped - low) / (high - low)


def scale_inverted(value: float, low: float, high: float) -> float:
    if np.isnan(value):
        return float("nan")
    if high == low:
        return 50.0
    clipped = clamp(value, low, high)
    return 100.0 * (1.0 - (clipped - low) / (high - low))


def scale_trend(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if np.isnan(value):
        return float("nan")
    clipped = clamp(value, low, high)
    return 100.0 * (clipped - low) / (high - low)


def get_role_weights(role: str) -> Dict[str, float]:
    return ROLE_WEIGHTS.get(role, ROLE_WEIGHTS["MID"]).copy()


def adjust_weights_for_unreliable(weights: Dict[str, float], fv_unreliable: bool) -> Dict[str, float]:
    adjusted = weights.copy()
    if fv_unreliable:
        original_bonus = adjusted.get("Bonus Impact", 0.0)
        capped = min(original_bonus, 0.05)
        diff = original_bonus - capped
        adjusted["Bonus Impact"] = capped
        if diff > 0:
            other_keys = [k for k in adjusted.keys() if k != "Bonus Impact"]
            base = sum(adjusted[k] for k in other_keys)
            if base > 0:
                for k in other_keys:
                    adjusted[k] += diff * (adjusted[k] / base)
    total = sum(adjusted.values())
    if total > 0:
        for k in list(adjusted.keys()):
            adjusted[k] /= total
    return adjusted


def weighted_moving_average(values: List[float]) -> Tuple[float, List[int]]:
    if not values:
        return float("nan"), []
    n = len(values)
    weights = list(range(1, n + 1))
    weight_sum = sum(weights)
    prediction = sum(v * w for v, w in zip(values, weights)) / weight_sum
    return prediction, weights


def compute_prediction(df: pd.DataFrame) -> Dict[str, Any]:
    fv_series = df["fv"].dropna()
    matches_n = int(fv_series.shape[0])

    if matches_n == 0:
        return {
            "predicted_fv": float("nan"),
            "predicted_next3": float("nan"),
            "range_low": float("nan"),
            "range_high": float("nan"),
            "range_low_next3": float("nan"),
            "range_high_next3": float("nan"),
            "p_big_game": float("nan"),
            "p_bad_game": float("nan"),
            "low_confidence": True,
            "std_fv": float("nan"),
            "debug": {"last_values": [], "weights": [], "std_fv": float("nan")},
        }

    last_values = fv_series.tail(5).to_list()
    prediction, weights = weighted_moving_average(last_values)

    std_fv = float(fv_series.std()) if matches_n > 0 else float("nan")
    std_fv = max(std_fv, 0.15) if not np.isnan(std_fv) else float("nan")

    mean_all = float(fv_series.mean())
    mean_last5 = float(np.mean(last_values)) if last_values else mean_all
    recent_trend = mean_last5 - mean_all

    predicted_next3 = prediction + 0.5 * recent_trend
    predicted_next3 = clamp(predicted_next3, 4.0, 9.0)

    range_low = prediction - 0.84 * std_fv
    range_high = prediction + 0.84 * std_fv
    range_low_next3 = predicted_next3 - 0.84 * std_fv
    range_high_next3 = predicted_next3 + 0.84 * std_fv

    p_big_game = float((fv_series >= 7.0).mean()) if matches_n > 0 else float("nan")
    p_bad_game = float((fv_series < 6.0).mean()) if matches_n > 0 else float("nan")

    return {
        "predicted_fv": prediction,
        "predicted_next3": predicted_next3,
        "range_low": range_low,
        "range_high": range_high,
        "range_low_next3": range_low_next3,
        "range_high_next3": range_high_next3,
        "p_big_game": p_big_game,
        "p_bad_game": p_bad_game,
        "low_confidence": matches_n < 8,
        "std_fv": std_fv,
        "debug": {
            "last_values": last_values,
            "weights": weights,
            "std_fv": std_fv,
            "recent_trend": recent_trend,
        },
    }


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    fv_series = df["fv"]
    voto_series = df["voto"]

    avg_voto = safe_mean(voto_series)
    avg_fv = safe_mean(fv_series)
    bonus_impact = safe_mean(fv_series - voto_series)

    fv_std = float(fv_series.dropna().std()) if fv_series.dropna().size > 0 else float("nan")
    consistency_raw = max(fv_std, 0.15) if not np.isnan(fv_std) else float("nan")

    last5 = fv_series.dropna().tail(5)
    recent_trend_raw = (
        float(last5.mean() - fv_series.dropna().mean()) if last5.size > 0 else float("nan")
    )

    matches_n = int(fv_series.dropna().shape[0])
    max_matchday = int(df["giornata"].dropna().max()) if df["giornata"].dropna().size > 0 else 0
    availability_ratio = matches_n / max(1, max_matchday) if matches_n > 0 else float("nan")

    paired = df[["voto", "fv"]].dropna()
    fv_equals_voto_ratio = float("nan")
    if not paired.empty:
        fv_equals_voto_ratio = float((paired["fv"] - paired["voto"]).abs().lt(0.01).mean())

    return {
        "avg_voto": avg_voto,
        "avg_fv": avg_fv,
        "bonus_impact": bonus_impact,
        "consistency_raw": consistency_raw,
        "recent_trend_raw": recent_trend_raw,
        "matches_n": matches_n,
        "max_matchday": max_matchday,
        "availability_ratio": availability_ratio,
        "fv_std": fv_std,
        "fv_equals_voto_ratio": fv_equals_voto_ratio,
        "fv_unreliable": not np.isnan(fv_equals_voto_ratio) and fv_equals_voto_ratio > 0.8,
    }


def components_from_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "Avg Voto": scale_positive(metrics["avg_voto"], 5.0, 7.5),
        "Bonus Impact": scale_positive(metrics["bonus_impact"], -0.2, 2.0),
        "Consistency": scale_inverted(metrics["consistency_raw"], 0.2, 2.0),
        "Recent Trend": scale_trend(metrics["recent_trend_raw"], -1.0, 1.0),
        "Availability": scale_positive(metrics["availability_ratio"], 0.0, 1.0),
    }


def weighted_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    valid_items = {k: v for k, v in components.items() if not np.isnan(v)}
    if not valid_items:
        return 50.0
    used_weights = {k: weights.get(k, 0.0) for k in valid_items}
    weight_sum = sum(used_weights.values())
    if weight_sum > 0:
        used_weights = {k: v / weight_sum for k, v in used_weights.items()}
    return sum(valid_items[k] * used_weights[k] for k in valid_items)


def calculate_player_score(df: pd.DataFrame, role: str) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    full_metrics = compute_metrics(df)
    df_recent = df.dropna(subset=["fv"]).tail(5)
    recent_metrics = compute_metrics(df_recent) if not df_recent.empty else full_metrics

    base_weights = get_role_weights(role)
    weights = adjust_weights_for_unreliable(base_weights, full_metrics["fv_unreliable"])

    components = components_from_metrics(full_metrics)
    seasonal_score = weighted_score(components, weights)

    recent_components = components_from_metrics(recent_metrics)
    recent_score = weighted_score(recent_components, weights)

    smoothed_score = 0.7 * seasonal_score + 0.3 * recent_score
    reliability = min(1.0, full_metrics["matches_n"] / 10.0)
    final_score = reliability * smoothed_score + (1.0 - reliability) * 50.0

    debug = {
        "fv_unreliable": full_metrics["fv_unreliable"],
        "fv_equals_voto_ratio": full_metrics["fv_equals_voto_ratio"],
        "matches_n": full_metrics["matches_n"],
        "reliability": reliability,
        "seasonal_score": seasonal_score,
        "recent_score": recent_score,
        "smoothed_score": smoothed_score,
        "weights": weights,
    }

    return final_score, components, debug


def season_trade_value(final_score: float, availability: float, std_fv: float) -> float:
    """Season value combines final score, availability bonus, and risk penalty."""
    value = final_score
    value += 10 * (availability - 0.80)
    value -= 5 * max(0.0, (std_fv - 1.2))
    return clamp(value, 0.0, 100.0)


def next3_trade_value(pred_next3: float, availability: float, std_fv: float) -> float:
    """Next-3 value combines forecast, availability, and higher risk penalty."""
    base = scale_positive(pred_next3, 5.0, 7.5)
    value = base
    value += 8 * (availability - 0.80)
    value -= 8 * max(0.0, (std_fv - 1.1))
    return clamp(value, 0.0, 100.0)


def build_player_payload(name: str, df: pd.DataFrame, role: str) -> Dict[str, Any]:
    metrics = compute_metrics(df)
    prediction = compute_prediction(df)
    final_score, components, debug = calculate_player_score(df, role)

    season_value = season_trade_value(final_score, metrics["availability_ratio"], prediction["std_fv"])
    next3_value = next3_trade_value(
        prediction["predicted_next3"], metrics["availability_ratio"], prediction["std_fv"]
    )

    series = (
        df[["giornata", "fv", "voto"]]
        .dropna(subset=["giornata"])
        .sort_values("giornata")
        .to_dict(orient="records")
    )

    return {
        "name": name,
        "role": role,
        "avg_voto": metrics["avg_voto"],
        "avg_fv": metrics["avg_fv"],
        "bonus_impact": metrics["bonus_impact"],
        "std_fv": prediction["std_fv"],
        "recent_trend": metrics["recent_trend_raw"],
        "matches_n": metrics["matches_n"],
        "availability": metrics["availability_ratio"],
        "fv_unreliable": metrics["fv_unreliable"],
        "fv_equals_voto_ratio": metrics["fv_equals_voto_ratio"],
        "final_score": final_score,
        "predicted_next_fv": prediction["predicted_fv"],
        "predicted_next3_fv": prediction["predicted_next3"],
        "predicted_range60": [prediction["range_low"], prediction["range_high"]],
        "predicted_next3_range60": [
            prediction["range_low_next3"],
            prediction["range_high_next3"],
        ],
        "p_big_game": prediction["p_big_game"],
        "p_bad_game": prediction["p_bad_game"],
        "season_value": season_value,
        "next3_value": next3_value,
        "components": components,
        "debug": debug,
        "series": series,
    }
