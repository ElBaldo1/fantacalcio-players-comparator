import os
import re
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup


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


class FantacalcioScraper:
    """Scrape match stats from a Fantacalcio player page."""

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    def __init__(self, url: str) -> None:
        self.url = url.strip()

    @staticmethod
    def _to_float(value: Any) -> float:
        """Convert Italian numeric formats to float; non-numeric -> NaN."""
        if value is None:
            return np.nan
        text = str(value).strip()
        if text in {"", "-", "S.V.", "S.V", "s.v.", "s.v"}:
            return np.nan
        text = text.replace(",", ".")
        text = re.sub(r"[^0-9\.\-]", "", text)
        try:
            return float(text)
        except ValueError:
            return np.nan

    @staticmethod
    def _normalize_col(name: Any) -> str:
        text = str(name).lower().strip()
        text = text.replace(".", "")
        text = re.sub(r"\s+", " ", text)
        return text

    def _map_columns(self, columns: list) -> Dict[str, str]:
        """Map detected column names to standardized keys."""
        mapping = {}
        giornata_assigned = 0
        for col in columns:
            norm = self._normalize_col(col)
            if "giornata" in norm:
                if giornata_assigned == 0:
                    mapping[col] = "giornata"
                else:
                    mapping[col] = "match_info"
                giornata_assigned += 1
            elif norm in {"voto", "voti"}:
                mapping[col] = "voto"
            elif norm in {"fv", "f v", "f v ", "fv ", "fantavoto", "fanta voto", "fanta-voto"}:
                mapping[col] = "fv"
            elif "bonus" in norm and "malus" in norm:
                mapping[col] = "bonus_malus"
        return mapping

    @staticmethod
    def _coalesce_duplicate_column(df: pd.DataFrame, colname: str) -> pd.DataFrame:
        """
        If duplicate columns exist (same name), keep the column with the most
        numeric values and drop the others.
        """
        cols = df.loc[:, df.columns == colname]
        if cols.shape[1] <= 1:
            return df
        best_series = None
        best_count = -1
        for i in range(cols.shape[1]):
            series = pd.to_numeric(cols.iloc[:, i], errors="coerce")
            count = int(series.notna().sum())
            if count > best_count:
                best_count = count
                best_series = series
        df = df.drop(columns=[colname])
        df[colname] = best_series
        return df

    def _find_stats_table(self, soup: BeautifulSoup) -> Optional[pd.DataFrame]:
        """
        Find the match stats table by searching for columns like
        'Giornata' and 'Voto' or variants.
        """
        for table in soup.find_all("table"):
            try:
                df = pd.read_html(StringIO(str(table)))[0]
            except Exception:
                continue
            mapping = self._map_columns(list(df.columns))
            if "giornata" in mapping.values() and "voto" in mapping.values():
                df = df.rename(columns=mapping)
                return df
        return None

    @staticmethod
    def _extract_player_name(soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1 and h1.text.strip():
            return h1.text.strip()
        title = soup.title.text.strip() if soup.title else "Player"
        return title.split("-")[0].strip() if "-" in title else title

    def _extract_stats_from_html(self, soup: BeautifulSoup) -> Optional[pd.DataFrame]:
        """
        Fallback parser for pages where vote values are stored in data-value
        attributes (e.g., <span class="grade" data-value="6">).
        """
        table = soup.find("table", class_="player-summary-table")
        if table is None:
            return None

        rows = []
        tbody = table.find("tbody")
        if not tbody:
            return None

        for tr in tbody.find_all("tr"):
            matchweek = tr.find("span", class_="matchweek")
            if not matchweek:
                continue
            giornata = matchweek.get_text(strip=True)
            voto_span = tr.find("span", class_="grade")
            fv_span = tr.find("span", class_="fanta-grade")
            events_span = tr.find("span", class_="events")

            voto_val = voto_span.get("data-value") if voto_span else None
            fv_val = fv_span.get("data-value") if fv_span else None
            bonus_val = events_span.get_text(strip=True) if events_span else None

            rows.append(
                {
                    "giornata": giornata,
                    "voto": voto_val,
                    "fv": fv_val,
                    "bonus_malus": bonus_val,
                }
            )

        if not rows:
            return None
        return pd.DataFrame(rows)

    def fetch(self) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
        headers = {"User-Agent": self.USER_AGENT}
        try:
            response = requests.get(self.url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(
                f"Unable to fetch the player page. Check the URL and connectivity. URL: {self.url}"
            ) from exc

        soup = BeautifulSoup(response.text, "html.parser")

        player_name = self._extract_player_name(soup)
        stats_table = self._find_stats_table(soup)
        if stats_table is None:
            stats_table = self._extract_stats_from_html(soup)

        if stats_table is None:
            raise ValueError(
                "Stats table not found. The page may have changed or the URL is wrong. "
                f"URL: {self.url}"
            )

        df = stats_table.copy()
        for key in ("giornata", "voto", "fv"):
            if key in df.columns:
                df = self._coalesce_duplicate_column(df, key)

        if "giornata" not in df.columns:
            raise ValueError(
                "Missing 'Giornata' column in stats table. "
                f"Found columns: {list(df.columns)}. URL: {self.url}"
            )
        if "voto" not in df.columns:
            raise ValueError(
                "Missing 'Voto' column in stats table. "
                f"Found columns: {list(df.columns)}. URL: {self.url}"
            )
        if "fv" not in df.columns:
            raise ValueError(
                "Missing 'FV' column in stats table. "
                f"Found columns: {list(df.columns)}. URL: {self.url}"
            )

        df["giornata"] = pd.to_numeric(df["giornata"], errors="coerce")
        df["voto"] = df["voto"].apply(self._to_float)
        df["fv"] = df["fv"].apply(self._to_float)

        if df["voto"].isna().all() and df["fv"].isna().all():
            fallback_df = self._extract_stats_from_html(soup)
            if fallback_df is not None:
                df = fallback_df
                df["giornata"] = pd.to_numeric(df["giornata"], errors="coerce")
                df["voto"] = df["voto"].apply(self._to_float)
                df["fv"] = df["fv"].apply(self._to_float)

        if df["voto"].dropna().empty and df["fv"].dropna().empty:
            raise ValueError(
                "No usable match rows after cleaning. The table exists but FV/Voto are empty. "
                f"URL: {self.url}"
            )

        return player_name, df, {"url": self.url}


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


def weighted_moving_average(values: list) -> Tuple[float, list]:
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


def confidence_label(matches_n: int, std_fv: float) -> str:
    if matches_n >= 15 and not np.isnan(std_fv) and std_fv <= 1.2:
        return "High"
    if matches_n >= 10:
        return "Medium"
    return "Low"


def lineup_index(predicted_fv: float, availability: float, std_fv: float) -> int:
    if np.isnan(predicted_fv):
        base_index = 3
    else:
        base = clamp(predicted_fv, 5.5, 7.5)
        scaled = 1 + (base - 5.5) / 2.0 * 4
        base_index = int(round(scaled))

    index_value = base_index
    if not np.isnan(availability) and availability > 0.85:
        index_value += 1
    if not np.isnan(std_fv) and std_fv > 1.6:
        index_value -= 1
    return int(clamp(index_value, 1, 5))


def calculate_player_score(
    df: pd.DataFrame, role: str
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
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
        "avg_voto": full_metrics["avg_voto"],
        "avg_fv": full_metrics["avg_fv"],
        "bonus_impact": full_metrics["bonus_impact"],
        "consistency_raw": full_metrics["consistency_raw"],
        "recent_trend_raw": full_metrics["recent_trend_raw"],
        "availability_ratio": full_metrics["availability_ratio"],
        "matches_n": full_metrics["matches_n"],
        "max_matchday": full_metrics["max_matchday"],
        "fv_equals_voto_ratio": full_metrics["fv_equals_voto_ratio"],
        "fv_unreliable": full_metrics["fv_unreliable"],
        "reliability": reliability,
        "seasonal_score": seasonal_score,
        "recent_score": recent_score,
        "smoothed_score": smoothed_score,
        "weights": weights,
        "base_weights": base_weights,
        "role": role,
    }

    return final_score, components, debug


def season_trade_value(final_score: float, availability: float, std_fv: float) -> float:
    value = final_score
    value += 10 * (availability - 0.80)
    value -= 5 * max(0.0, (std_fv - 1.2))
    return clamp(value, 0.0, 100.0)


def next3_trade_value(pred_next3: float, availability: float, std_fv: float) -> float:
    base = scale_positive(pred_next3, 5.0, 7.5)
    value = base
    value += 8 * (availability - 0.80)
    value -= 8 * max(0.0, (std_fv - 1.1))
    return clamp(value, 0.0, 100.0)


def format_metric(value: float, digits: int = 2, signed: bool = False) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(value)


def build_player_profile(name: str, df: pd.DataFrame, role: str) -> Dict[str, Any]:
    metrics = compute_metrics(df)
    prediction = compute_prediction(df)
    final_score, components, debug = calculate_player_score(df, role)

    conf_label = confidence_label(metrics["matches_n"], prediction["std_fv"])
    index_value = lineup_index(
        prediction["predicted_fv"], metrics["availability_ratio"], prediction["std_fv"]
    )

    season_value = season_trade_value(final_score, metrics["availability_ratio"], prediction["std_fv"])
    next3_value = next3_trade_value(
        prediction["predicted_next3"], metrics["availability_ratio"], prediction["std_fv"]
    )

    return {
        "name": name,
        "df": df,
        "role": role,
        "avg_voto": metrics["avg_voto"],
        "avg_fv": metrics["avg_fv"],
        "bonus_impact": metrics["bonus_impact"],
        "consistency": metrics["consistency_raw"],
        "recent_trend": metrics["recent_trend_raw"],
        "matches_n": metrics["matches_n"],
        "availability": metrics["availability_ratio"],
        "final_score": final_score,
        "components": components,
        "debug": debug,
        "prediction": prediction,
        "confidence_label": conf_label,
        "lineup_index": index_value,
        "season_value": season_value,
        "next3_value": next3_value,
    }


@st.cache_data(show_spinner=False)
def fetch_player(url: str) -> Tuple[str, pd.DataFrame]:
    name, df, _ = FantacalcioScraper(url).fetch()
    return name, df


def role_counts(entries: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for entry in entries:
        role = entry.get("role")
        if role in counts:
            counts[role] += 1
    return counts


def validate_trade(left: List[Dict[str, str]], right: List[Dict[str, str]]) -> Tuple[bool, str]:
    if len(left) != len(right):
        return False, f"Player count mismatch: Left={len(left)}, Right={len(right)}"

    left_counts = role_counts(left)
    right_counts = role_counts(right)
    if left_counts != right_counts:
        return (
            False,
            "Role counts mismatch. "
            f"Left: {left_counts} | Right: {right_counts}",
        )

    return True, ""


def build_trade_tables(players: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in players:
        rows.append(
            {
                "Player": p["name"],
                "Role": p["role"],
                "Final Score": round(p["final_score"], 1),
                "Season Value": round(p["season_value"], 1),
                "Pred Next FV": round(p["prediction"]["predicted_fv"], 2),
                "Pred Next3 FV": round(p["prediction"]["predicted_next3"], 2),
                "Next3 Value": round(p["next3_value"], 1),
                "Confidence": p["confidence_label"],
                "Availability": round(p["availability"], 2),
                "Avg FV": round(p["avg_fv"], 2),
                "Std FV": round(p["prediction"]["std_fv"], 2),
            }
        )
    return pd.DataFrame(rows)


def driver_contributions(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    value_key: str,
) -> List[Tuple[str, float]]:
    left_role_avgs = {}
    right_role_avgs = {}
    for role in ["GK", "DEF", "MID", "FWD"]:
        left_vals = [p[value_key] for p in left if p["role"] == role]
        right_vals = [p[value_key] for p in right if p["role"] == role]
        left_role_avgs[role] = float(np.mean(left_vals)) if left_vals else np.nan
        right_role_avgs[role] = float(np.mean(right_vals)) if right_vals else np.nan

    contributions = []
    for p in right:
        baseline = left_role_avgs.get(p["role"], np.nan)
        if np.isnan(baseline):
            continue
        contributions.append((f"{p['name']} (Right)", p[value_key] - baseline))
    for p in left:
        baseline = right_role_avgs.get(p["role"], np.nan)
        if np.isnan(baseline):
            continue
        contributions.append((f"{p['name']} (Left)", p[value_key] - baseline))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:3]


def verdict_box(title: str, delta: float, drivers: List[Tuple[str, float]], threshold: float) -> None:
    if delta > threshold:
        msg = "Right side wins"
        msg_type = st.success
    elif delta < -threshold:
        msg = "Left side wins"
        msg_type = st.success
    else:
        msg = "Roughly balanced"
        msg_type = st.info

    drivers_text = ", ".join([f"{name} ({val:+.1f})" for name, val in drivers])
    details = f"{msg}. Delta: {delta:+.1f}. Key drivers: {drivers_text or 'N/A'}"
    msg_type(f"{title}: {details}")


def render_player_charts(players: List[Dict[str, Any]]) -> None:
    options = [p["name"] for p in players]
    selected = st.selectbox("Select player for FV trend", options)
    player = next(p for p in players if p["name"] == selected)
    chart_df = player["df"][["giornata", "fv"]].dropna()
    fig = px.line(chart_df, x="giornata", y="fv", markers=True, title=f"FV Trend - {player['name']}")
    st.plotly_chart(fig, use_container_width=True)


def ensure_session_state() -> None:
    if "left_players" not in st.session_state:
        st.session_state.left_players = [{"url": "", "role": "MID"}]
    if "right_players" not in st.session_state:
        st.session_state.right_players = [{"url": "", "role": "MID"}]


def render_side_inputs(side_key: str, label: str) -> None:
    st.sidebar.markdown(f"### {label}")
    entries = st.session_state[side_key]
    for idx, entry in enumerate(entries):
        st.sidebar.markdown(f"Player {idx + 1}")
        entry["url"] = st.sidebar.text_input(
            f"{label} URL {idx + 1}",
            value=entry.get("url", ""),
            key=f"{side_key}_url_{idx}",
        )
        entry["role"] = st.sidebar.selectbox(
            f"{label} Role {idx + 1}",
            ["GK", "DEF", "MID", "FWD"],
            index=["GK", "DEF", "MID", "FWD"].index(entry.get("role", "MID")),
            key=f"{side_key}_role_{idx}",
        )
        if st.sidebar.button(f"Remove {label} {idx + 1}", key=f"{side_key}_remove_{idx}"):
            st.session_state[side_key].pop(idx)
            st.rerun()

    if st.sidebar.button(f"Add player to {label}", key=f"{side_key}_add"):
        st.session_state[side_key].append({"url": "", "role": "MID"})
        st.rerun()


def run_self_test() -> None:
    assert validate_trade(
        [{"role": "GK"}, {"role": "DEF"}],
        [{"role": "GK"}, {"role": "DEF"}],
    )[0]
    ok, msg = validate_trade(
        [{"role": "GK"}],
        [{"role": "DEF"}],
    )
    assert not ok and "Role counts" in msg

    df = pd.DataFrame(
        {
            "giornata": [1, 2, 3, 4, 5],
            "voto": [6.0, 6.5, 5.5, 7.0, 6.0],
            "fv": [6.5, 7.0, 5.5, 7.5, 6.0],
        }
    )
    prediction = compute_prediction(df)
    assert abs(prediction["predicted_next3"] - clamp(prediction["predicted_fv"] + 0.5 * prediction["debug"]["recent_trend"], 4.0, 9.0)) < 1e-6

    left = [{"season_value": 50.0, "next3_value": 60.0, "role": "MID"}]
    right = [{"season_value": 55.0, "next3_value": 50.0, "role": "MID"}]
    delta_season = sum(p["season_value"] for p in right) - sum(p["season_value"] for p in left)
    delta_next3 = sum(p["next3_value"] for p in right) - sum(p["next3_value"] for p in left)
    assert delta_season > 0
    assert delta_next3 < 0

    print("SELF_TEST=1 passed")


def main() -> None:
    st.set_page_config(page_title="Fanta-Analyst Trade Analyzer", layout="wide")

    ensure_session_state()

    st.sidebar.title("Trade Analyzer")
    st.sidebar.write("Build two sides of a trade and compare value.")
    show_debug = st.sidebar.checkbox("Show debug")
    show_charts = st.sidebar.checkbox("Show per-player charts")

    render_side_inputs("left_players", "Left")
    render_side_inputs("right_players", "Right")

    analyze = st.sidebar.button("Analyze Trade")

    st.title("Fantacalcio Trade Analyzer")

    if analyze:
        left_entries = st.session_state.left_players
        right_entries = st.session_state.right_players

        valid, message = validate_trade(left_entries, right_entries)
        if not valid:
            st.error(message)
            if show_debug:
                st.write({"left_counts": role_counts(left_entries), "right_counts": role_counts(right_entries)})
            return

        for entry in left_entries + right_entries:
            if not entry.get("url"):
                st.error("All player URLs must be filled before analysis.")
                return

        left_players = []
        right_players = []

        try:
            for entry in left_entries:
                name, df = fetch_player(entry["url"])
                left_players.append(build_player_profile(name, df, entry["role"]))
            for entry in right_entries:
                name, df = fetch_player(entry["url"])
                right_players.append(build_player_profile(name, df, entry["role"]))
        except Exception as exc:
            st.error(f"Failed to load player: {exc}")
            return

        left_table = build_trade_tables(left_players)
        right_table = build_trade_tables(right_players)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Left Side")
            st.dataframe(left_table, use_container_width=True)
        with col2:
            st.subheader("Right Side")
            st.dataframe(right_table, use_container_width=True)

        left_total_season = sum(p["season_value"] for p in left_players)
        right_total_season = sum(p["season_value"] for p in right_players)
        delta_season = right_total_season - left_total_season

        left_total_next3 = sum(p["next3_value"] for p in left_players)
        right_total_next3 = sum(p["next3_value"] for p in right_players)
        delta_next3 = right_total_next3 - left_total_next3

        st.markdown("### Trade Totals")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Season Total (Left)", f"{left_total_season:.1f}")
            st.metric("Season Total (Right)", f"{right_total_season:.1f}")
        with s2:
            st.metric("Season Delta (Right - Left)", f"{delta_season:+.1f}")
        with s3:
            st.metric("Next3 Delta (Right - Left)", f"{delta_next3:+.1f}")

        totals_df = pd.DataFrame(
            {
                "Side": ["Left", "Right"],
                "Season Value": [left_total_season, right_total_season],
                "Next3 Value": [left_total_next3, right_total_next3],
            }
        )
        fig_bar = go.Figure(
            data=[
                go.Bar(name="Season", x=totals_df["Side"], y=totals_df["Season Value"]),
                go.Bar(name="Next3", x=totals_df["Side"], y=totals_df["Next3 Value"]),
            ]
        )
        fig_bar.update_layout(barmode="group", title="Trade Value Totals")
        st.plotly_chart(fig_bar, use_container_width=True)

        if show_charts:
            st.markdown("### Player FV Trends")
            render_player_charts(left_players + right_players)

        st.markdown("### Verdict")
        threshold = 5.0
        drivers_next3 = driver_contributions(left_players, right_players, "next3_value")
        drivers_season = driver_contributions(left_players, right_players, "season_value")
        verdict_box("Short-Term Verdict (Next 3)", delta_next3, drivers_next3, threshold)
        verdict_box("Long-Term Verdict (Season)", delta_season, drivers_season, threshold)

        if show_debug:
            st.markdown("### Debug")
            st.write({"left_counts": role_counts(left_entries), "right_counts": role_counts(right_entries)})
            for side_label, players in [("Left", left_players), ("Right", right_players)]:
                st.write(f"{side_label} details")
                for p in players:
                    st.write(
                        {
                            "name": p["name"],
                            "role": p["role"],
                            "fv_unreliable": p["debug"]["fv_unreliable"],
                            "fv_equals_voto_ratio": p["debug"]["fv_equals_voto_ratio"],
                            "prediction": p["prediction"]["debug"],
                            "season_value": p["season_value"],
                            "next3_value": p["next3_value"],
                            "seasonal_score": p["debug"]["seasonal_score"],
                            "recent_score": p["debug"]["recent_score"],
                            "smoothed_score": p["debug"]["smoothed_score"],
                            "reliability": p["debug"]["reliability"],
                        }
                    )


if __name__ == "__main__":
    if os.getenv("SELF_TEST") == "1":
        run_self_test()
    else:
        main()
