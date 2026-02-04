import os
import re
from io import StringIO
from typing import Any, Dict, Optional, Tuple

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
        "Consistency": 0.25,
        "Availability": 0.20,
        "Recent Trend": 0.10,
        "Bonus Impact": 0.00,
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
            "range_low": float("nan"),
            "range_high": float("nan"),
            "p_big_game": float("nan"),
            "p_bad_game": float("nan"),
            "low_confidence": True,
            "debug": {"last_values": [], "weights": [], "std_fv": float("nan")},
        }

    last_values = fv_series.tail(5).to_list()
    prediction, weights = weighted_moving_average(last_values)

    std_fv = float(fv_series.std()) if matches_n > 0 else float("nan")
    std_fv = max(std_fv, 0.15) if not np.isnan(std_fv) else float("nan")

    range_low = prediction - 0.84 * std_fv
    range_high = prediction + 0.84 * std_fv

    p_big_game = float((fv_series >= 7.0).mean()) if matches_n > 0 else float("nan")
    p_bad_game = float((fv_series < 6.0).mean()) if matches_n > 0 else float("nan")

    return {
        "predicted_fv": prediction,
        "range_low": range_low,
        "range_high": range_high,
        "p_big_game": p_big_game,
        "p_bad_game": p_bad_game,
        "low_confidence": matches_n < 8,
        "debug": {"last_values": last_values, "weights": weights, "std_fv": std_fv},
    }


def calculate_player_score(
    df: pd.DataFrame, role: str
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
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
    if paired.empty:
        fv_equals_voto_ratio = float("nan")
    else:
        fv_equals_voto_ratio = float((paired["fv"] - paired["voto"]).abs().lt(0.01).mean())

    fv_unreliable = False
    if not np.isnan(fv_equals_voto_ratio) and fv_equals_voto_ratio > 0.8:
        fv_unreliable = True

    components = {
        "Avg Voto": scale_positive(avg_voto, 5.0, 7.5),
        "Bonus Impact": scale_positive(bonus_impact, -0.2, 2.0),
        "Consistency": scale_inverted(consistency_raw, 0.2, 2.0),
        "Recent Trend": scale_trend(recent_trend_raw, -1.0, 1.0),
        "Availability": scale_positive(availability_ratio, 0.0, 1.0),
    }

    base_weights = get_role_weights(role)
    weights = adjust_weights_for_unreliable(base_weights, fv_unreliable)

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
        "availability_ratio": availability_ratio,
        "matches_n": matches_n,
        "max_matchday": max_matchday,
        "fv_equals_voto_ratio": fv_equals_voto_ratio,
        "fv_unreliable": fv_unreliable,
        "reliability": reliability,
        "weighted_sum": weighted_sum,
        "weights": weights,
        "base_weights": base_weights,
        "role": role,
    }

    return final_score, components, debug


def format_metric(value: float, digits: int = 2, signed: bool = False) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(value)


def build_player_profile(name: str, df: pd.DataFrame, role: str) -> Dict[str, Any]:
    avg_voto = safe_mean(df["voto"])
    avg_fv = safe_mean(df["fv"])
    bonus_impact = safe_mean(df["fv"] - df["voto"])
    fv_std = float(df["fv"].dropna().std()) if df["fv"].dropna().size > 0 else float("nan")
    consistency = max(fv_std, 0.15) if not np.isnan(fv_std) else float("nan")
    last5 = df["fv"].dropna().tail(5)
    recent_trend = (
        float(last5.mean() - df["fv"].dropna().mean()) if last5.size > 0 else float("nan")
    )
    matches_n = int(df["fv"].dropna().shape[0])
    max_matchday = int(df["giornata"].dropna().max()) if df["giornata"].dropna().size > 0 else 0
    availability_ratio = matches_n / max(1, max_matchday) if matches_n > 0 else float("nan")

    prediction = compute_prediction(df)
    final_score, components, debug = calculate_player_score(df, role)

    return {
        "name": name,
        "df": df,
        "avg_voto": avg_voto,
        "avg_fv": avg_fv,
        "bonus_impact": bonus_impact,
        "consistency": consistency,
        "recent_trend": recent_trend,
        "matches_n": matches_n,
        "availability": availability_ratio,
        "final_score": final_score,
        "components": components,
        "debug": debug,
        "prediction": prediction,
    }


@st.cache_data(show_spinner=False)
def fetch_player(url: str) -> Tuple[str, pd.DataFrame]:
    name, df, _ = FantacalcioScraper(url).fetch()
    return name, df


def render_prediction(profile: Dict[str, Any]) -> None:
    prediction = profile["prediction"]
    predicted = prediction["predicted_fv"]
    avg_fv = profile["avg_fv"]

    st.markdown("#### Prediction")
    st.metric(
        "Predicted FV (next)",
        format_metric(predicted, digits=2),
        format_metric(predicted - avg_fv, digits=2, signed=True),
    )
    range_text = (
        f"60% range: [{format_metric(prediction['range_low'], digits=2)}, "
        f"{format_metric(prediction['range_high'], digits=2)}]"
    )
    st.write(range_text)
    st.write(
        f"P(FV >= 7.0): {format_metric(prediction['p_big_game'] * 100, digits=1)}% | "
        f"P(FV < 6.0): {format_metric(prediction['p_bad_game'] * 100, digits=1)}%"
    )
    st.caption("Context-free estimate based on recent form and historical variance.")
    if prediction["low_confidence"]:
        st.warning("Prediction is low confidence due to limited matches.")


def render_player_card(profile: Dict[str, Any], other_score: float) -> None:
    st.subheader(profile["name"])
    st.metric(
        "Final Score",
        format_metric(profile["final_score"], digits=1),
        format_metric(profile["final_score"] - other_score, digits=1, signed=True),
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Avg Voto", format_metric(profile["avg_voto"], digits=2))
        st.metric("Avg FV", format_metric(profile["avg_fv"], digits=2))
        st.metric("Bonus Impact", format_metric(profile["bonus_impact"], digits=2, signed=True))
    with m2:
        st.metric("Consistency (Std FV)", format_metric(profile["consistency"], digits=2))
        st.metric("Recent Trend", format_metric(profile["recent_trend"], digits=2, signed=True))
    with m3:
        st.metric("Matches", f"{profile['matches_n']}")
        st.metric("Availability", format_metric(profile["availability"], digits=2))

    render_prediction(profile)


def build_radar_chart(p1: Dict[str, Any], p2: Dict[str, Any]) -> go.Figure:
    categories = [
        "Avg Voto",
        "Bonus Impact",
        "Consistency",
        "Recent Trend",
        "Availability",
    ]

    def values_for(profile: Dict[str, Any]) -> list:
        values = []
        for cat in categories:
            val = profile["components"].get(cat, float("nan"))
            if np.isnan(val):
                val = 50.0
            values.append(val)
        return values

    values_1 = values_for(p1)
    values_2 = values_for(p2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_1 + values_1[:1],
            theta=categories + categories[:1],
            fill="toself",
            name=p1["name"],
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=values_2 + values_2[:1],
            theta=categories + categories[:1],
            fill="toself",
            name=p2["name"],
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=30, r=30, t=30, b=30),
    )
    return fig


def verdict_block(p1: Dict[str, Any], p2: Dict[str, Any]) -> None:
    diff = p1["final_score"] - p2["final_score"]
    if abs(diff) < 1.0:
        title = "Verdict: Tie"
        msg_type = st.info
    elif abs(diff) < 5.0:
        title = f"Verdict: {p1['name']} wins" if diff > 0 else f"Verdict: {p2['name']} wins"
        msg_type = st.warning
    else:
        title = f"Verdict: {p1['name']} wins" if diff > 0 else f"Verdict: {p2['name']} wins"
        msg_type = st.success

    comp_diff = {}
    for key in p1["components"].keys():
        if key == "Bonus Impact" and (p1["debug"]["fv_unreliable"] or p2["debug"]["fv_unreliable"]):
            continue
        v1 = p1["components"].get(key, float("nan"))
        v2 = p2["components"].get(key, float("nan"))
        if np.isnan(v1) or np.isnan(v2):
            continue
        comp_diff[key] = v1 - v2

    strongest = sorted(comp_diff.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    parts = [f"{name} ({val:+.1f})" for name, val in strongest]
    explanation = (
        "Key differences: " + ", ".join(parts) if parts else "Key differences unavailable."
    )

    msg_type(f"{title}. Score diff: {diff:+.1f}. {explanation}")


def run_self_test() -> None:
    df = pd.DataFrame(
        {
            "giornata": [1, 2, 3, 4, 5],
            "voto": [6.0, 6.5, 5.5, 7.0, 6.0],
            "fv": [6.5, 7.0, 5.5, 7.5, 6.0],
        }
    )

    assert scale_positive(5.0, 5.0, 7.5) == 0.0
    assert scale_positive(7.5, 5.0, 7.5) == 100.0
    assert scale_trend(0.0) == 50.0
    assert scale_inverted(0.2, 0.2, 2.0) == 100.0
    assert scale_inverted(2.0, 0.2, 2.0) == 0.0

    pred, weights = weighted_moving_average([5, 6, 7, 8, 9])
    assert weights == [1, 2, 3, 4, 5]
    assert abs(pred - (115 / 15)) < 1e-6

    probs_df = pd.DataFrame(
        {
            "giornata": [1, 2, 3, 4],
            "voto": [5.0, 6.0, 7.0, 8.0],
            "fv": [5.0, 6.0, 7.0, 8.0],
        }
    )
    prediction = compute_prediction(probs_df)
    assert abs(prediction["p_big_game"] - 0.5) < 1e-6
    assert abs(prediction["p_bad_game"] - 0.25) < 1e-6

    unreliable_df = pd.DataFrame(
        {
            "giornata": [1, 2, 3, 4],
            "voto": [6.0, 6.0, 6.0, 6.0],
            "fv": [6.0, 6.0, 6.0, 6.0],
        }
    )
    _, _, debug = calculate_player_score(unreliable_df, "MID")
    assert debug["fv_unreliable"] is True
    assert debug["weights"]["Bonus Impact"] <= 0.05 + 1e-9
    assert abs(sum(debug["weights"].values()) - 1.0) < 1e-6

    final_score, components, debug = calculate_player_score(df, "MID")
    assert 0.0 <= final_score <= 100.0
    assert all(0.0 <= v <= 100.0 for v in components.values() if not np.isnan(v))
    assert debug["matches_n"] == 5

    print("SELF_TEST=1 passed")


def main() -> None:
    st.set_page_config(page_title="Fanta-Analyst Pro", layout="wide")

    st.sidebar.title("Fanta-Analyst Pro")
    st.sidebar.write(
        "Paste two player URLs from fantacalcio.it to compare ratings, trends, and availability."
    )
    role = st.sidebar.selectbox("Role", ["GK", "DEF", "MID", "FWD"], index=2)
    url1 = st.sidebar.text_input("Player 1 URL")
    url2 = st.sidebar.text_input("Player 2 URL")
    show_debug = st.sidebar.checkbox("Show debug details")
    analyze = st.sidebar.button("Analyze")

    st.title("Fantacalcio Player Comparator")

    if analyze:
        if not url1 or not url2:
            st.error("Please provide both player URLs before analyzing.")
            return

        try:
            name1, df1 = fetch_player(url1)
            name2, df2 = fetch_player(url2)
        except Exception as exc:
            st.error(str(exc))
            return

        if df1.empty or df2.empty:
            st.error("No usable match data found after cleaning. Please verify the URLs.")
            return

        p1 = build_player_profile(name1, df1, role)
        p2 = build_player_profile(name2, df2, role)

        col1, col2 = st.columns(2)
        with col1:
            render_player_card(p1, p2["final_score"])
            if p1["debug"]["fv_unreliable"]:
                st.warning(
                    "Fantavoto appears unreliable for this page/table (FV ~= Voto for most matches)."
                )
        with col2:
            render_player_card(p2, p1["final_score"])
            if p2["debug"]["fv_unreliable"]:
                st.warning(
                    "Fantavoto appears unreliable for this page/table (FV ~= Voto for most matches)."
                )

        st.markdown("### FV Trend by Matchday")
        chart_df = pd.concat(
            [
                p1["df"][["giornata", "fv"]].dropna().assign(player=p1["name"]),
                p2["df"][["giornata", "fv"]].dropna().assign(player=p2["name"]),
            ],
            ignore_index=True,
        )
        fig_line = px.line(
            chart_df,
            x="giornata",
            y="fv",
            color="player",
            markers=True,
        )
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("### Radar Comparison (0-100)")
        fig_radar = build_radar_chart(p1, p2)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### Verdict")
        verdict_block(p1, p2)

        if show_debug:
            with st.expander("Debug details"):
                st.write(
                    {
                        "player1": {**p1["debug"], "prediction": p1["prediction"]["debug"]},
                        "player2": {**p2["debug"], "prediction": p2["prediction"]["debug"]},
                    }
                )


if __name__ == "__main__":
    if os.getenv("SELF_TEST") == "1":
        run_self_test()
    else:
        main()
