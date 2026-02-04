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


class FantacalcioScraper:
    """Scrape match stats from a Fantacalcio player page."""

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    def __init__(self, url: str) -> None:
        self.url = url.strip()

    def _to_float(self, value: Any) -> float:
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

    def _normalize_col(self, name: str) -> str:
        return re.sub(r"\s+", " ", str(name).strip().lower())

    def _find_stats_table(self, soup: BeautifulSoup) -> Optional[pd.DataFrame]:
        """
        Find the match stats table by searching for columns like
        'Giornata' and 'Voto'.
        """
        for table in soup.find_all("table"):
            try:
                df = pd.read_html(StringIO(str(table)))[0]
            except Exception:
                continue
            cols_norm = [self._normalize_col(c) for c in df.columns]
            if "giornata" in cols_norm and "voto" in cols_norm:
                df.columns = cols_norm
                return df
        return None

    def _extract_stats_from_html(self, soup: BeautifulSoup) -> Optional[pd.DataFrame]:
        """
        Fallback parser for pages where vote values are stored in data-value
        attributes (e.g., <span class=\"grade\" data-value=\"6\">).
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

    def _extract_player_name(self, soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1 and h1.text.strip():
            return h1.text.strip()
        title = soup.title.text.strip() if soup.title else "Giocatore"
        return title.split("-")[0].strip() if "-" in title else title

    def _extract_optional_stats(self, soup: BeautifulSoup) -> Dict[str, Optional[int]]:
        """
        Try to extract goals/assists from page text using regex.
        If not found, return None values.
        """
        text = soup.get_text(" ", strip=True).lower()
        stats: Dict[str, Optional[int]] = {"gol": None, "assist": None}
        for key in ["gol", "assist"]:
            match = re.search(rf"{key}\s+(\d+)", text)
            if match:
                stats[key] = int(match.group(1))
        return stats

    def fetch(self) -> Tuple[str, pd.DataFrame, Dict[str, Optional[int]]]:
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
            raise ValueError(
                "Stats table not found. The page may have changed or the URL is wrong. "
                f"URL: {self.url}"
            )

        df = stats_table.copy()
        col_map = {
            "giornata": "giornata",
            "voto": "voto",
            "fv": "fv",
            "fantavoto": "fv",
            "fanta voto": "fv",
            "bonus/malus": "bonus_malus",
            "bonus malus": "bonus_malus",
        }
        df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

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

        raw_rows = len(df)
        df = df.dropna(subset=["voto", "fv"], how="all").sort_values("giornata")
        if df.empty:
            voto_count = int(df["voto"].dropna().shape[0]) if "voto" in df.columns else 0
            fv_count = int(df["fv"].dropna().shape[0]) if "fv" in df.columns else 0
            raise ValueError(
                "No usable match rows after cleaning. "
                f"Raw rows: {raw_rows}. Voto count: {voto_count}. FV count: {fv_count}. "
                f"URL: {self.url}"
            )

        optional_stats = self._extract_optional_stats(soup)
        return player_name, df, optional_stats


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
    consistency_raw = float(fv_series.dropna().std()) if fv_series.dropna().size > 0 else float("nan")

    last5 = fv_series.dropna().tail(5)
    recent_trend_raw = float(last5.mean() - fv_series.dropna().mean()) if last5.size > 0 else float("nan")

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


def format_metric(value: float, digits: int = 2, signed: bool = False) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(value)


def build_player_profile(
    name: str, df: pd.DataFrame, optional_stats: Dict[str, Optional[int]]
) -> Dict[str, Any]:
    avg_voto = safe_mean(df["voto"])
    avg_fv = safe_mean(df["fv"])
    bonus_impact = safe_mean(df["fv"] - df["voto"])
    consistency = float(df["fv"].dropna().std()) if df["fv"].dropna().size > 0 else float("nan")
    last5 = df["fv"].dropna().tail(5)
    recent_trend = float(last5.mean() - df["fv"].dropna().mean()) if last5.size > 0 else float("nan")
    matches_n = int(df["fv"].dropna().shape[0])

    final_score, components, debug = calculate_player_score(df, optional_stats)

    return {
        "name": name,
        "df": df,
        "avg_voto": avg_voto,
        "avg_fv": avg_fv,
        "bonus_impact": bonus_impact,
        "consistency": consistency,
        "recent_trend": recent_trend,
        "matches_n": matches_n,
        "final_score": final_score,
        "components": components,
        "debug": debug,
        "optional_stats": optional_stats,
    }


@st.cache_data(show_spinner=False)
def fetch_player(url: str) -> Tuple[str, pd.DataFrame, Dict[str, Optional[int]]]:
    return FantacalcioScraper(url).fetch()


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
    with m2:
        st.metric(
            "Bonus Impact",
            format_metric(profile["bonus_impact"], digits=2, signed=True),
        )
        st.metric(
            "Consistency (Std FV)",
            format_metric(profile["consistency"], digits=2),
        )
    with m3:
        st.metric("Matches", f"{profile['matches_n']}")
        st.metric(
            "Recent Trend",
            format_metric(profile["recent_trend"], digits=2, signed=True),
        )


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

    comp_diff = {
        k: p1["components"].get(k, float("nan"))
        - p2["components"].get(k, float("nan"))
        for k in p1["components"].keys()
    }
    comp_diff = {k: v for k, v in comp_diff.items() if not np.isnan(v)}
    strongest = sorted(comp_diff.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    parts = [f"{name} ({val:+.1f})" for name, val in strongest]
    explanation = (
        "Key differences: " + ", ".join(parts) if parts else "Key differences unavailable."
    )

    msg_type(f"{title}. Score diff: {diff:+.1f}. {explanation}")


def build_radar_chart(p1: Dict[str, Any], p2: Dict[str, Any]) -> Tuple[go.Figure, bool]:
    categories = [
        "Avg Voto",
        "Bonus Impact",
        "Consistency",
        "Recent Trend",
        "Goal Involvement",
    ]

    def values_for(profile: Dict[str, Any]) -> list:
        vals = []
        for cat in categories:
            v = profile["components"].get(cat, float("nan"))
            if np.isnan(v):
                v = 50.0
            vals.append(v)
        return vals

    values_1 = values_for(p1)
    values_2 = values_for(p2)
    has_missing = np.isnan(p1["components"].get("Goal Involvement", float("nan"))) or np.isnan(
        p2["components"].get("Goal Involvement", float("nan"))
    )

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
    return fig, has_missing


def run_self_test() -> None:
    df = pd.DataFrame(
        {
            "giornata": [1, 2, 3, 4, 5],
            "voto": [6.0, 6.5, 5.5, 7.0, 6.0],
            "fv": [6.5, 7.0, 5.5, 7.5, 6.0],
            "bonus_malus": ["+1", "+3", "-0.5", "", "+1"],
        }
    )
    optional_stats = {"gol": 2, "assist": 1}

    assert scale_positive(5.0, 5.0, 7.5) == 0.0
    assert scale_positive(7.5, 5.0, 7.5) == 100.0
    assert scale_trend(0.0) == 50.0
    assert scale_inverted(0.2, 0.2, 2.0) == 100.0
    assert scale_inverted(2.0, 0.2, 2.0) == 0.0

    final_score, components, debug = calculate_player_score(df, optional_stats)
    assert 0.0 <= final_score <= 100.0
    assert all(0.0 <= v <= 100.0 for v in components.values() if not np.isnan(v))
    assert debug["matches_n"] == 5

    print("SELF_TEST=1 passed")


def main() -> None:
    st.set_page_config(page_title="Fanta-Analyst Pro", layout="wide")

    st.sidebar.title("Fanta-Analyst Pro")
    st.sidebar.write(
        "Paste two player URLs from fantacalcio.it to compare ratings, "
        "form trends, and a composite score."
    )
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
            name1, df1, opt1 = fetch_player(url1)
            name2, df2, opt2 = fetch_player(url2)
        except Exception as exc:
            st.error(str(exc))
            return

        if df1.empty or df2.empty:
            def describe_df(label: str, df: pd.DataFrame, url: str) -> str:
                fv_count = int(df["fv"].dropna().shape[0]) if "fv" in df.columns else 0
                voto_count = int(df["voto"].dropna().shape[0]) if "voto" in df.columns else 0
                matchdays = int(df["giornata"].dropna().nunique()) if "giornata" in df.columns else 0
                return (
                    f"{label} | rows: {len(df)}, fv_count: {fv_count}, "
                    f"voto_count: {voto_count}, matchdays: {matchdays}, url: {url}"
                )

            details = "\n".join(
                [
                    describe_df("Player 1", df1, url1),
                    describe_df("Player 2", df2, url2),
                ]
            )
            st.error(
                "No usable match data found after cleaning. Please verify the URLs "
                "and that the season page has match stats.\n\nDetails:\n" + details
            )
            return

        p1 = build_player_profile(name1, df1, opt1)
        p2 = build_player_profile(name2, df2, opt2)

        col1, col2 = st.columns(2)
        with col1:
            render_player_card(p1, p2["final_score"])
        with col2:
            render_player_card(p2, p1["final_score"])

        st.markdown("### FV Trend by Matchday")
        chart_df = pd.concat(
            [
                p1["df"][["giornata", "fv"]].assign(player=p1["name"]),
                p2["df"][["giornata", "fv"]].assign(player=p2["name"]),
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
        fig_radar, has_missing = build_radar_chart(p1, p2)
        st.plotly_chart(fig_radar, use_container_width=True)
        if has_missing:
            st.caption(
                "Goal involvement data was not available for at least one player; "
                "shown as 50 (neutral) in the radar chart."
            )

        st.markdown("### Verdict")
        verdict_block(p1, p2)

        if show_debug:
            with st.expander("Debug details"):
                st.write({"player1": p1["debug"], "player2": p2["debug"]})


if __name__ == "__main__":
    if os.getenv("SELF_TEST") == "1":
        run_self_test()
    else:
        main()
