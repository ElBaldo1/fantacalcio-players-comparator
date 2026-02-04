import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
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
    def _normalize_col(name: str) -> str:
        return re.sub(r"\s+", " ", str(name).strip().lower())

    def _find_stats_table(self, soup: BeautifulSoup) -> Optional[pd.DataFrame]:
        """
        Find the match stats table by searching for columns like
        'Giornata' and 'Voto'.
        """
        for table in soup.find_all("table"):
            try:
                df = pd.read_html(str(table))[0]
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

    @staticmethod
    def _extract_player_name(soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1 and h1.text.strip():
            return h1.text.strip()
        title = soup.title.text.strip() if soup.title else "Player"
        return title.split("-")[0].strip() if "-" in title else title

    @staticmethod
    def _extract_optional_stats(soup: BeautifulSoup) -> Dict[str, Optional[int]]:
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
            raise ValueError("Unable to fetch the player page. Check the URL.") from exc

        soup = BeautifulSoup(response.text, "html.parser")

        player_name = self._extract_player_name(soup)
        stats_table = self._find_stats_table(soup)
        if stats_table is None:
            raise ValueError(
                "Stats table not found. The page may have changed or the URL is wrong."
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
            raise ValueError("Missing 'Giornata' column in stats table.")
        if "voto" not in df.columns:
            raise ValueError("Missing 'Voto' column in stats table.")
        if "fv" not in df.columns:
            raise ValueError("Missing 'FV' column in stats table.")

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

        df = df.dropna(subset=["voto", "fv"], how="all").sort_values("giornata")

        optional_stats = self._extract_optional_stats(soup)
        return player_name, df, optional_stats
