import re
from io import StringIO
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

    def fetch(self) -> Tuple[str, pd.DataFrame]:
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

        return player_name, df
