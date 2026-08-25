"""Load and clean Northstar Desk case exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

VALID_SENTIMENTS = {"Negative", "Neutral", "Positive"}
TEAM_ALIASES = {"operations": "support"}


def load_raw_cases(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def clean_cases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
    out["snapshot_at"] = pd.to_datetime(out["snapshot_at"], utc=True, errors="coerce")

    out["escalated"] = (
        out["escalated"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    )
    out["sentiment"] = out["sentiment"].where(out["sentiment"].isin(VALID_SENTIMENTS))
    out["assigned_team"] = (
        out["assigned_team"].astype(str).str.strip().str.lower().replace(TEAM_ALIASES)
    )

    for col in ["first_response_time_hours", "resolution_time_hours", "csat_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["case_summary"] = out["case_summary"].fillna("").astype(str).str.strip()
    out["tags"] = out["tags"].fillna("").astype(str)

    out = out.sort_values(["case_id", "snapshot_at"], na_position="last")
    out = out.drop_duplicates(subset="case_id", keep="last").reset_index(drop=True)
    return out


def load_cases(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return clean_cases(load_raw_cases(data_dir))


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sla_breach"] = out["resolution_time_hours"] > out["sla_target_hours"]
    out["slow_resolution"] = out["resolution_time_hours"] > out["resolution_time_hours"].median()
    out["low_csat"] = out["csat_score"] <= 2
    out["search_text"] = (
        out["case_summary"]
        + " "
        + out["category"].fillna("")
        + " "
        + out["subcategory"].fillna("")
        + " "
        + out["tags"]
    ).str.strip()
    return out
