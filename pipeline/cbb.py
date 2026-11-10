"""Barttorvik game logs joined to the previous season's team ratings.

Two public tables, no API key:

- Game logs: https://barttorvik.com/getgamestats.php?year=YEAR&json=1
- Season ratings: https://barttorvik.com/YEAR_team_results.csv

The site rejects a bare script user agent, so downloads send a browser one.
Only the home team's row is kept, and only last season's ratings are joined,
so the model cannot see the final score or this season's box line.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
INTERIM = REPO / "data" / "interim"
PUBLISHED = REPO / "data" / "published"
AIRFLOW_ARTIFACTS = REPO / "artifacts" / "airflow"

GAME_SEASONS = (2024, 2025)
RATING_SEASONS = (2023, 2024)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

FEATURE_SETS: dict[str, list[str]] = {
    "tempo": ["home_adjt", "away_adjt"],
    "barthag": ["home_barthag", "away_barthag", "barthag_gap"],
    "full": [
        "home_barthag",
        "away_barthag",
        "barthag_gap",
        "home_adjoe",
        "away_adjoe",
        "home_adjde",
        "away_adjde",
        "home_adjt",
        "away_adjt",
    ],
}

BASELINE_FEATURES = FEATURE_SETS["barthag"]
TARGET = "home_win"

PUBLISHED_COLUMNS = [
    "game_date",
    "season",
    "home_team",
    "away_team",
    "home_conf",
    "away_conf",
    "home_barthag",
    "away_barthag",
    "home_adjoe",
    "away_adjoe",
    "home_adjde",
    "away_adjde",
    "home_adjt",
    "away_adjt",
    "barthag_gap",
    "home_win",
]


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=120)
            response.raise_for_status()
            if not response.content or response.content[:1] == b"<":
                raise RuntimeError(f"HTML response from {url}")
            dest.write_bytes(response.content)
            return dest
        except Exception as exc:  # noqa: BLE001 - retry any transport or parse failure
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Download failed for {url}") from last_error


def fetch_games(seasons: tuple[int, ...] = GAME_SEASONS) -> dict:
    files: list[str] = []
    for year in seasons:
        path = RAW / f"games_{year}.json"
        download(
            f"https://barttorvik.com/getgamestats.php?year={year}&json=1",
            path,
        )
        files.append(str(path))
    return {"files": files}


def fetch_ratings(seasons: tuple[int, ...] = RATING_SEASONS) -> dict:
    files: list[str] = []
    for year in seasons:
        path = RAW / f"ratings_{year}.csv"
        download(f"https://barttorvik.com/{year}_team_results.csv", path)
        files.append(str(path))
    return {"files": files}


def clean_games(fetch_meta: dict) -> dict:
    records: list[dict] = []
    skipped = 0
    for file in fetch_meta["files"]:
        payload = json.loads(Path(file).read_text())
        if not isinstance(payload, list):
            raise RuntimeError(f"{file} is not a JSON list")
        for row in payload:
            if not isinstance(row, list) or len(row) < 23:
                skipped += 1
                continue
            if row[5] != "H":
                continue
            result = str(row[6])
            if not result or result[0] not in {"W", "L"}:
                skipped += 1
                continue
            records.append(
                {
                    "game_date": str(row[0]),
                    "season": int(row[22]),
                    "home_team": str(row[2]).strip(),
                    "away_team": str(row[4]).strip(),
                    "home_win": 1 if result.startswith("W") else 0,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("No home games parsed from Barttorvik game logs")
    frame["game_date"] = pd.to_datetime(frame["game_date"], format="%m/%d/%y", errors="coerce")
    frame = frame.dropna(subset=["game_date", "home_team", "away_team"])
    frame["game_date"] = frame["game_date"].dt.strftime("%Y-%m-%d")
    frame = frame.drop_duplicates(["game_date", "home_team", "away_team"])
    frame = frame.sort_values(["game_date", "home_team"]).reset_index(drop=True)
    path = _write_csv(frame, INTERIM / "games.csv")
    return {"path": str(path), "rows": int(len(frame)), "skipped_short_rows": skipped}


def clean_ratings(fetch_meta: dict) -> dict:
    frames: list[pd.DataFrame] = []
    for file in fetch_meta["files"]:
        year = int(Path(file).stem.split("_")[1])
        raw = pd.read_csv(file)
        raw.columns = [str(column).strip() for column in raw.columns]
        missing = {"team", "conf", "adjoe", "adjde", "barthag", "adjt"} - set(raw.columns)
        if missing:
            raise RuntimeError(f"{file} is missing columns: {sorted(missing)}")
        frames.append(
            pd.DataFrame(
                {
                    "rating_season": year,
                    "team": raw["team"].astype(str).str.strip(),
                    "conf": raw["conf"].astype(str).str.strip(),
                    "adjoe": pd.to_numeric(raw["adjoe"], errors="coerce"),
                    "adjde": pd.to_numeric(raw["adjde"], errors="coerce"),
                    "barthag": pd.to_numeric(raw["barthag"], errors="coerce"),
                    "adjt": pd.to_numeric(raw["adjt"], errors="coerce"),
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna()
    frame = frame.drop_duplicates(["rating_season", "team"])
    path = _write_csv(frame, INTERIM / "ratings.csv")
    return {"path": str(path), "rows": int(len(frame)), "seasons": sorted(frame["rating_season"].unique().tolist())}


def merge_games_and_ratings(games_meta: dict, ratings_meta: dict) -> dict:
    games = pd.read_csv(games_meta["path"])
    ratings = pd.read_csv(ratings_meta["path"])
    games["rating_season"] = games["season"] - 1
    home = ratings.rename(
        columns={
            "team": "home_team",
            "conf": "home_conf",
            "adjoe": "home_adjoe",
            "adjde": "home_adjde",
            "barthag": "home_barthag",
            "adjt": "home_adjt",
        }
    )
    away = ratings.rename(
        columns={
            "team": "away_team",
            "conf": "away_conf",
            "adjoe": "away_adjoe",
            "adjde": "away_adjde",
            "barthag": "away_barthag",
            "adjt": "away_adjt",
        }
    )
    merged = games.merge(home, on=["home_team", "rating_season"], how="left")
    merged = merged.merge(away, on=["away_team", "rating_season"], how="left")
    before = len(merged)
    merged = merged.dropna(
        subset=["home_barthag", "away_barthag", "home_adjoe", "away_adjoe", "home_adjde", "away_adjde", "home_adjt", "away_adjt"]
    )
    merged["barthag_gap"] = merged["home_barthag"] - merged["away_barthag"]
    merged = merged[PUBLISHED_COLUMNS].sort_values(["game_date", "home_team"]).reset_index(drop=True)
    if len(merged) < 1000:
        raise RuntimeError(f"Merge kept only {len(merged)} rows; team names likely failed to match")
    path = _write_csv(merged, INTERIM / "merged.csv")
    return {
        "path": str(path),
        "rows": int(len(merged)),
        "dropped_unmatched": int(before - len(merged)),
        "home_win_rate": float(merged["home_win"].mean()),
    }


def load_postgres(merge_meta: dict, warehouse_url: str | None = None) -> dict:
    from sqlalchemy import create_engine, text

    url = warehouse_url or os.environ["WAREHOUSE_URL"]
    frame = pd.read_csv(merge_meta["path"])
    engine = create_engine(url)
    frame.to_sql("cbb_home_games", engine, if_exists="replace", index=False)
    with engine.begin() as connection:
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_cbb_home_games_date ON cbb_home_games (game_date)"))
        stored = connection.execute(text("SELECT COUNT(*) FROM cbb_home_games")).scalar()
    if int(stored) != len(frame):
        raise RuntimeError(f"Postgres has {stored} rows, file has {len(frame)}")
    return {**merge_meta, "table": "cbb_home_games", "stored_rows": int(stored)}


def profile_frame(merge_meta: dict) -> dict:
    frame = read_games(merge_meta["path"])
    forbidden = {"points_for", "points_against", "oe", "de", "adj_oe", "adj_de"}
    leaked = sorted(forbidden & set(frame.columns))
    if leaked:
        raise RuntimeError(f"Same-game columns leaked into the model table: {leaked}")
    summary = {
        "rows": int(len(frame)),
        "seasons": sorted(int(year) for year in frame["season"].unique()),
        "date_min": str(frame["game_date"].min()),
        "date_max": str(frame["game_date"].max()),
        "home_win_rate": round(float(frame["home_win"].mean()), 4),
        "teams": int(frame["home_team"].nunique()),
        "null_cells": int(frame.isna().sum().sum()),
        "features_available": FEATURE_SETS["full"],
        "target": TARGET,
    }
    path = _write_json(summary, AIRFLOW_ARTIFACTS / "profile.json")
    return {**merge_meta, "profile_path": str(path), "profile": summary}


def train_baseline(profile_meta: dict) -> dict:
    frame = read_games(profile_meta["path"])
    train, _val, holdout = split_frame(frame)
    development = pd.concat([train, _val], ignore_index=True)
    metrics = classification_metrics(development, holdout, BASELINE_FEATURES, C=1.0)
    majority_label = int(development[TARGET].mean() >= 0.5)
    majority_accuracy = float((holdout[TARGET] == majority_label).mean())
    report = {
        "model": "logistic_regression",
        "features": BASELINE_FEATURES,
        "C": 1.0,
        "fit_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "holdout": metrics,
        "majority_class_accuracy": round(majority_accuracy, 4),
        "majority_label": majority_label,
    }
    path = _write_json(report, AIRFLOW_ARTIFACTS / "baseline_metrics.json")
    chart = plot_win_rate(frame, AIRFLOW_ARTIFACTS / "win_rate_by_barthag_gap.png")
    return {**profile_meta, "baseline_path": str(path), "chart_path": str(chart), "baseline": report}


def publish(train_meta: dict) -> dict:
    frame = read_games(train_meta["path"])
    path = _write_csv(frame, PUBLISHED / "merged_latest.csv")
    return {
        "published_path": str(path),
        "rows": int(len(frame)),
        "baseline_path": train_meta["baseline_path"],
        "chart_path": train_meta["chart_path"],
    }


def read_games(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(PUBLISHED_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    return frame.sort_values(["game_date", "home_team"]).reset_index(drop=True)


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["game_date", "home_team"]).reset_index(drop=True)
    count = len(ordered)
    train_end = int(count * 0.70)
    val_end = int(count * 0.85)
    if train_end < 50 or val_end - train_end < 20 or count - val_end < 20:
        raise RuntimeError(f"Not enough rows to split ({count})")
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:val_end].copy(),
        ordered.iloc[val_end:].copy(),
    )


def classification_metrics(train: pd.DataFrame, test: pd.DataFrame, features: list[str], C: float) -> dict:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=500, solver="lbfgs")),
        ]
    )
    model.fit(train[features], train[TARGET])
    probabilities = model.predict_proba(test[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    labels = test[TARGET].to_numpy()
    if len(set(labels.tolist())) < 2:
        raise RuntimeError("Test fold has a single class")
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "f1": round(float(f1_score(labels, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
        "n": int(len(test)),
    }


def plot_win_rate(frame: pd.DataFrame, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [-1.01, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 1.01]
    labels = ["<-0.4", "-0.4 to -0.2", "-0.2 to -0.1", "-0.1 to 0", "0 to 0.1", "0.1 to 0.2", "0.2 to 0.4", ">0.4"]
    work = frame.copy()
    work["gap_bin"] = pd.cut(work["barthag_gap"], bins=bins, labels=labels)
    grouped = work.groupby("gap_bin", observed=False)["home_win"].mean()
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.bar(range(len(grouped)), grouped.to_numpy(), color="#1f4b99")
    axis.set_xticks(range(len(grouped)), labels, rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Home win rate")
    axis.set_xlabel("Prior-season Barthag gap (home minus away)")
    axis.set_title("Home teams win more often when they were better last year")
    axis.axhline(work["home_win"].mean(), color="#c47b00", linestyle="--", linewidth=1.2, label="Overall home win rate")
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def build_all(skip_db: bool = False) -> dict:
    games = clean_games(fetch_games())
    ratings = clean_ratings(fetch_ratings())
    merged = merge_games_and_ratings(games, ratings)
    if not skip_db:
        merged = load_postgres(merged)
    profiled = profile_frame(merged)
    trained = train_baseline(profiled)
    published = publish(trained)
    return {"games": games, "ratings": ratings, "merged": merged, "published": published, "baseline": trained["baseline"]}


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the college basketball modeling table.")
    parser.add_argument("--skip-db", action="store_true")
    args = parser.parse_args()
    result = build_all(skip_db=args.skip_db)
    print(json.dumps({"rows": result["published"]["rows"], "baseline": result["baseline"]}, indent=2))
