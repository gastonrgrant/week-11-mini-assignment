"""Scheduled college-basketball pipeline.

Two TaskGroups start together: game logs, and the previous season's team
ratings. The merge waits for both. Later tasks load Postgres, profile the
table, fit one logistic regression, and publish the CSV the agent loop reads.

Schedule is 06:00 America/New_York. catchup is off, so unpausing does not
replay old days. Trigger still runs the pipeline immediately.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from airflow.decorators import dag, task
from airflow.timetables.trigger import CronTriggerTimetable
from airflow.utils.task_group import TaskGroup

from pipeline.cbb import (  # noqa: E402
    clean_games,
    clean_ratings,
    fetch_games,
    fetch_ratings,
    load_postgres,
    merge_games_and_ratings,
    profile_frame,
    publish,
    train_baseline,
)


@dag(
    dag_id="cbb_home_win_pipeline",
    description="Join Barttorvik game logs to prior-season ratings, then score a home-win baseline.",
    start_date=datetime(2024, 11, 1),
    schedule=CronTriggerTimetable("0 6 * * *", timezone="America/New_York"),
    catchup=False,
    default_args={"retries": 1},
    tags=["cbb", "barttorvik", "week11"],
)
def cbb_home_win_pipeline():
    with TaskGroup("game_logs"):
        fetched_games = task(fetch_games)()
        cleaned_games = task(clean_games)(fetched_games)

    with TaskGroup("team_ratings"):
        fetched_ratings = task(fetch_ratings)()
        cleaned_ratings = task(clean_ratings)(fetched_ratings)

    merged = task(merge_games_and_ratings)(cleaned_games, cleaned_ratings)
    loaded = task(load_postgres)(merged)
    profiled = task(profile_frame)(loaded)
    trained = task(train_baseline)(profiled)
    task(publish)(trained)


cbb_home_win_pipeline()
