"""A home-win research loop.

This is not the class fraud loop. The critic has two different ways back:

- weak separation goes to the feature planner, which adds a richer prior-season
  feature set (tempo, then Barthag, then the full rating profile)
- overfit goes straight back to the trainer with stronger regularization

The holdout slice is scored once, after the critic stops.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from pipeline.cbb import (
    FEATURE_SETS,
    REPO,
    classification_metrics,
    read_games,
    split_frame,
)

ACCEPT_AUC = 0.70
MAX_ATTEMPTS = 4
OVERFIT_GAP = 0.08
REPORT_PATH = REPO / "artifacts" / "agents" / "evaluation.json"
FEATURE_ORDER = ["tempo", "barthag", "full"]
FEATURE_LABELS = {
    "tempo": "pace",
    "barthag": "team strength",
    "full": "strength, offense, defense, and pace",
}


class LoopState(TypedDict, total=False):
    dataset_path: str
    attempt: int
    max_attempts: int
    feature_set: str
    C: float
    route: str
    stop_reason: str
    note: str
    profile: dict
    train_metrics: dict
    val_metrics: dict
    holdout_metrics: dict
    best: dict
    history: Annotated[list, operator.add]


def node_steward(state: LoopState) -> dict:
    frame = read_games(state["dataset_path"])
    _train, _val, _hold = split_frame(frame)
    win_rate = float(frame["home_win"].mean())
    if not 0.45 <= win_rate <= 0.85:
        raise RuntimeError(f"Home win rate {win_rate:.3f} is outside the expected range")
    profile = {
        "rows": int(len(frame)),
        "date_min": str(frame["game_date"].min()),
        "date_max": str(frame["game_date"].max()),
        "home_win_rate": round(win_rate, 4),
        "seasons": sorted(int(year) for year in frame["season"].unique()),
    }
    return {
        "profile": profile,
        "note": (
            f"{profile['rows']} home games from {profile['date_min']} to {profile['date_max']}. "
            f"Home teams won {profile['home_win_rate']:.1%}."
        ),
    }


def node_feature_planner(state: LoopState) -> dict:
    current = state.get("feature_set")
    if current not in FEATURE_ORDER:
        chosen = "tempo"
    else:
        chosen = FEATURE_ORDER[min(FEATURE_ORDER.index(current) + 1, len(FEATURE_ORDER) - 1)]
    notes = {
        "tempo": "First pass uses only last season's pace, possessions per game. Pace should not decide the winner.",
        "barthag": "Pace was weak. Switch to last season's team strength for both teams.",
        "full": "Team strength was not enough. Add last season's offense, defense, and pace.",
    }
    return {"feature_set": chosen, "C": 1.0, "note": notes[chosen]}


def node_trainer(state: LoopState) -> dict:
    frame = read_games(state["dataset_path"])
    train, _val, _hold = split_frame(frame)
    features = FEATURE_SETS[state["feature_set"]]
    metrics = classification_metrics(train, train, features, state["C"])
    attempt = int(state.get("attempt", 0)) + 1
    return {
        "attempt": attempt,
        "train_metrics": metrics,
        "note": (
            f"Attempt {attempt}: logistic regression on {FEATURE_LABELS[state['feature_set']]} "
            f"with C={state['C']}. Training ROC-AUC {metrics['roc_auc']:.3f}."
        ),
    }


def node_scorer(state: LoopState) -> dict:
    frame = read_games(state["dataset_path"])
    train, val, _hold = split_frame(frame)
    features = FEATURE_SETS[state["feature_set"]]
    metrics = classification_metrics(train, val, features, state["C"])
    return {
        "val_metrics": metrics,
        "note": (
            f"Validation accuracy {metrics['accuracy']:.3f}, "
            f"F1 {metrics['f1']:.3f}, ROC-AUC {metrics['roc_auc']:.3f}."
        ),
    }


def node_critic(state: LoopState) -> dict:
    val_auc = float(state["val_metrics"]["roc_auc"])
    train_auc = float(state["train_metrics"]["roc_auc"])
    attempt = int(state["attempt"])
    record = {
        "attempt": attempt,
        "feature_set": state["feature_set"],
        "C": state["C"],
        "train": state["train_metrics"],
        "val": state["val_metrics"],
    }
    best = dict(state.get("best") or {})
    if not best or val_auc > float(best.get("val_roc_auc", -1)):
        best = {
            "feature_set": state["feature_set"],
            "C": state["C"],
            "val_roc_auc": val_auc,
            "attempt": attempt,
        }
    if val_auc >= ACCEPT_AUC:
        route = "reporter"
        reason = "auc"
        note = f"Validation ROC-AUC {val_auc:.3f} cleared {ACCEPT_AUC:.2f}."
        extra: dict = {}
    elif state.get("feature_set") == "full" or attempt >= int(state.get("max_attempts", MAX_ATTEMPTS)):
        route = "reporter"
        reason = "features_exhausted" if state.get("feature_set") == "full" else "budget"
        note = (
            f"Validation ROC-AUC {val_auc:.3f} is still below {ACCEPT_AUC:.2f}, "
            "and the full prior-season profile is already in use. Report the best setup."
            if reason == "features_exhausted"
            else f"Attempt budget of {state.get('max_attempts', MAX_ATTEMPTS)} is used. Keep the best setup."
        )
        extra = {}
    elif train_auc - val_auc > OVERFIT_GAP and float(state["C"]) > 0.1:
        route = "trainer"
        reason = "overfit"
        note = (
            f"Training AUC {train_auc:.3f} is well ahead of validation {val_auc:.3f}. "
            "Refit with stronger regularization."
        )
        extra = {"C": 0.1}
    else:
        route = "feature_planner"
        reason = "weak_separation"
        note = f"Validation ROC-AUC {val_auc:.3f} is below {ACCEPT_AUC:.2f}. Ask for a better feature set."
        extra = {}
    record["route"] = route
    record["reason"] = reason
    return {
        "best": best,
        "route": route,
        "stop_reason": reason,
        "note": note,
        "history": [record],
        **extra,
    }


def node_reporter(state: LoopState) -> dict:
    frame = read_games(state["dataset_path"])
    train, val, holdout = split_frame(frame)
    development = pd.concat([train, val], ignore_index=True)
    best = state["best"]
    features = FEATURE_SETS[best["feature_set"]]
    metrics = classification_metrics(development, holdout, features, float(best["C"]))
    report = {
        "question": "Did the home team win?",
        "target": "home_win",
        "accept_auc": ACCEPT_AUC,
        "stop_reason": state.get("stop_reason"),
        "best": best,
        "holdout": metrics,
        "attempts": state.get("history", []),
        "profile": state.get("profile", {}),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return {
        "holdout_metrics": metrics,
        "note": (
            f"Holdout accuracy {metrics['accuracy']:.3f}, "
            f"F1 {metrics['f1']:.3f}, ROC-AUC {metrics['roc_auc']:.3f} "
            f"using {FEATURE_LABELS[best['feature_set']]}."
        ),
    }


def route_after_critic(state: LoopState) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(LoopState)
    graph.add_node("steward", node_steward)
    graph.add_node("feature_planner", node_feature_planner)
    graph.add_node("trainer", node_trainer)
    graph.add_node("scorer", node_scorer)
    graph.add_node("critic", node_critic)
    graph.add_node("reporter", node_reporter)
    graph.add_edge(START, "steward")
    graph.add_edge("steward", "feature_planner")
    graph.add_edge("feature_planner", "trainer")
    graph.add_edge("trainer", "scorer")
    graph.add_edge("scorer", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "feature_planner": "feature_planner",
            "trainer": "trainer",
            "reporter": "reporter",
        },
    )
    graph.add_edge("reporter", END)
    return graph.compile()


def initial_state(dataset_path: str) -> LoopState:
    return {
        "dataset_path": dataset_path,
        "attempt": 0,
        "max_attempts": MAX_ATTEMPTS,
        "C": 1.0,
        "route": "start",
        "history": [],
    }


def iter_updates(dataset_path: str):
    compiled = build_graph()
    for update in compiled.stream(initial_state(dataset_path), stream_mode="updates"):
        for node, patch in update.items():
            yield {"node": node, "patch": patch}
