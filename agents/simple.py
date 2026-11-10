"""One-pass agent, the simple baseline.

Check the table, pick prior-season Barthag, train once, score the holdout.
There is no critic and no retry. That is the class demo's simple shape:
a fixed sequence of agents, not a router that changes its mind.
"""

from __future__ import annotations

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.graph import LoopState, node_steward
from pipeline.cbb import FEATURE_SETS, classification_metrics, read_games, split_frame


def node_pick(state: LoopState) -> dict:
    del state
    return {
        "feature_set": "barthag",
        "C": 1.0,
        "attempt": 1,
        "note": "Simple loop uses last season's team strength for both teams and does not retry.",
    }


def node_holdout(state: LoopState) -> dict:
    frame = read_games(state["dataset_path"])
    train, validation, holdout = split_frame(frame)
    development = pd.concat([train, validation], ignore_index=True)
    metrics = classification_metrics(development, holdout, FEATURE_SETS["barthag"], 1.0)
    return {
        "holdout_metrics": metrics,
        "note": (
            f"Holdout accuracy {metrics['accuracy']:.3f}, "
            f"F1 {metrics['f1']:.3f}, ROC-AUC {metrics['roc_auc']:.3f}."
        ),
    }


def build_simple_graph():
    graph = StateGraph(LoopState)
    graph.add_node("steward", node_steward)
    graph.add_node("pick", node_pick)
    graph.add_node("holdout", node_holdout)
    graph.add_edge(START, "steward")
    graph.add_edge("steward", "pick")
    graph.add_edge("pick", "holdout")
    graph.add_edge("holdout", END)
    return graph.compile()


def run_simple(dataset_path: str) -> dict:
    trace = []
    final = {}
    for update in build_simple_graph().stream(
        {"dataset_path": dataset_path, "history": []},
        stream_mode="updates",
    ):
        for node, patch in update.items():
            trace.append(node)
            final.update(patch)
    return {"trace": trace, "holdout": final.get("holdout_metrics", {})}
