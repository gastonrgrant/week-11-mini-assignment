"""Score the adaptive loop against two baselines.

Baselines from class:
- the fixed Airflow pipeline, which always trains Barthag
- a simple one-pass agent, which checks the table, picks Barthag, and stops

Each case records the input, the expected behavior, how it is scored,
and the pass/fail line. Intermediate cases score a single routing decision.
Final cases score the holdout. Coordination cases score whether the agents
handed work to the right next agent.

Run: python -m eval.suite
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from agents.graph import (
    ACCEPT_AUC,
    build_graph,
    initial_state,
    node_critic,
)
from agents.simple import run_simple
from pipeline.cbb import (
    FEATURE_SETS,
    PUBLISHED,
    REPO,
    classification_metrics,
    read_games,
    split_frame,
)

REPORT_PATH = REPO / "artifacts" / "eval" / "report.json"
DATASET = PUBLISHED / "merged_latest.csv"


def _case(case_id: str, case_input: str, expected: str, scoring: str, threshold: float, score: float) -> dict:
    return {
        "id": case_id,
        "input": case_input,
        "expected": expected,
        "scoring": scoring,
        "threshold": threshold,
        "score": round(float(score), 4),
        "passed": bool(score >= threshold),
    }


def airflow_holdout(frame: pd.DataFrame) -> dict:
    train, validation, holdout = split_frame(frame)
    development = pd.concat([train, validation], ignore_index=True)
    return classification_metrics(development, holdout, FEATURE_SETS["barthag"], 1.0)


def run_adaptive(dataset_path: str) -> dict:
    trace = []
    history = []
    holdout = {}
    for update in build_graph().stream(initial_state(dataset_path), stream_mode="updates"):
        for node, patch in update.items():
            trace.append(node)
            if patch.get("history"):
                history.extend(patch["history"])
            if patch.get("holdout_metrics"):
                holdout = patch["holdout_metrics"]
    return {"trace": trace, "history": history, "holdout": holdout}


def critic_case(train_auc: float, val_auc: float, feature_set: str, attempt: int) -> dict:
    return node_critic(
        {
            "attempt": attempt,
            "max_attempts": 4,
            "feature_set": feature_set,
            "C": 1.0,
            "train_metrics": {"roc_auc": train_auc, "accuracy": 0.0, "f1": 0.0, "n": 1},
            "val_metrics": {"roc_auc": val_auc, "accuracy": 0.0, "f1": 0.0, "n": 1},
        }
    )


def evaluate(dataset_path: str = str(DATASET)) -> dict:
    frame = read_games(dataset_path)
    fixed = airflow_holdout(frame)
    simple = run_simple(dataset_path)
    adaptive = run_adaptive(dataset_path)
    history = adaptive["history"]
    trace = adaptive["trace"]

    expected_trace = [
        "steward",
        "feature_planner",
        "trainer",
        "scorer",
        "critic",
        "feature_planner",
        "trainer",
        "scorer",
        "critic",
        "feature_planner",
        "trainer",
        "scorer",
        "critic",
        "reporter",
    ]
    gain = history[1]["val"]["roc_auc"] - history[0]["val"]["roc_auc"]
    overfit = critic_case(0.80, 0.60, "barthag", 1)
    weak = critic_case(0.55, 0.50, "tempo", 1)
    accept = critic_case(0.80, 0.75, "barthag", 1)

    cases = [
        _case(
            "fixed_airflow_beats_chance",
            "Published home games. Fixed pipeline trains Barthag once.",
            "Holdout ROC-AUC is above a coin flip.",
            "Holdout ROC-AUC",
            0.60,
            fixed["roc_auc"],
        ),
        _case(
            "simple_agent_matches_fixed_pipeline",
            "Same table. One-pass agent also trains Barthag and stops.",
            "Its holdout ROC-AUC matches the fixed pipeline within 0.01.",
            "1 if the absolute AUC gap is under 0.01, else 0",
            1.0,
            1.0 if abs(simple["holdout"]["roc_auc"] - fixed["roc_auc"]) < 0.01 else 0.0,
        ),
        _case(
            "simple_agent_has_no_retry",
            "One-pass agent on the published table.",
            "The only agents are steward, pick, and holdout.",
            "1 if that is the whole trace, else 0",
            1.0,
            1.0 if simple["trace"] == ["steward", "pick", "holdout"] else 0.0,
        ),
        _case(
            "adaptive_rejects_tempo",
            "Adaptive loop, first validation score.",
            "Tempo is sent back to the feature planner.",
            "1 if attempt 1 is tempo and the route is feature_planner, else 0",
            1.0,
            1.0
            if history[0]["feature_set"] == "tempo" and history[0]["route"] == "feature_planner"
            else 0.0,
        ),
        _case(
            "adaptive_tempo_is_weak",
            "Adaptive loop, first validation ROC-AUC.",
            "Tempo does not separate winners. AUC stays under 0.55.",
            "1 if validation ROC-AUC is under 0.55, else 0",
            1.0,
            1.0 if history[0]["val"]["roc_auc"] < 0.55 else 0.0,
        ),
        _case(
            "adaptive_barthag_is_a_real_gain",
            "Adaptive loop, first two validation scores.",
            "Adding Barthag lifts validation ROC-AUC by at least 0.10.",
            "Validation AUC of attempt 2 minus attempt 1",
            0.10,
            gain,
        ),
        _case(
            "adaptive_stops_on_full_profile",
            "Adaptive loop, third attempt.",
            "The full profile is reported instead of trying the same features again.",
            "1 if attempt 3 is full and the route is reporter, else 0",
            1.0,
            1.0 if history[2]["feature_set"] == "full" and history[2]["route"] == "reporter" else 0.0,
        ),
        _case(
            "adaptive_agent_order",
            "Adaptive loop trace on the published table.",
            "Steward, then three plan-train-score-critic cycles, then one reporter.",
            "1 if the node order matches, else 0",
            1.0,
            1.0 if trace == expected_trace else 0.0,
        ),
        _case(
            "holdout_scored_once",
            "Adaptive loop trace.",
            "The reporter runs once. Earlier attempts stay on the validation slice.",
            "1 if reporter appears once, else 0",
            1.0,
            1.0 if trace.count("reporter") == 1 else 0.0,
        ),
        _case(
            "overfit_routes_to_trainer",
            "Critic sees training AUC 0.80 and validation AUC 0.60 on Barthag.",
            "It retries the trainer with stronger regularization.",
            "1 if the route is trainer and C becomes 0.1, else 0",
            1.0,
            1.0 if overfit["route"] == "trainer" and overfit["C"] == 0.1 else 0.0,
        ),
        _case(
            "weak_score_routes_to_planner",
            "Critic sees validation AUC 0.50 on tempo, attempt 1.",
            "It sends the run back to the feature planner.",
            "1 if the route is feature_planner, else 0",
            1.0,
            1.0 if weak["route"] == "feature_planner" else 0.0,
        ),
        _case(
            "strong_score_stops",
            f"Critic sees validation AUC 0.75, above the {ACCEPT_AUC:.2f} bar.",
            "It accepts and sends the run to the reporter.",
            "1 if the route is reporter, else 0",
            1.0,
            1.0 if accept["route"] == "reporter" and accept["stop_reason"] == "auc" else 0.0,
        ),
        _case(
            "adaptive_holdout_auc",
            "Published table, adaptive loop, sealed last 15% of games.",
            "Holdout ROC-AUC is at least 0.64.",
            "Holdout ROC-AUC",
            0.64,
            adaptive["holdout"]["roc_auc"],
        ),
        _case(
            "adaptive_holdout_accuracy",
            "Same holdout. Home teams win about 61% of these late games.",
            "Accuracy is at least 0.62, above always picking the home team.",
            "Holdout accuracy",
            0.62,
            adaptive["holdout"]["accuracy"],
        ),
        _case(
            "adaptive_matches_or_beats_simple",
            "Holdout ROC-AUC of the adaptive loop versus the one-pass agent.",
            "The extra tries do not make the final model worse.",
            "Adaptive AUC minus simple AUC. Pass at 0, meaning a tie or a gain.",
            0.0,
            adaptive["holdout"]["roc_auc"] - simple["holdout"]["roc_auc"],
        ),
    ]
    passed = sum(1 for case in cases if case["passed"])
    summary = (
        f"The fixed Airflow pipeline trains Barthag once and scores "
        f"{fixed['roc_auc']:.3f} ROC-AUC on the holdout. The simple one-pass agent "
        f"does that same job and scores {simple['holdout']['roc_auc']:.3f}. "
        f"The adaptive critic throws out tempo, then Barthag, then keeps the full "
        f"prior-season profile and scores {adaptive['holdout']['roc_auc']:.3f}. "
        f"The extra routing is a small gain. Last season's ratings are a limited signal "
        f"in all three systems. {passed} of {len(cases)} evaluation cases passed."
    )
    return {
        "summary": summary,
        "baselines": {
            "fixed_airflow": fixed,
            "simple_agent": simple["holdout"],
            "adaptive": adaptive["holdout"],
        },
        "passed": passed,
        "total": len(cases),
        "cases": cases,
    }


def main() -> None:
    if not DATASET.exists():
        raise SystemExit(f"Missing {DATASET}. Run the Airflow pipeline first.")
    report = evaluate()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    for case in report["cases"]:
        mark = "pass" if case["passed"] else "FAIL"
        print(f"{mark:4} {case['id']:38} score={case['score']}  line={case['threshold']}")
    print(report["summary"])
    print(f"Wrote {REPORT_PATH}")
    if report["passed"] != report["total"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
