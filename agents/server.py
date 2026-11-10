"""Live page for the home-win agent graph."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from agents.graph import iter_updates
from pipeline.cbb import PUBLISHED

ROOT = Path(__file__).resolve().parents[1]
DATASET = PUBLISHED / "merged_latest.csv"
app = FastAPI(title="Home-win agent loop")


@app.get("/")
def index():
    return FileResponse(ROOT / "ui" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "dataset": DATASET.exists()}


@app.get("/api/run")
def run():
    if not DATASET.exists():
        return JSONResponse(
            {"error": "data/published/merged_latest.csv is missing. Run the Airflow DAG first."},
            status_code=400,
        )

    def events():
        try:
            for update in iter_updates(str(DATASET)):
                yield f"data: {json.dumps(update, default=str)}\n\n"
            yield 'data: {"event": "done"}\n\n'
        except Exception as exc:  # noqa: BLE001 - surface the failure on the page
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
