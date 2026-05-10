import sys
import time
import random
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# to run: python -m uvicorn ui.app:app --reload --port 8000

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.inference import (
    load_all_models,
    run_full_pipeline,
    model_a_predict,
    get_model_a_metrics,
    get_model_b_metrics,
)

app = FastAPI(title="RACE Quiz Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session log
_session_log: list[dict] = []


@app.on_event("startup")
def startup():
    load_all_models()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    article: str

class VerifyRequest(BaseModel):
    article:  str
    question: str
    option:   str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/generate")
def generate(req: GenerateRequest):
    try:
        if not req.article.strip():
            raise HTTPException(status_code=422, detail="Article cannot be empty.")
        result = run_full_pipeline(req.article)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify")
def verify(req: VerifyRequest):
    try:
        if not req.article.strip() or not req.question.strip() or not req.option.strip():
            raise HTTPException(status_code=422, detail="article, question and option are required.")
        t0     = time.time()
        result = model_a_predict(req.article, req.question, req.option)
        entry  = {
            'timestamp':   time.strftime('%Y-%m-%dT%H:%M:%S'),
            'question':    req.question,
            'option':      req.option,
            'is_correct':  result['is_correct'],
            'confidence':  result['confidence'],
            'latency_ms':  result['latency_ms'],
        }
        _session_log.append(entry)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/random-sample")
def random_sample():
    try:
        from src.inference import model_b_predict
        df = pd.read_csv('data/raw/train.csv')
        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        df.dropna(subset=['A', 'B', 'C', 'D'], inplace=True)
        row      = df.sample(1, random_state=random.randint(0, 99999)).iloc[0]
        article  = str(row['article'])
        question = str(row['question'])
        answer_label = str(row['answer'])
        answer_text  = str(row[answer_label])

        b = model_b_predict(article, question, answer_text)

        options_list = [answer_text] + b['distractors'][:3]
        while len(options_list) < 4:
            options_list.append('None of the above')
        random.shuffle(options_list)
        labels  = ['A', 'B', 'C', 'D']
        options = {labels[i]: options_list[i] for i in range(4)}
        correct_label = labels[options_list.index(answer_text)]

        return {
            'article':            article,
            'question':           question,
            'options':            options,
            'correct_answer':     correct_label,
            'hints':              b['hints'],
            'model_a_latency_ms': 0,
            'model_b_latency_ms': b['latency_ms'],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
def metrics():
    try:
        return {
            'model_a': get_model_a_metrics(),
            'model_b': get_model_b_metrics(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export-session")
def export_session():
    try:
        if not _session_log:
            raise HTTPException(status_code=422, detail="No session data to export.")
        df     = pd.DataFrame(_session_log)
        buf    = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=session_log.csv"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
