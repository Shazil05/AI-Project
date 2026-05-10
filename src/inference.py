import re
import sys
import time
import random
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import one_hot_encode, encode_row, extract_lexical_features
from src.model_a_train import generate_question
from src.model_b_train import select_top_distractors, generate_hints

PROC  = Path('data/processed')
OUT_A = Path('models/model_a/traditional')
OUT_B = Path('models/model_b/traditional')

# Module-level model cache — loaded once at startup
_models = {}
_metrics_cache = {}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_all_models() -> None:
    global _models, _metrics_cache
    _models['vocab']             = joblib.load(PROC  / 'vocab.joblib')
    _models['lr']                = joblib.load(OUT_A / 'lr.joblib')
    _models['svm']               = joblib.load(OUT_A / 'svm.joblib')
    _models['nb']                = joblib.load(OUT_A / 'nb.joblib')
    _models['rf']                = joblib.load(OUT_A / 'rf.joblib')
    _models['ensemble']          = joblib.load(OUT_A / 'ensemble.joblib')
    _models['distractor_ranker'] = joblib.load(OUT_B / 'distractor_ranker.joblib')
    _models['hint_scorer']       = joblib.load(OUT_B / 'hint_scorer.joblib')
    print("All models loaded.")
    print("Pre-computing metrics cache...")
    _compute_and_cache_metrics()
    print("Metrics cached.")


def _ensure_loaded() -> None:
    if not _models:
        load_all_models()


# ---------------------------------------------------------------------------
# Model A predict
# ---------------------------------------------------------------------------

def model_a_predict(article: str, question: str, option: str) -> dict:
    _ensure_loaded()
    t0    = time.time()
    vocab = _models['vocab']
    lr    = _models['lr']
    svm   = _models['svm']
    ens   = _models['ensemble']

    feat = encode_row(article, question, option, vocab)

    # Stacking ensemble: needs proba from base models
    lr_proba  = lr.predict_proba([feat])
    svm_proba = svm.predict_proba([feat])
    meta_feat = np.column_stack([lr_proba, svm_proba])

    proba      = ens.predict_proba(meta_feat)[0]
    is_correct = bool(np.argmax(proba) == 1)
    confidence = float(proba[1])

    return {
        'is_correct':   is_correct,
        'confidence':   round(confidence, 4),
        'latency_ms':   round((time.time() - t0) * 1000, 2),
    }


# ---------------------------------------------------------------------------
# Model B predict
# ---------------------------------------------------------------------------

def model_b_predict(article: str, question: str, answer: str) -> dict:
    _ensure_loaded()
    t0 = time.time()

    distractors = select_top_distractors(
        article, answer,
        _models['distractor_ranker'],
        _models['vocab'],
    )
    hints = generate_hints(
        article, question, answer,
        _models['hint_scorer'],
        _models['vocab'],
    )

    return {
        'distractors': distractors,
        'hints':       hints,
        'latency_ms':  round((time.time() - t0) * 1000, 2),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(article: str) -> dict:
    _ensure_loaded()
    t_a0 = time.time()

    vocab     = _models['vocab']
    svm_model = _models['svm']

    sentences   = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 10]
    answer_text = sentences[0].split()[-1] if sentences else 'unknown'

    question = generate_question(article, answer_text, vocab, svm_model)
    t_a_ms   = round((time.time() - t_a0) * 1000, 2)

    t_b0 = time.time()
    b    = model_b_predict(article, question, answer_text)
    t_b_ms = round((time.time() - t_b0) * 1000, 2)

    # Exclude any distractor that exactly matches the answer
    distractors = [d for d in b['distractors'] if d.strip().lower() != answer_text.strip().lower()]

    options_list = [answer_text] + distractors[:3]
    while len(options_list) < 4:
        options_list.append('None of the above')

    random.shuffle(options_list)
    labels        = ['A', 'B', 'C', 'D']
    options       = {labels[i]: options_list[i] for i in range(4)}
    correct_label = labels[options_list.index(answer_text)]

    return {
        'article':           article,
        'question':          question,
        'options':           options,
        'correct_answer':    correct_label,
        'hints':             b['hints'],
        'model_a_latency_ms': t_a_ms,
        'model_b_latency_ms': t_b_ms,
    }


# ---------------------------------------------------------------------------
# Metrics (loaded from saved eval results or computed on the fly)
# ---------------------------------------------------------------------------

def _compute_and_cache_metrics() -> None:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, confusion_matrix,
    )
    X_test, y_test = joblib.load(PROC / 'X_verify_test.joblib')
    lr  = _models['lr']
    svm = _models['svm']
    ens = _models['ensemble']

    lr_preds  = lr.predict(X_test)
    svm_preds = svm.predict(X_test)
    X_meta    = np.column_stack([lr.predict_proba(X_test), svm.predict_proba(X_test)])
    ens_preds = ens.predict(X_meta)

    def _m(preds):
        return {
            'accuracy': round(accuracy_score(y_test, preds), 4),
            'f1':       round(f1_score(y_test, preds, average='macro'), 4),
        }

    _metrics_cache['model_a'] = {
        'accuracy':         round(accuracy_score(y_test, ens_preds), 4),
        'macro_f1':         round(f1_score(y_test, ens_preds, average='macro'), 4),
        'precision':        round(precision_score(y_test, ens_preds, average='macro', zero_division=0), 4),
        'recall':           round(recall_score(y_test, ens_preds, average='macro', zero_division=0), 4),
        'confusion_matrix': confusion_matrix(y_test, ens_preds).tolist(),
        'model_comparison': {
            'lr':       _m(lr_preds),
            'svm':      _m(svm_preds),
            'ensemble': _m(ens_preds),
        },
    }


def get_model_a_metrics() -> dict:
    _ensure_loaded()
    return _metrics_cache['model_a']


def get_model_b_metrics() -> dict:
    return {
        'distractor_precision': 0.005,
        'distractor_recall':    0.005,
        'distractor_f1':        0.005,
        'distractor_accuracy':  1.0,
        'hint_precision_at_3':  0.145,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    load_all_models()

    sample_article = (
        "Alice lives with her grandparents in a small town. "
        "Her last name is Black. She loves reading books and "
        "playing with her two brothers and one sister every afternoon."
    )

    print("\n--- model_a_predict ---")
    result = model_a_predict(sample_article, "What is the girl's last name?", "Black")
    print(result)

    print("\n--- model_b_predict ---")
    result = model_b_predict(sample_article, "What is the girl's last name?", "Black")
    print(result)

    print("\n--- run_full_pipeline ---")
    result = run_full_pipeline(sample_article)
    for k, v in result.items():
        if k != 'article':
            print(f"  {k}: {v}")

    print("\n--- get_model_a_metrics ---")
    metrics = get_model_a_metrics()
    print({k: v for k, v in metrics.items() if k != 'confusion_matrix'})

    print("\ninference.py smoke test complete.")
