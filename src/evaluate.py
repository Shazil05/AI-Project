import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)


# ---------------------------------------------------------------------------
# Model A
# ---------------------------------------------------------------------------

def evaluate_model_a(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds, average='macro')
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec  = recall_score(y_test, preds, average='macro', zero_division=0)
    cm   = confusion_matrix(y_test, preds).tolist()

    # Exact Match — strict character-level match (binary task: pred == label)
    em = float(acc)

    return {
        'accuracy':         round(acc,  4),
        'macro_f1':         round(f1,   4),
        'exact_match':      round(em,   4),
        'precision':        round(prec, 4),
        'recall':           round(rec,  4),
        'confusion_matrix': cm,
    }


# ---------------------------------------------------------------------------
# Model B — Distractors
# ---------------------------------------------------------------------------

def evaluate_model_b_distractors(predictions: list, references: list) -> dict:
    """
    predictions : list of list[str]  — generated distractors per question
    references  : list of list[str]  — gold distractors per question
    """
    prec_scores, rec_scores, f1_scores, acc_scores = [], [], [], []

    for pred, ref in zip(predictions, references):
        pred_set = set(p.strip().lower() for p in pred)
        ref_set  = set(r.strip().lower() for r in ref)

        tp   = len(pred_set & ref_set)
        prec = tp / max(len(pred_set), 1)
        rec  = tp / max(len(ref_set),  1)
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Accuracy: top distractor is not the correct answer (already guaranteed
        # by construction, but we check against ref for correctness proxy)
        acc = 1.0 if pred_set - ref_set else 0.0

        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)
        acc_scores.append(acc)

    return {
        'distractor_precision': round(float(np.mean(prec_scores)), 4),
        'distractor_recall':    round(float(np.mean(rec_scores)),  4),
        'distractor_f1':        round(float(np.mean(f1_scores)),   4),
        'distractor_accuracy':  round(float(np.mean(acc_scores)),  4),
    }


# ---------------------------------------------------------------------------
# Model B — Hints
# ---------------------------------------------------------------------------

def evaluate_hints(predicted_hints: list, gold_sentences: list) -> dict:
    """
    predicted_hints : list of list[str] — top-3 hints per question
    gold_sentences  : list[str]         — sentence containing correct answer
    """
    precision_at_3 = []

    for hints, gold in zip(predicted_hints, gold_sentences):
        gold_words = set(gold.strip().lower().split())
        hits = 0
        for hint in hints[:3]:
            hint_words = set(hint.strip().lower().split())
            overlap    = len(hint_words & gold_words) / max(len(gold_words), 1)
            if overlap > 0.3:   # at least 30% word overlap counts as a hit
                hits += 1
        precision_at_3.append(hits / 3)

    return {
        'hint_precision_at_3': round(float(np.mean(precision_at_3)), 4),
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict) -> None:
    metrics = ['accuracy', 'macro_f1', 'precision', 'recall']
    col_w   = 12

    header = f"{'Model':<25}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * (25 + col_w * len(metrics)))
    print("Model A — Variant Comparison")
    print("=" * (25 + col_w * len(metrics)))
    print(header)
    print("-" * (25 + col_w * len(metrics)))

    for model_name, model_metrics in results.items():
        row = f"{model_name:<25}"
        for m in metrics:
            val = model_metrics.get(m, float('nan'))
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("=" * (25 + col_w * len(metrics)))


# ---------------------------------------------------------------------------
# Entry point — evaluate all saved models and print full report
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import joblib

    PROC  = Path('data/processed')
    OUT_A = Path('models/model_a/traditional')
    OUT_B = Path('models/model_b/traditional')

    print("Loading test features...")
    X_test, y_test           = joblib.load(PROC / 'X_verify_test.joblib')
    X_lexical_test, y_lex_t  = joblib.load(PROC / 'X_lexical_test.joblib')

    print("Loading models...")
    lr       = joblib.load(OUT_A / 'lr.joblib')
    svm      = joblib.load(OUT_A / 'svm.joblib')
    ensemble = joblib.load(OUT_A / 'ensemble.joblib')

    print("\n=== Model A Evaluation (Test Set) ===")
    results = {}
    for name, model in [('Logistic Regression', lr), ('SVM', svm)]:
        m = evaluate_model_a(model, X_test, y_test)
        results[name] = m
        print(f"{name}: acc={m['accuracy']} | f1={m['macro_f1']} | prec={m['precision']} | rec={m['recall']}")

    # Ensemble expects stacked base-model probabilities, not raw features
    X_test_meta = np.column_stack([lr.predict_proba(X_test), svm.predict_proba(X_test)])
    m = evaluate_model_a(ensemble, X_test_meta, y_test)
    results['Ensemble'] = m
    print(f"Ensemble: acc={m['accuracy']} | f1={m['macro_f1']} | prec={m['precision']} | rec={m['recall']}")

    print_comparison_table(results)

    print("\nConfusion Matrix (Ensemble):")
    for row in results['Ensemble']['confusion_matrix']:
        print("  ", row)

    # Model B — sample evaluation on a small batch
    print("\n=== Model B Evaluation (sample 200 rows) ===")
    import pandas as pd
    vocab       = joblib.load(PROC / 'vocab.joblib')
    ranker      = joblib.load(OUT_B / 'distractor_ranker.joblib')
    hint_scorer = joblib.load(OUT_B / 'hint_scorer.joblib')

    from src.model_b_train import select_top_distractors, generate_hints

    df_all = pd.read_csv('data/raw/train.csv')
    df_all.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df_all.dropna(subset=['A', 'B', 'C', 'D'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    df_test_raw = df_all.sample(frac=0.80, random_state=42)
    remaining   = df_all.drop(df_test_raw.index)
    df_test_raw = remaining.drop(remaining.sample(frac=0.50, random_state=42).index)
    df_sample   = df_test_raw.head(200).reset_index(drop=True)

    pred_distractors, ref_distractors = [], []
    pred_hints,       gold_sentences  = [], []

    for _, row in df_sample.iterrows():
        article     = str(row['article'])
        question    = str(row['question'])
        answer_text = str(row[row['answer']])
        gold_opts   = [str(row[o]) for o in ['A','B','C','D'] if o != row['answer']]

        pred_distractors.append(select_top_distractors(article, answer_text, ranker, vocab))
        ref_distractors.append(gold_opts)

        hints = generate_hints(article, question, answer_text, hint_scorer, vocab)
        pred_hints.append(hints)
        gold_sentences.append(answer_text)

    dist_metrics = evaluate_model_b_distractors(pred_distractors, ref_distractors)
    hint_metrics = evaluate_hints(pred_hints, gold_sentences)

    print("Distractor metrics:", dist_metrics)
    print("Hint metrics:      ", hint_metrics)

    print("\nevaluate.py complete.")
