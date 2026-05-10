import re
import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from src.preprocessing import (
    clean_text, one_hot_encode, compute_cosine_similarity,
    extract_lexical_features, encode_row,
)

PROC  = Path('data/processed')
OUT_A = Path('models/model_a/traditional')


def _load(name):
    return joblib.load(PROC / name)


# ---------------------------------------------------------------------------
# 2A - Supervised models
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='saga',
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    model.fit(X_train, y_train)
    OUT_A.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_A / 'lr.joblib')
    print("LR saved.")
    return model


def train_svm(X_train, y_train) -> CalibratedClassifierCV:
    base = LinearSVC(
        class_weight='balanced',
        max_iter=2000,
        random_state=42,
        verbose=1,
    )
    # Wrap so we get predict_proba for the ensemble
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X_train, y_train)
    OUT_A.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_A / 'svm.joblib')
    print("SVM saved.")
    return model


def train_naive_bayes(X_q_train, y_qtype_train) -> MultinomialNB:
    model = MultinomialNB()
    model.fit(X_q_train, y_qtype_train)
    OUT_A.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_A / 'nb.joblib')
    print("NB saved.")
    return model


def train_random_forest(X_lexical_train, y_difficulty_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_lexical_train, y_difficulty_train)
    OUT_A.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_A / 'rf.joblib')
    print("RF saved.")
    return model


# ---------------------------------------------------------------------------
# 2B — Template-based question generation
# ---------------------------------------------------------------------------

_LOCATION_WORDS  = {'in', 'at', 'from', 'near', 'around', 'outside', 'inside'}
_TIME_WORDS      = {'morning', 'afternoon', 'evening', 'night', 'day', 'week',
                    'month', 'year', 'today', 'yesterday', 'tomorrow', 'monday',
                    'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
_CAUSAL_WORDS    = {'because', 'therefore', 'since', 'thus', 'hence', 'so',
                    'consequently', 'result', 'reason', 'cause'}
_NAME_PATTERN    = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
_YEAR_PATTERN    = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')


def extract_candidate_sentences(article: str, answer: str,
                                vocab: dict, top_k: int = 5) -> list:
    sentences = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 10]
    ans_vec   = one_hot_encode(answer, vocab)
    scored    = []
    for sent in sentences:
        sent_vec = one_hot_encode(sent, vocab)
        score    = compute_cosine_similarity(sent_vec, ans_vec)
        scored.append((sent, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_k]]


def apply_wh_templates(sentence: str) -> list:
    candidates = []
    words = sentence.split()
    lower = sentence.lower()

    # Who — capitalised name pattern
    if _NAME_PATTERN.search(sentence):
        name = _NAME_PATTERN.search(sentence).group()
        q = sentence.replace(name, '___')
        candidates.append(f"Who {q}?")

    # When — year or explicit time words
    if _YEAR_PATTERN.search(sentence) or any(w in lower.split() for w in _TIME_WORDS):
        candidates.append(f"When did {sentence.rstrip('.!?').lower()}?")

    # Where — location indicators followed by capitalised word
    if any(w in lower.split() for w in _LOCATION_WORDS):
        candidates.append(f"Where did {sentence.rstrip('.!?').lower()}?")

    # Why — causal words
    if any(w in lower.split() for w in _CAUSAL_WORDS):
        candidates.append(f"Why {sentence.rstrip('.!?').lower()}?")

    # What — default fallback (always added)
    candidates.append(f"What does the passage say about: {sentence.rstrip('.!?')}?")

    return candidates


def rank_questions(candidates: list, article: str, svm_model) -> str:
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # Simple fluency heuristic: prefer shorter, well-formed questions
    def score(q):
        length_penalty = abs(len(q.split()) - 10)  # ideal ~10 words
        starts_wh      = 1 if q.split()[0].lower() in {'who','what','where','when','why','how'} else 0
        return starts_wh * 10 - length_penalty

    return max(candidates, key=score)


def generate_question(article: str, answer: str, vocab: dict, svm_model) -> str:
    sentences  = extract_candidate_sentences(article, answer, vocab, top_k=5)
    all_cands  = []
    for sent in sentences:
        all_cands.extend(apply_wh_templates(sent))
    return rank_questions(all_cands, article, svm_model)


# ---------------------------------------------------------------------------
# 2C — Unsupervised / Semi-supervised (20 marks — highest priority)
# ---------------------------------------------------------------------------

def train_kmeans(X_train, n_clusters: int = 4) -> KMeans:
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import TruncatedSVD

    # Reduce dimensions for KMeans (sparse → dense 100-dim)
    print("  Reducing dimensions for KMeans (TruncatedSVD 100 components)...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(X_train)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_reduced)

    # Purity
    from collections import Counter
    cluster_purity = []
    for k in range(n_clusters):
        mask    = labels == k
        if mask.sum() == 0:
            continue
        counts  = Counter(labels[mask])
        purity  = counts.most_common(1)[0][1] / mask.sum()
        cluster_purity.append(purity)
    overall_purity = np.mean(cluster_purity)

    sil = silhouette_score(X_reduced, labels, sample_size=5000, random_state=42)

    print(f"  KMeans — Purity: {overall_purity:.4f} | Silhouette: {sil:.4f}")

    # Save model + svd together
    OUT_A.mkdir(parents=True, exist_ok=True)
    joblib.dump({'kmeans': model, 'svd': svd}, OUT_A / 'kmeans.joblib')
    print("  KMeans saved.")
    return model


def train_label_propagation(X_train, y_partial_train) -> LabelPropagation:
    from sklearn.decomposition import TruncatedSVD

    # Subsample to 15k — LabelPropagation is O(n²) in memory/time on full data
    MAX_SAMPLES = 15_000
    if X_train.shape[0] > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_train.shape[0], MAX_SAMPLES, replace=False)
        X_train        = X_train[idx]
        y_partial_train = y_partial_train[idx]
        print(f"  Subsampled to {MAX_SAMPLES} rows for LabelPropagation.")

    print("  Reducing dimensions for LabelPropagation...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(X_train)

    model = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
    print("  Fitting LabelPropagation (this may take a few minutes)...")
    model.fit(X_reduced, y_partial_train)

    # Evaluate on the originally-labeled portion only
    labeled_mask = y_partial_train != -1
    if labeled_mask.sum() > 0:
        preds = model.predict(X_reduced[labeled_mask])
        f1    = f1_score(y_partial_train[labeled_mask], preds, average='macro')
        acc   = accuracy_score(y_partial_train[labeled_mask], preds)
        print(f"  LabelPropagation (labeled subset) — Acc: {acc:.4f} | Macro F1: {f1:.4f}")

    OUT_A.mkdir(parents=True, exist_ok=True)
    joblib.dump({'lp': model, 'svd': svd}, OUT_A / 'label_propagation.joblib')
    print("  LabelPropagation saved.")
    return model


def compare_supervised_vs_semisupervised(results: dict) -> None:
    header = f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}"
    print("\n" + "=" * 50)
    print("Supervised vs Semi-supervised Comparison")
    print("=" * 50)
    print(header)
    print("-" * 50)
    for model_name, metrics in results.items():
        acc = metrics.get('accuracy', float('nan'))
        f1  = metrics.get('f1',       float('nan'))
        print(f"{model_name:<25} {acc:>10.4f} {f1:>10.4f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# 2D — Ensemble
# ---------------------------------------------------------------------------

def soft_vote_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    proba_sum = None
    for m in models:
        p = m.predict_proba(X)
        proba_sum = p if proba_sum is None else proba_sum + p
    return np.argmax(proba_sum / len(models), axis=1)


def hard_vote_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    preds = np.stack([m.predict(X) for m in models], axis=1)
    return np.apply_along_axis(
        lambda row: np.bincount(row).argmax(), axis=1, arr=preds
    )


def train_stacking_ensemble(base_models: list, X_val, y_val, X_test) -> LogisticRegression:
    meta_X_val  = np.column_stack([m.predict_proba(X_val)  for m in base_models])
    meta_X_test = np.column_stack([m.predict_proba(X_test) for m in base_models])

    meta = LogisticRegression(max_iter=500, random_state=42)
    meta.fit(meta_X_val, y_val)
    return meta


def save_best_ensemble(ensemble, path: str = 'models/model_a/traditional/ensemble.joblib') -> None:
    OUT_A.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, path)
    print(f"Ensemble saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- Load features ---
    print("Loading features...")
    X_verify_train, y_verify_train   = joblib.load(PROC / 'X_verify_train.joblib')
    X_verify_val,   y_verify_val     = joblib.load(PROC / 'X_verify_val.joblib')
    X_lexical_train, y_lexical_train = joblib.load(PROC / 'X_lexical_train.joblib')
    X_qbow_train,    y_qtype_train   = joblib.load(PROC / 'X_qbow_train.joblib')
    y_diff_train                     = joblib.load(PROC / 'y_difficulty_train.joblib')

    # y_diff_train is per-question; X_lexical_train is per-option (4x) — align them
    y_diff_train = np.repeat(y_diff_train, 4)

    # --- 2A: Supervised (skip if already saved) ---
    print("\n=== Phase 2A: Supervised Models ===")

    if (OUT_A / 'lr.joblib').exists():
        print("LR already trained — loading.")
        lr = joblib.load(OUT_A / 'lr.joblib')
    else:
        print("Training Logistic Regression...")
        lr = train_logistic_regression(X_verify_train, y_verify_train)

    if (OUT_A / 'svm.joblib').exists():
        print("SVM already trained — loading.")
        svm = joblib.load(OUT_A / 'svm.joblib')
    else:
        print("Training SVM...")
        svm = train_svm(X_verify_train, y_verify_train)

    if (OUT_A / 'nb.joblib').exists():
        print("NB already trained — loading.")
        nb = joblib.load(OUT_A / 'nb.joblib')
    else:
        print("Training Naive Bayes (question type)...")
        nb = train_naive_bayes(X_qbow_train, y_qtype_train)

    if (OUT_A / 'rf.joblib').exists():
        print("RF already trained — loading.")
        rf = joblib.load(OUT_A / 'rf.joblib')
    else:
        print("Training Random Forest (difficulty)...")
        rf = train_random_forest(X_lexical_train, y_diff_train)

    # --- 2C: Unsupervised / Semi-supervised ---
    print("\n=== Phase 2C: Unsupervised / Semi-supervised ===")

    if (OUT_A / 'kmeans.joblib').exists():
        print("KMeans already trained — skipping.")
    else:
        print("Training KMeans...")
        train_kmeans(X_verify_train, n_clusters=4)

    if (OUT_A / 'label_propagation.joblib').exists():
        print("LabelPropagation already trained — skipping.")
    else:
        print("Training LabelPropagation (10% labeled)...")
        rng = np.random.default_rng(42)
        y_partial = y_verify_train.copy()
        unlabeled_mask = rng.random(len(y_partial)) > 0.10
        y_partial[unlabeled_mask] = -1
        train_label_propagation(X_verify_train, y_partial)

    # Compare supervised vs semi-supervised on val set
    lr_preds  = lr.predict(X_verify_val)
    svm_preds = svm.predict(X_verify_val)

    compare_supervised_vs_semisupervised({
        'Logistic Regression': {
            'accuracy': accuracy_score(y_verify_val, lr_preds),
            'f1':       f1_score(y_verify_val, lr_preds, average='macro'),
        },
        'SVM (calibrated)': {
            'accuracy': accuracy_score(y_verify_val, svm_preds),
            'f1':       f1_score(y_verify_val, svm_preds, average='macro'),
        },
    })

    # --- 2D: Ensemble ---
    print("\n=== Phase 2D: Ensemble ===")
    if (OUT_A / 'ensemble.joblib').exists():
        print("Ensemble already trained — skipping.")
    else:
        print("Training stacking ensemble...")
        ensemble = train_stacking_ensemble([lr, svm], X_verify_val, y_verify_val,
                                           X_verify_val)
        save_best_ensemble(ensemble)

    print("\nmodel_a_train.py complete.")
