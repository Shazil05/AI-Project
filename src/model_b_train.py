import sys
import numpy as np
import joblib
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression

from src.preprocessing import clean_text, one_hot_encode, compute_cosine_similarity

PROC  = Path('data/processed')
OUT_B = Path('models/model_b/traditional')


# ---------------------------------------------------------------------------
# 3A — Distractor Generation
# ---------------------------------------------------------------------------

def extract_distractor_candidates(article: str, answer: str) -> list:
    words      = clean_text(article).split()
    answer_set = set(clean_text(answer).split())
    freq       = Counter(words)

    # Build 1-gram and 2-gram candidates from article
    candidates = {}
    for i, w in enumerate(words):
        if w in answer_set or len(w) < 3:
            continue
        candidates[w] = candidates.get(w, 0) + freq[w]

    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        bigram_set = set(bigram.split())
        if bigram_set & answer_set:
            continue
        candidates[bigram] = candidates.get(bigram, 0) + 1

    # Return as (candidate, frequency) sorted by frequency desc
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)


def compute_distractor_features(candidate: str, answer: str,
                                article: str, vocab: dict) -> np.ndarray:
    cand_vec   = one_hot_encode(candidate, vocab)
    ans_vec    = one_hot_encode(answer,    vocab)
    cos_sim    = compute_cosine_similarity(cand_vec, ans_vec)

    # Character-level match score
    cand_chars = set(candidate.replace(' ', ''))
    ans_chars  = set(answer.replace(' ', ''))
    max_len    = max(len(cand_chars), len(ans_chars), 1)
    char_score = len(cand_chars & ans_chars) / max_len

    # Passage frequency (normalised by article length)
    art_words  = clean_text(article).split()
    art_len    = max(len(art_words), 1)
    cand_words = clean_text(candidate).split()
    freq       = sum(1 for w in art_words if w in set(cand_words))
    norm_freq  = freq / art_len

    return np.array([cos_sim, char_score, norm_freq], dtype=np.float32)


def train_distractor_ranker(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        random_state=42,
    )
    model.fit(X_train, y_train)
    OUT_B.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_B / 'distractor_ranker.joblib')
    print("Distractor ranker saved.")
    return model


def apply_diversity_penalty(candidates: list) -> list:
    selected = []
    for cand, score in candidates:
        cand_words = set(clean_text(cand).split())
        too_similar = False
        for kept in selected:
            kept_words = set(clean_text(kept).split())
            union      = cand_words | kept_words
            if not union:
                continue
            overlap = len(cand_words & kept_words) / len(union)
            if overlap > 0.50:
                too_similar = True
                break
        if not too_similar:
            selected.append(cand)
        if len(selected) == 3:
            break
    return selected


def select_top_distractors(article: str, answer: str,
                            ranker, vocab: dict) -> list:
    candidates = extract_distractor_candidates(article, answer)
    if not candidates:
        return ["Option 1", "Option 2", "Option 3"]

    scored = []
    for cand, _ in candidates[:50]:  # score top-50 frequent candidates
        feat  = compute_distractor_features(cand, answer, article, vocab)
        score = ranker.predict_proba([feat])[0][1]
        scored.append((cand, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    distractors = apply_diversity_penalty(scored)

    # Pad to exactly 3 if diversity filter removed too many
    fallbacks = [c for c, _ in candidates if c not in distractors]
    while len(distractors) < 3 and fallbacks:
        distractors.append(fallbacks.pop(0))

    return distractors[:3]


# ---------------------------------------------------------------------------
# 3B — Hint Generation
# ---------------------------------------------------------------------------

def score_sentences_for_hints(article: str, question: str,
                               vocab: dict) -> list:
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 10]
    if not sentences:
        return []

    q_words   = set(clean_text(question).split())
    total     = len(sentences)
    scored    = []

    for pos, sent in enumerate(sentences):
        sent_words = set(clean_text(sent).split())

        # keyword overlap (normalised)
        overlap = len(sent_words & q_words) / max(len(q_words), 1)

        # sentence position (0 = first, 1 = last)
        position = pos / max(total - 1, 1)

        # sentence length (normalised by 50 words as rough max)
        length = min(len(sent_words) / 50.0, 1.0)

        feat  = np.array([overlap, position, length], dtype=np.float32)
        scored.append((sent, feat))

    return scored


def train_hint_scorer(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        random_state=42,
    )
    model.fit(X_train, y_train)
    OUT_B.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_B / 'hint_scorer.joblib')
    print("Hint scorer saved.")
    return model


def generate_hints(article: str, question: str, answer: str,
                   hint_scorer, vocab: dict) -> list:
    scored = score_sentences_for_hints(article, question, vocab)
    if not scored:
        return [article[:100], article[100:200], article[200:300]]

    ans_words = set(clean_text(answer).split())

    # Score each sentence; exclude the one that contains the answer directly
    results = []
    for sent, feat in scored:
        sent_words = set(clean_text(sent).split())
        contains_answer = ans_words and ans_words.issubset(sent_words)
        score = hint_scorer.predict_proba([feat])[0][1]
        results.append((sent, score, contains_answer))

    # Filter out the answer sentence itself
    filtered = [(s, sc) for s, sc, is_ans in results if not is_ans]

    # Fallback: if all sentences contain the answer, use all
    if not filtered:
        filtered = [(s, sc) for s, sc, _ in results]

    filtered.sort(key=lambda x: x[1])  # ascending: hint[0] least relevant

    # Return exactly 3: low, mid, high relevance
    n = len(filtered)
    if n >= 3:
        hints = [
            filtered[0][0],
            filtered[n // 2][0],
            filtered[-1][0],
        ]
    elif n == 2:
        hints = [filtered[0][0], filtered[1][0], filtered[1][0]]
    else:
        hints = [filtered[0][0]] * 3

    return hints


# ---------------------------------------------------------------------------
# Build training data for distractor ranker & hint scorer from processed features
# ---------------------------------------------------------------------------

def _build_distractor_training_data(df, vocab, max_rows=5000):
    import pandas as pd
    X, y = [], []
    count = 0
    for _, row in df.iterrows():
        if count >= max_rows:
            break
        article = str(row['article'])
        answer  = str(row[row['answer']])  # actual answer text

        candidates = extract_distractor_candidates(article, answer)
        actual_distractors = {
            clean_text(str(row[opt]))
            for opt in ['A', 'B', 'C', 'D']
            if opt != row['answer']
        }

        for cand, _ in candidates[:20]:
            feat  = compute_distractor_features(cand, answer, article, vocab)
            label = 1 if clean_text(cand) in actual_distractors else 0
            X.append(feat)
            y.append(label)
        count += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def _build_hint_training_data(df, vocab, max_rows=5000):
    import re
    X, y = [], []
    count = 0
    for _, row in df.iterrows():
        if count >= max_rows:
            break
        article  = str(row['article'])
        question = str(row['question'])
        answer   = str(row[row['answer']])
        ans_words = set(clean_text(answer).split())

        scored = score_sentences_for_hints(article, question, vocab)
        for sent, feat in scored:
            sent_words = set(clean_text(sent).split())
            label = 1 if (ans_words and ans_words.issubset(sent_words)) else 0
            X.append(feat)
            y.append(label)
        count += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import pandas as pd

    print("Loading vocab and raw data...")
    vocab    = joblib.load(PROC / 'vocab.joblib')
    df_all   = pd.read_csv('data/raw/train.csv')
    df_all.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df_all.dropna(subset=['A', 'B', 'C', 'D'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    df_train = df_all.sample(frac=0.80, random_state=42).reset_index(drop=True)

    # --- 3A: Distractor ranker ---
    print("\n=== Phase 3A: Distractor Ranker ===")
    if (OUT_B / 'distractor_ranker.joblib').exists():
        print("Distractor ranker already trained — loading.")
        ranker = joblib.load(OUT_B / 'distractor_ranker.joblib')
    else:
        print("Building distractor training data (5000 rows)...")
        X_dist, y_dist = _build_distractor_training_data(df_train, vocab, max_rows=5000)
        print(f"  Distractor samples: {len(X_dist)}, positives: {y_dist.sum()}")
        ranker = train_distractor_ranker(X_dist, y_dist)

    # Quick demo
    sample = df_train.iloc[0]
    answer_text = str(sample[sample['answer']])
    distractors = select_top_distractors(str(sample['article']), answer_text, ranker, vocab)
    print(f"\nSample answer:      {answer_text}")
    print(f"Generated distractors: {distractors}")

    # --- 3B: Hint scorer ---
    print("\n=== Phase 3B: Hint Scorer ===")
    if (OUT_B / 'hint_scorer.joblib').exists():
        print("Hint scorer already trained — loading.")
        hint_scorer = joblib.load(OUT_B / 'hint_scorer.joblib')
    else:
        print("Building hint training data (5000 rows)...")
        X_hint, y_hint = _build_hint_training_data(df_train, vocab, max_rows=5000)
        print(f"  Hint samples: {len(X_hint)}, positives: {y_hint.sum()}")
        hint_scorer = train_hint_scorer(X_hint, y_hint)

    # Quick demo
    hints = generate_hints(
        str(sample['article']),
        str(sample['question']),
        answer_text,
        hint_scorer,
        vocab,
    )
    print(f"\nSample question: {sample['question']}")
    print(f"Hint 1 (vague):  {hints[0][:80]}...")
    print(f"Hint 2 (mid):    {hints[1][:80]}...")
    print(f"Hint 3 (close):  {hints[2][:80]}...")

    print("\nmodel_b_train.py complete.")
