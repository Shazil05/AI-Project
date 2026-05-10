import re
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from pathlib import Path
from scipy.sparse import lil_matrix, csr_matrix, hstack


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(texts: list, max_vocab: int = 5000) -> dict:
    counter = Counter()
    for text in texts:
        counter.update(clean_text(text).split())
    vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(max_vocab))}
    return vocab


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def one_hot_encode(text: str, vocab: dict) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for word in clean_text(text).split():
        if word in vocab:
            vec[vocab[word]] = 1.0
    return vec


def encode_row(article: str, question: str, option: str, vocab: dict) -> np.ndarray:
    return np.concatenate([
        one_hot_encode(article, vocab),
        one_hot_encode(question, vocab),
        one_hot_encode(option, vocab),
    ])


# ---------------------------------------------------------------------------
# Cosine similarity (pure NumPy)
# ---------------------------------------------------------------------------

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Lexical features (5-dim handcrafted vector)
# ---------------------------------------------------------------------------

def extract_lexical_features(article: str, question: str, option: str) -> np.ndarray:
    art_words = clean_text(article).split()
    opt_words = set(clean_text(option).split())
    q_words   = clean_text(question).split()

    overlap  = sum(1 for w in art_words if w in opt_words)
    opt_len  = len(opt_words)
    q_len    = len(q_words)
    art_len  = len(art_words)

    first_pos = 0.0
    if art_words and opt_words:
        for i, w in enumerate(art_words):
            if w in opt_words:
                first_pos = i / len(art_words)
                break

    return np.array([overlap, opt_len, q_len, art_len, first_pos], dtype=np.float32)


# ---------------------------------------------------------------------------
# Question-type label (for Naive Bayes)
# ---------------------------------------------------------------------------

_Q_TYPE_MAP = {
    'what': 0, 'which': 1, 'who': 2, 'where': 3,
    'when': 4, 'why': 5, 'how': 6, 'according': 7,
}

def _question_type_label(question: str) -> int:
    first = clean_text(question).split()
    if not first:
        return 8
    return _Q_TYPE_MAP.get(first[0], 8)


# ---------------------------------------------------------------------------
# Difficulty proxy label (for Random Forest)
# ---------------------------------------------------------------------------

def _difficulty_label(question: str) -> int:
    length = len(clean_text(question).split())
    if length <= 7:
        return 0
    elif length <= 12:
        return 1
    else:
        return 2


# ---------------------------------------------------------------------------
# Sparse encoding helpers
# ---------------------------------------------------------------------------

def _fill_sparse_row(mat: lil_matrix, row_idx: int, text: str, vocab: dict) -> None:
    for word in clean_text(text).split():
        if word in vocab:
            mat[row_idx, vocab[word]] = 1.0


def _encode_split_sparse(df: pd.DataFrame, vocab: dict):
    V = len(vocab)
    n_opts = len(df) * 4

    # X_verify: sparse (n_opts, 3*V) — article + question + option one-hot concat
    art_mat = lil_matrix((n_opts, V), dtype=np.float32)
    q_mat   = lil_matrix((n_opts, V), dtype=np.float32)
    opt_mat = lil_matrix((n_opts, V), dtype=np.float32)

    # X_qbow: sparse (n_rows, V) — question only
    qbow_mat = lil_matrix((len(df), V), dtype=np.float32)

    y_verify     = np.empty(n_opts, dtype=np.int32)
    X_lexical    = np.empty((n_opts, 5), dtype=np.float32)
    y_lexical    = np.empty(n_opts, dtype=np.int32)
    y_qtype      = np.empty(len(df), dtype=np.int32)
    y_difficulty = np.empty(len(df), dtype=np.int32)

    for row_i, (_, row) in enumerate(df.iterrows()):
        article  = str(row['article'])
        question = str(row['question'])
        answer   = row['answer']

        # Question-level features (one per row)
        _fill_sparse_row(qbow_mat, row_i, question, vocab)
        y_qtype[row_i]      = _question_type_label(question)
        y_difficulty[row_i] = _difficulty_label(question)

        for opt_i, opt_label in enumerate(['A', 'B', 'C', 'D']):
            idx    = row_i * 4 + opt_i
            option = str(row[opt_label])

            _fill_sparse_row(art_mat, idx, article,  vocab)
            _fill_sparse_row(q_mat,   idx, question, vocab)
            _fill_sparse_row(opt_mat, idx, option,   vocab)

            y_verify[idx]        = 1 if opt_label == answer else 0
            X_lexical[idx]       = extract_lexical_features(article, question, option)
            y_lexical[idx]       = y_verify[idx]

        if (row_i + 1) % 5000 == 0:
            print(f"    {row_i + 1}/{len(df)} rows done")

    X_verify = hstack([art_mat, q_mat, opt_mat]).tocsr()
    X_qbow   = qbow_mat.tocsr()

    return X_verify, y_verify, X_lexical, y_lexical, X_qbow, y_qtype, y_difficulty


# ---------------------------------------------------------------------------
# Persist helpers
# ---------------------------------------------------------------------------

def save_features(X, y, path: str) -> None:
    joblib.dump((X, y), path)


def load_features(path: str):
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_all() -> None:
    raw_dir  = Path('data/raw')
    proc_dir = Path('data/processed')
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV...")
    df_all = pd.read_csv(raw_dir / 'train.csv')
    df_all.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df_all.dropna(subset=['A', 'B', 'C', 'D'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    df_train = df_all.sample(frac=0.80, random_state=42)
    remaining = df_all.drop(df_train.index)
    df_val  = remaining.sample(frac=0.50, random_state=42)
    df_test = remaining.drop(df_val.index)

    print(f"Splits — Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    print("Building vocabulary from training articles...")
    vocab = build_vocabulary(df_train['article'].tolist(), max_vocab=5000)
    joblib.dump(vocab, proc_dir / 'vocab.joblib')
    print(f"Vocabulary size: {len(vocab)}")

    for split_name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        print(f"\nEncoding {split_name} ({len(df)} rows × 4 options)...")
        df = df.reset_index(drop=True)

        X_verify, y_verify, X_lexical, y_lexical, X_qbow, y_qtype, y_difficulty = \
            _encode_split_sparse(df, vocab)

        print(f"  X_verify (sparse): {X_verify.shape}, nnz={X_verify.nnz:,}")
        print(f"  X_lexical (dense): {X_lexical.shape}")
        print(f"  X_qbow (sparse):   {X_qbow.shape}")

        joblib.dump((X_verify,  y_verify),  proc_dir / f'X_verify_{split_name}.joblib')
        joblib.dump((X_lexical, y_lexical), proc_dir / f'X_lexical_{split_name}.joblib')
        joblib.dump((X_qbow,    y_qtype),   proc_dir / f'X_qbow_{split_name}.joblib')
        joblib.dump(y_difficulty,            proc_dir / f'y_difficulty_{split_name}.joblib')
        print(f"  Saved to data/processed/")

    print("\nPreprocessing complete.")


if __name__ == '__main__':
    preprocess_all()
