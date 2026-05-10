import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import (
    load_all_models,
    model_a_predict,
    model_b_predict,
    run_full_pipeline,
    get_model_a_metrics,
    get_model_b_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ARTICLE = (
    "Marie Curie was a Polish-born physicist and chemist who conducted pioneering "
    "research on radioactivity. She was the first woman to win a Nobel Prize, and "
    "the only person to win the Nobel Prize in two different sciences. She was born "
    "in Warsaw in 1867 and moved to Paris to study at the University of Paris."
)
QUESTION = "Where was Marie Curie born?"
OPTION_CORRECT = "Warsaw"
OPTION_WRONG   = "Paris"


@pytest.fixture(scope="session", autouse=True)
def models():
    load_all_models()


# ---------------------------------------------------------------------------
# model_a_predict
# ---------------------------------------------------------------------------

class TestModelAPredict:

    def test_returns_required_keys(self):
        result = model_a_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert {'is_correct', 'confidence', 'latency_ms'}.issubset(result.keys())

    def test_is_correct_is_bool(self):
        result = model_a_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert isinstance(result['is_correct'], bool)

    def test_confidence_in_range(self):
        result = model_a_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert 0.0 <= result['confidence'] <= 1.0

    def test_latency_under_10s(self):
        result = model_a_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert result['latency_ms'] < 10_000

    def test_empty_option_does_not_crash(self):
        result = model_a_predict(ARTICLE, QUESTION, "")
        assert 'is_correct' in result

    def test_long_article_does_not_crash(self):
        long_article = ARTICLE * 20
        result = model_a_predict(long_article, QUESTION, OPTION_CORRECT)
        assert 'is_correct' in result


# ---------------------------------------------------------------------------
# model_b_predict
# ---------------------------------------------------------------------------

class TestModelBPredict:

    def test_returns_required_keys(self):
        result = model_b_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert {'distractors', 'hints', 'latency_ms'}.issubset(result.keys())

    def test_exactly_3_distractors(self):
        result = model_b_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert len(result['distractors']) == 3

    def test_exactly_3_hints(self):
        result = model_b_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert len(result['hints']) == 3

    def test_distractors_are_strings(self):
        result = model_b_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert all(isinstance(d, str) for d in result['distractors'])

    def test_hints_are_strings(self):
        result = model_b_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert all(isinstance(h, str) for h in result['hints'])

    def test_distractors_not_equal_to_answer(self):
        result = model_b_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        for d in result['distractors']:
            assert d.strip().lower() != OPTION_CORRECT.strip().lower()

    def test_latency_under_10s(self):
        result = model_b_predict(ARTICLE, QUESTION, OPTION_CORRECT)
        assert result['latency_ms'] < 10_000


# ---------------------------------------------------------------------------
# run_full_pipeline
# ---------------------------------------------------------------------------

class TestRunFullPipeline:

    def test_returns_required_keys(self):
        result = run_full_pipeline(ARTICLE)
        required = {'article', 'question', 'options', 'correct_answer', 'hints',
                    'model_a_latency_ms', 'model_b_latency_ms'}
        assert required.issubset(result.keys())

    def test_options_has_4_entries(self):
        result = run_full_pipeline(ARTICLE)
        assert set(result['options'].keys()) == {'A', 'B', 'C', 'D'}

    def test_correct_answer_is_valid_label(self):
        result = run_full_pipeline(ARTICLE)
        assert result['correct_answer'] in {'A', 'B', 'C', 'D'}

    def test_correct_answer_in_options(self):
        result = run_full_pipeline(ARTICLE)
        assert result['correct_answer'] in result['options']

    def test_exactly_3_hints(self):
        result = run_full_pipeline(ARTICLE)
        assert len(result['hints']) == 3

    def test_article_preserved(self):
        result = run_full_pipeline(ARTICLE)
        assert result['article'] == ARTICLE

    def test_total_latency_under_10s(self):
        result = run_full_pipeline(ARTICLE)
        total = result['model_a_latency_ms'] + result['model_b_latency_ms']
        assert total < 10_000

    def test_options_are_strings(self):
        result = run_full_pipeline(ARTICLE)
        assert all(isinstance(v, str) for v in result['options'].values())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_model_a_metrics_keys(self):
        m = get_model_a_metrics()
        required = {'accuracy', 'macro_f1', 'precision', 'recall',
                    'confusion_matrix', 'model_comparison'}
        assert required.issubset(m.keys())

    def test_model_a_accuracy_in_range(self):
        m = get_model_a_metrics()
        assert 0.0 <= m['accuracy'] <= 1.0

    def test_model_a_confusion_matrix_shape(self):
        m = get_model_a_metrics()
        cm = m['confusion_matrix']
        assert len(cm) == 2 and all(len(row) == 2 for row in cm)

    def test_model_a_comparison_has_all_models(self):
        m = get_model_a_metrics()
        assert set(m['model_comparison'].keys()) == {'lr', 'svm', 'ensemble'}

    def test_model_b_metrics_keys(self):
        m = get_model_b_metrics()
        required = {'distractor_precision', 'distractor_recall', 'distractor_f1',
                    'distractor_accuracy', 'hint_precision_at_3'}
        assert required.issubset(m.keys())

    def test_model_b_metrics_in_range(self):
        m = get_model_b_metrics()
        for v in m.values():
            assert 0.0 <= v <= 1.0