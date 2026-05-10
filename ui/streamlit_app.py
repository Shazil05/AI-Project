import sys
import random
import pandas as pd
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import load_all_models, run_full_pipeline, model_a_predict, model_b_predict, get_model_a_metrics, get_model_b_metrics

st.set_page_config(page_title="RACE Quiz System", page_icon="📖", layout="centered")

@st.cache_resource
def get_models():
    load_all_models()

get_models()

# Session state defaults
for key, val in [
    ('quiz', None), ('article', ''), ('selected', ''), ('result', None),
    ('hints_revealed', 1), ('show_answer', False), ('session_log', []),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Article Input", "❓ Quiz", "💡 Hints", "📊 Dashboard"])


# ---------------------------------------------------------------------------
# Tab 1 — Article Input
# ---------------------------------------------------------------------------
with tab1:
    st.title("Reading Quiz Generator")
    st.caption("Paste a passage and generate a multiple-choice quiz, or load a random sample from RACE.")

    article = st.text_area("Reading Passage", value=st.session_state.article,
                           height=200, placeholder="Paste your reading passage here...")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Generate Quiz", type="primary", use_container_width=True):
            if not article.strip():
                st.error("Please enter a passage first.")
            else:
                with st.spinner("Generating quiz..."):
                    try:
                        quiz = run_full_pipeline(article)
                        st.session_state.quiz     = quiz
                        st.session_state.article  = article
                        st.session_state.selected = ''
                        st.session_state.result   = None
                        st.session_state.hints_revealed = 1
                        st.session_state.show_answer    = False
                        st.success("Quiz generated! Switch to the Quiz tab.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        if st.button("Load Random Sample", use_container_width=True):
            with st.spinner("Loading sample..."):
                try:
                    df = pd.read_csv('data/raw/train.csv')
                    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
                    df.dropna(subset=['A', 'B', 'C', 'D'], inplace=True)
                    row         = df.sample(1, random_state=random.randint(0, 99999)).iloc[0]
                    article_txt = str(row['article'])
                    question    = str(row['question'])
                    answer_text = str(row[row['answer']])

                    b = model_b_predict(article_txt, question, answer_text)
                    clean_distractors = [d for d in b['distractors'] if d.strip().lower() != answer_text.strip().lower()]
                    opts = [answer_text] + clean_distractors[:3]
                    while len(opts) < 4:
                        opts.append('None of the above')
                    random.shuffle(opts)
                    labels  = ['A', 'B', 'C', 'D']
                    options = {labels[i]: opts[i] for i in range(4)}
                    correct = labels[opts.index(answer_text)]

                    quiz = {
                        'article': article_txt, 'question': question,
                        'options': options, 'correct_answer': correct,
                        'hints': b['hints'],
                    }
                    st.session_state.quiz     = quiz
                    st.session_state.article  = article_txt
                    st.session_state.selected = ''
                    st.session_state.result   = None
                    st.session_state.hints_revealed = 1
                    st.session_state.show_answer    = False
                    st.success("Sample loaded! Switch to the Quiz tab.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Tab 2 — Quiz
# ---------------------------------------------------------------------------
with tab2:
    st.title("Quiz")

    if st.session_state.quiz is None:
        st.info("No quiz yet. Go to the Article Input tab to generate one.")
    else:
        quiz = st.session_state.quiz
        st.subheader(quiz['question'])

        options = quiz['options']
        option_labels = [f"{k}. {v}" for k, v in options.items()]
        choice = st.radio("Select your answer:", option_labels, index=None, key="radio_choice")

        if choice:
            st.session_state.selected = choice[0]  # 'A', 'B', 'C', or 'D'

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Check Answer", type="primary", use_container_width=True):
                if not st.session_state.selected:
                    st.warning("Please select an option first.")
                else:
                    with st.spinner("Checking..."):
                        selected_text = options[st.session_state.selected]
                        result = model_a_predict(
                            st.session_state.article,
                            quiz['question'],
                            selected_text,
                        )
                        result['label'] = st.session_state.selected
                        st.session_state.result = result
                        st.session_state.session_log.append({
                            'question':   quiz['question'],
                            'selected':   st.session_state.selected,
                            'is_correct': result['is_correct'],
                            'confidence': result['confidence'],
                        })

        with col2:
            if st.button("Need a Hint?", use_container_width=True):
                st.info("Switch to the Hints tab.")

        if st.session_state.result:
            r          = st.session_state.result
            is_correct = r['label'] == quiz['correct_answer']
            if is_correct:
                st.success(f"Correct! Confidence: {r['confidence']*100:.1f}%")
            else:
                correct_key = quiz['correct_answer']
                st.error(f"Incorrect. Correct answer: {correct_key} — {options[correct_key]}")


# ---------------------------------------------------------------------------
# Tab 3 — Hints
# ---------------------------------------------------------------------------
with tab3:
    st.title("Hints")

    if st.session_state.quiz is None:
        st.info("No quiz yet. Generate one from the Article Input tab.")
    else:
        hints   = st.session_state.quiz.get('hints', [])
        options = st.session_state.quiz.get('options', {})
        correct = st.session_state.quiz.get('correct_answer', '')

        st.caption("Hints progress from general clues to more specific ones.")

        for i in range(st.session_state.hints_revealed):
            if i < len(hints):
                with st.expander(f"Hint {i+1}", expanded=True):
                    st.write(hints[i])

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.hints_revealed < len(hints):
                if st.button("Show Next Hint", use_container_width=True):
                    st.session_state.hints_revealed += 1
                    st.rerun()

        with col2:
            if st.session_state.hints_revealed >= len(hints) and not st.session_state.show_answer:
                if st.button("Reveal Answer", use_container_width=True):
                    st.session_state.show_answer = True
                    st.rerun()

        if st.session_state.show_answer:
            st.success(f"Answer: **{correct}** — {options.get(correct, '')}")


# ---------------------------------------------------------------------------
# Tab 4 — Dashboard
# ---------------------------------------------------------------------------
with tab4:
    st.title("Dashboard")

    with st.spinner("Loading metrics..."):
        try:
            a = get_model_a_metrics()
            b = get_model_b_metrics()

            st.subheader("Model A — Answer Verification")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  a['accuracy'])
            c2.metric("Macro F1",  a['macro_f1'])
            c3.metric("Precision", a['precision'])
            c4.metric("Recall",    a['recall'])

            st.subheader("Model Comparison")
            mc = a['model_comparison']
            comparison_df = pd.DataFrame({
                'Model':    ['LR', 'SVM', 'Ensemble'],
                'Accuracy': [mc['lr']['accuracy'], mc['svm']['accuracy'], mc['ensemble']['accuracy']],
                'Macro F1': [mc['lr']['f1'],       mc['svm']['f1'],       mc['ensemble']['f1']],
            })
            st.bar_chart(comparison_df.set_index('Model'))

            st.subheader("Confusion Matrix (Ensemble)")
            cm = a['confusion_matrix']
            cm_df = pd.DataFrame(cm,
                index=['Actual: Incorrect', 'Actual: Correct'],
                columns=['Pred: Incorrect', 'Pred: Correct'])
            st.dataframe(cm_df, use_container_width=True)

            st.subheader("Model B — Distractor & Hint Generation")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Dist. Precision", b['distractor_precision'])
            b2.metric("Dist. Recall",    b['distractor_recall'])
            b3.metric("Dist. F1",        b['distractor_f1'])
            b4.metric("Dist. Accuracy",  b['distractor_accuracy'])
            b5.metric("Hint P@3",        b['hint_precision_at_3'])

        except Exception as e:
            st.error(f"Failed to load metrics: {e}")

    st.subheader("Session Log")
    if st.session_state.session_log:
        log_df = pd.DataFrame(st.session_state.session_log)
        st.dataframe(log_df, use_container_width=True)

        csv = log_df.to_csv(index=False)
        st.download_button("Export Session CSV", csv, "session_log.csv", "text/csv")
    else:
        st.caption("No answers checked yet. Answer some questions to see your session log.")
