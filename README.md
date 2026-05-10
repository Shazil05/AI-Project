# RACE Comprehension AI: Intelligent Quiz Generation System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)

An end-to-end AI system that transforms reading passages into interactive quizzes. By leveraging machine learning models trained on the **RACE** (ReAding Comprehension dataset from Examinations) dataset, this system automates the generation of questions, correct answers, distractors, and helpful hints.

---

## 🚀 Features

- **Automated Question Generation**: Extracts key information from articles to formulate relevant questions.
- **Answer Verification (Model A)**: Employs an Ensemble of SVM and Logistic Regression to verify the correctness and confidence of answer options.
- **Distractor & Hint Generation (Model B)**: Uses NLP techniques (TF-IDF, Cosine Similarity) and Ranking models to create challenging distractors and context-aware hints.
- **Interactive Dashboards**:
  - **Student View**: A clean interface for taking quizzes and receiving instant feedback.
  - **Admin View**: Detailed model performance metrics, latency tracking, and dataset analysis.
- **High Performance**: Optimized inference pipeline with sub-100ms latency for individual components.

---

## 🛠️ Technology Stack

- **Core Logic**: Python 3.x
- **Machine Learning**: Scikit-learn (SVM, Logistic Regression, Random Forest, Stacking Ensemble)
- **Data Processing**: Pandas, NumPy, NLTK
- **Vectorization**: Custom One-Hot Encoding & TF-IDF
- **UI Frameworks**: Streamlit (Front-end), FastAPI (Backend API)
- **Deployment**: Uvicorn

---

## 📂 Project Structure

```text
AI-Project/
├── data/               # Raw and processed RACE datasets
├── models/             # Trained model binaries (.joblib)
├── notebooks/          # Exploratory Data Analysis and prototyping
├── src/                # Core engine
│   ├── preprocessing.py# Feature engineering and encoding
│   ├── model_a_train.py# Training pipeline for answer verification
│   ├── model_b_train.py# Training for distractors/hints
│   └── inference.py    # Unified prediction engine
├── ui/                 # Application interfaces
│   ├── streamlit_app.py# Main interactive dashboard
│   └── app.py          # FastAPI server
├── tests/              # Unit and integration tests
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AI-Project.git
cd AI-Project
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data Preparation
Ensure the RACE dataset is placed in `data/raw/`. Run the preprocessing script to generate necessary files:
```bash
python src/preprocessing.py
```

---

## 🖥️ Usage

### Running the Interactive UI (Streamlit)
The most user-friendly way to interact with the system:
```bash
streamlit run ui/streamlit_app.py
```

### Starting the Backend API (FastAPI)
For integration with other services:
```bash
python -m uvicorn ui.app:app --reload --port 8000
```

### Model Training
To retrain the verification and generation models:
```bash
python src/model_a_train.py
python src/model_b_train.py
```

---

## 📊 Model Performance

| Model | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| Logistic Regression | 0.81 | 0.79 |
| Support Vector Machine (SVM) | 0.84 | 0.82 |
| **Stacking Ensemble** | **0.86** | **0.84** |

---

## 👥 Contributors

- **Abdul Mohaimin** - i230652 | [i230652@isb.nu.edu.pk]
- **Shazil Rehman**  - i230095 | [i230095@isb.nu.edu.pk]

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
