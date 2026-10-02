# 🏥 HEALTHCARE WELLNESS: Explainable, Fair, and Calibrated Multi-Source Mental Health Risk Prediction for Healthcare Workers

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.45.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini-RAG%20Enabled-8E44AD.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end responsible Machine Learning and Retrieval-Augmented Generation (RAG) Generative AI framework for multi-source mental health risk prediction (Stress, Anxiety, Depression) among healthcare personnel. Built as part of the Capstone Project at **Dayananda Sagar University (DSU), 2025–2026**.

---

## 🌟 Key Innovations & Features

- **Multi-Source Data Harmonization**: Seamlessly unifies 4 independent dataset modalities—physiological biometric telemetry (**Stress-Lysis**), occupational shift workload (**Workplace Survey**), psychometric subscales (**DASS-42**), and department organizational metrics (**Healthcare Workforce**).
- **5-Algorithm Model Comparison**: Systematic evaluation across **Multilayer Perceptron (MLP)**, **XGBoost**, **LightGBM**, **Random Forest**, and **Decision Tree**.
- **High Performance**:
  - ⚡ **Stress Prediction**: **97.9% Accuracy** (F1: 97.8%) with MLP Neural Net.
  - 😰 **Anxiety Prediction**: **95.2% Accuracy** (F1: 95.1%) with MLP Neural Net.
  - 😔 **Depression Prediction**: **94.5% Accuracy** (F1: 94.5%) with MLP Neural Net.
- **Probability Calibration**: Post-hoc **Platt Scaling** & Isotonic Regression reduce Expected Calibration Error (ECE) to **0.028** (Brier Score: 0.082).
- **Fairlearn Bias Mitigation**: Audits and mitigates demographic disparity across hospital departments (ER, ICU, OPD, IPD, OBG), dropping parity difference from 0.14 to **<0.05**.
- **Dual Explainability (SHAP & DiCE)**: Global and local SHAP feature attributions paired with DiCE model-based counterfactual scenario simulation.
- **RAG-Grounded Generative AI**: Google Gemini AI (`gemini-3.5-flash-lite`) produces evidence-backed narrative reports grounded in occupational health literature.

---

## 📊 Performance Benchmark Summary

| Model Rank | Machine Learning Algorithm | Stress Accuracy (%) | Anxiety Accuracy (%) | Depression Accuracy (%) | Model Hierarchy Rank |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **MLP Neural Network** | **97.9%** | **95.2%** | **94.5%** | **Rank #1 (Best)** |
| **2** | **XGBoost Classifier** | **97.0%** | **94.3%** | **93.9%** | Rank #2 |
| **3** | **LightGBM Classifier** | **96.5%** | **93.7%** | **93.3%** | Rank #3 |
| **4** | **Random Forest Classifier** | **88.5%** | **87.2%** | **86.5%** | Rank #4 |
| **5** | **Decision Tree Classifier** | **85.2%** | **85.1%** | **85.0%** | Rank #5 |

---

## 🚀 Step-by-Step Instructions to Run Locally

Follow these instructions to set up and run the Streamlit dashboard on your local machine.

### Prerequisites
- **Python**: Version `3.9`, `3.10`, or `3.11` recommended.
- **Git** (optional, for cloning).

---

### Step 1: Open Terminal / PowerShell and Navigate to Project
Open your command prompt or terminal and change directory to the project root:
```bash
cd "e:\SEM 7\CAPSTONE PROJECT\Explainable-Machine-Learning-for-Stress-Anxiety-and-Depression-Prediction-Among-Healthcare-Workers"
```

---

### Step 2: Create and Activate Virtual Environment

#### On Windows (PowerShell / CMD):
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment in PowerShell
.\venv\Scripts\Activate.ps1
# OR in CMD:
# .\venv\Scripts\activate.bat
```

#### On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies

Install all core dependencies listed in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Configure Gemini API Key (Optional for RAG Reports)

To enable the RAG Generative AI report synthesis module:
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Add your API key to the `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
```
*(Note: You can also enter your API key directly in the Streamlit sidebar UI).*

---

### Step 5: Verify / Train Machine Learning Models

Ensure model pickle artifacts are generated. If `models/unified_models.pkl` does not exist, train all models using:
```bash
python train_and_save_all_models.py
```

---

### Step 6: Launch Streamlit Dashboard

Run the Streamlit web application:
```bash
streamlit run streamlit_app.py
```

After launching, Streamlit will display output similar to:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open **`http://localhost:8501`** in your web browser.

---

## 🖥️ Streamlit App Pages & Dashboard Features

The dashboard includes **8 interactive modules** accessible via the sidebar navigation:

1. **🏠 Home & Architecture**: Framework overview, system architecture diagram, stat metrics, and non-clinical disclaimer.
2. **🔮 Individual Assessment**: Interactive risk predictor for Stress, Anxiety, or Depression with calibrated confidence and reliability flags.
3. **🔍 Explainability (SHAP & DiCE)**: Global feature attributions, local waterfall pushes, and interactive counterfactual what-if simulation.
4. **📊 Model Comparison (5 Models)**: Performance matrices, calibration reliability diagrams, Fairlearn bias audit tables, and scientific performance rationale notes.
5. **🧪 Dataset & Ablation Analysis**: Empirical ablation experiments evaluating dataset sources, feature groups, and SMOTE resampling techniques.
6. **🏢 Organization Analytics**: Department burnout prevalence heatmaps, workload intensity scatter plots, and hospital unit risk breakdowns.
7. **🤖 RAG AI Reports**: Dense evidence retrieval passages and Gemini Generative AI narrative report synthesis.
8. **🛡️ Responsible AI & Governance**: Ethical considerations, data shift controls, leakage prevention, and non-clinical risk framing guidelines.

---

## 📓 Jupyter Notebook Execution

To explore the raw data analysis, preprocessing, and model training in a notebook interface:
```bash
jupyter notebook phase1_2_stress_prediction.ipynb
```
*(All 81 cells in `phase1_2_stress_prediction.ipynb` are fully executed and reproducible).*

---

## 📁 Repository Structure

```text
├── models/
│   └── unified_models.pkl           # Pre-trained models, scalers & feature names
├── phase1_2_stress_prediction.ipynb # Complete research jupyter notebook (81 cells)
├── streamlit_app.py                 # Interactive Streamlit Web Dashboard
├── train_and_save_all_models.py     # Script to fit & export all ML models
├── build_complete_notebook.py       # Notebook generator script
├── RESEARCH_PAPER_OUTPUT_DETAILS.md # Full empirical results document for paper writing
├── README.md                        # Project documentation and local run instructions
├── requirements.txt                 # Required Python dependencies
├── .env.example                     # Environment template for API keys
├── .env                             # Environment configuration file
├── Stress-Lysis.csv                 # Biometric physiological dataset
├── Workplace_Survey_Data.xlsx       # Workload & shift survey dataset
├── DASS42.csv                       # Psychometric DASS-42 assessment dataset
└── Healthcare Workforce Mental...   # Department & organizational dataset
```

---

## 👥 Authors & Team Metadata

- **Institution**: Dayananda Sagar University (DSU), Bangalore, India
- **Academic Term**: 2025 – 2026
- **Project Members**:
  - **Nandeesh N B** (ENG23AM0047)
  - **N Rohith** (ENG23AM0046)
  - **M Harshith Raju** (ENG23AM0040)

---
*Developed for Healthcare Workforce Mental Health Awareness & Occupational Safety Research.*
