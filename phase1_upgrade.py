"""
PHASE 1 UPGRADE SCRIPT
======================
Updates the notebook header, phase overview, and imports section.
Run this script once to apply Phase 1 changes to the notebook.
"""

import json, copy, sys
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOK_PATH = 'phase1_2_stress_prediction.ipynb'

# ── Load the notebook ─────────────────────────────────────────────────────────
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Loaded notebook with {len(cells)} cells")

# ── Helper to make a new markdown cell ────────────────────────────────────────
def md_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

# ── Helper to make a new code cell ────────────────────────────────────────────
def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 1: Update Cell 0 — Main title
# ══════════════════════════════════════════════════════════════════════════════
NEW_TITLE = """\
# Mental Health Prediction Using Machine Learning for Healthcare Workers
## An Explainable, Fair and Calibrated Multi-Source Framework for Stress, Anxiety and Depression Risk Prediction

**Team Members:**
- Nandeesh N B - ENG23AM0047
- N Rohith - ENG23AM0046
- M Harshith Raju - ENG23AM0040

**Course:** 23AM3609 - Generative AI  
**Institution:** Dayananda Sagar University

---

### Project Overview

This notebook implements a **unified, multi-source machine-learning framework** for predicting mental health risk (Stress, Anxiety, and Depression) among healthcare workers.

| | Previous Semester | Current Semester |
|---|---|---|
| **Datasets** | 2 (Stress-Lysis, Workplace Survey) | 4 (+ DASS-42, Healthcare Workforce) |
| **Targets** | Stress only | Stress + Anxiety + Depression |
| **Models** | RF, XGBoost, MLP | DT, RF, XGBoost, LightGBM, MLP |
| **Evaluation** | Accuracy / F1 | + Calibration + Fairness |
| **Explainability** | SHAP | SHAP + DiCE Counterfactuals |
| **AI Reports** | Basic Generative AI | RAG-Grounded Generative AI |

### Implementation Phases
| Phase | Description |
|---|---|
| Phase 1 | Dataset Loading & Audit (all 4 datasets) |
| Phase 2 | Source-Specific Preprocessing & Feature Engineering |
| Phase 3 | Unified Feature Representation |
| Phase 4 | Train/Val/Test Split + Class Imbalance Analysis |
| Phase 5 | Baseline ML Training (5 Models × 3 Targets) |
| Phase 6 | Class Imbalance Experiments (SMOTE / Class Weighting) |
| Phase 7 | Model Evaluation (Accuracy, Precision, Recall, F1, AUC, CM) |
| Phase 8 | Probability Calibration (Brier, ECE, Reliability Diagrams) |
| Phase 9 | Fairness Assessment & Mitigation (Fairlearn) |
| Phase 10 | Explainable AI — SHAP (Global + Local) |
| Phase 11 | Explainable AI — DiCE Counterfactuals |
| Phase 12 | Ablation Studies |
| Phase 13 | RAG-Grounded Generative AI Reports |
| Phase 14 | Final Summary & Results |

> **Disclaimer:** This system is an experimental and educational framework for mental-health risk prediction.  
> It is **not** a clinical diagnostic tool. Predictions are risk estimates, not diagnoses.\
"""

cells[0]['source'] = NEW_TITLE
print("✓ Cell 0 (title) updated")

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2: Update Cell 1 — Phase overview block
# ══════════════════════════════════════════════════════════════════════════════
NEW_PHASE_OVERVIEW = """\
---
## Phase 1: Data Collection, Audit and Integration

### Objectives:
1. Load and inspect all **four** complementary datasets
2. Audit each dataset for missing values, data types, and class distributions
3. Understand DASS-42 questionnaire structure and scoring
4. Perform Exploratory Data Analysis (EDA) for all four sources
5. Establish source labels for the unified analytical framework

### Datasets:
| Dataset | Role | Prediction Contribution |
|---|---|---|
| **Stress-Lysis** | Physical / Environmental | Stress prediction |
| **Healthcare Workplace Survey** | Occupational | Stress prediction |
| **DASS-42** | Psychological questionnaire | Anxiety + Depression + Stress |
| **Healthcare Workforce Mental Health** | Workforce-level | Auxiliary analysis |

> **Important:** These datasets originate from different sources and populations.  
> They will **not** be row-concatenated as if they represent the same individuals.  
> Each dataset provides complementary information in a unified analytical framework.\
"""

cells[1]['source'] = NEW_PHASE_OVERVIEW
print("✓ Cell 1 (phase overview) updated")

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 3: Update Cell 2 — Section header for imports
# ══════════════════════════════════════════════════════════════════════════════
cells[2]['source'] = "### 1.1 Import Required Libraries\n\nAll libraries for the full framework are imported here (some will be used in later phases)."
print("✓ Cell 2 (import header) updated")

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 4: Replace Cell 3 — Expanded imports
# ══════════════════════════════════════════════════════════════════════════════
NEW_IMPORTS = """\
# ── Core data manipulation ────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings
import os
import pickle
import json
warnings.filterwarnings('ignore')

# ── Visualization ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Scikit-learn: preprocessing ───────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, cross_validate
)
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, MinMaxScaler, label_binarize
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── Scikit-learn: metrics ─────────────────────────────────────────────────────
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    brier_score_loss, roc_curve, auc
)

# ── Scikit-learn: calibration ─────────────────────────────────────────────────
from sklearn.calibration import (
    CalibratedClassifierCV, calibration_curve
)

# ── Machine Learning models ───────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier               # NEW: interpretable baseline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import lightgbm as lgb                                       # NEW: LightGBM
from lightgbm import LGBMClassifier

# ── Deep Learning ─────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ── Class Imbalance handling ──────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠ imbalanced-learn not installed — SMOTE will be skipped. Install: pip install imbalanced-learn")

# ── Fairness (Fairlearn) ──────────────────────────────────────────────────────
try:
    from fairlearn.metrics import MetricFrame, equalized_odds_difference, demographic_parity_difference
    from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    print("⚠ Fairlearn not installed — Fairness section will be skipped. Install: pip install fairlearn")

# ── SHAP ──────────────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP not installed — Install: pip install shap")

# ── DiCE Counterfactual Explanations ─────────────────────────────────────────
try:
    import dice_ml
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    print("⚠ DiCE not installed — Counterfactual section will be skipped. Install: pip install dice-ml")

# ── Statistical analysis ──────────────────────────────────────────────────────
from scipy.stats import pearsonr, spearmanr

# ── Visualization settings ────────────────────────────────────────────────────
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ── Reproducibility seeds ─────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ── Report library availability ───────────────────────────────────────────────
print("=" * 65)
print("  Mental Health Prediction Framework — Library Status")
print("=" * 65)
print(f"  pandas          v{pd.__version__}")
print(f"  numpy           v{np.__version__}")
print(f"  scikit-learn    ✓")
print(f"  tensorflow      v{tf.__version__}")
print(f"  xgboost         ✓")
print(f"  lightgbm        v{lgb.__version__}")
print(f"  SMOTE           {'✓' if SMOTE_AVAILABLE else '✗ (install imbalanced-learn)'}")
print(f"  Fairlearn       {'✓' if FAIRLEARN_AVAILABLE else '✗ (install fairlearn)'}")
print(f"  SHAP            {'✓' if SHAP_AVAILABLE else '✗ (install shap)'}")
print(f"  DiCE            {'✓' if DICE_AVAILABLE else '✗ (install dice-ml)'}")
print("=" * 65)
print("  ✓ Core libraries loaded successfully!")
print("=" * 65)\
"""

cells[3]['source'] = NEW_IMPORTS
cells[3]['outputs'] = []
cells[3]['execution_count'] = None
print("✓ Cell 3 (imports) updated")

# ── Save notebook ─────────────────────────────────────────────────────────────
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n" + "="*60)
print("✓ PHASE 1 UPDATE COMPLETE")
print("="*60)
print(f"  Updated cells: 0, 1, 2, 3")
print(f"  Total cells: {len(cells)}")
print(f"  Notebook saved to: {NOTEBOOK_PATH}")
