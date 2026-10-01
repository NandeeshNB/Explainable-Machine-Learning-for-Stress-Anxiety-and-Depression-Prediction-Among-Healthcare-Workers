import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)

print("Training and Saving Models for All 3 Targets (Stress, Anxiety, Depression)...")

# 1. Load Datasets
physio = pd.read_csv('Stress-Lysis.csv')
workplace = pd.read_excel('Workplace_Survey_Data.xlsx')
dass = pd.read_csv('DASS42.csv', sep=None, engine='python', on_bad_lines='skip')
try:
    workforce = pd.read_csv('Healthcare Workforce Mental Health Dataset.csv')
except:
    workforce = None

# 2. Prepare Stress Dataset (Physio + Workplace)
physio_fe = physio.copy()
physio_fe['activity_level'] = pd.qcut(physio_fe['Step_count'], q=3, labels=[0, 1, 2], duplicates='drop').astype(float)
physio_fe['env_stress'] = (
    (physio_fe['Temperature'] - physio_fe['Temperature'].mean()) / physio_fe['Temperature'].std() +
    (physio_fe['Humidity'] - physio_fe['Humidity'].mean()) / physio_fe['Humidity'].std()
) / 2

wp_fe = workplace.copy()
wp_fe['workload_intensity'] = wp_fe['patients_attended'] / (wp_fe['workhours'] + 1e-6)
wp_fe['department_encoded'] = LabelEncoder().fit_transform(wp_fe['department'].astype(str))

np.random.seed(42)
sample_wp = wp_fe.sample(n=len(physio_fe), replace=True).reset_index(drop=True)

stress_df = pd.concat([
    physio_fe[['Humidity', 'Temperature', 'Step_count', 'activity_level', 'env_stress', 'Stress_Level']].reset_index(drop=True),
    sample_wp[['workhours', 'patients_attended', 'dept_stress', 'workload_intensity', 'department_encoded']].reset_index(drop=True)
], axis=1)

X_stress = stress_df.drop(columns=['Stress_Level'])
y_stress = stress_df['Stress_Level']

# 3. Prepare DASS-42 Datasets (Anxiety & Depression)
DEPRESSION_ITEMS = [3,5,10,13,16,17,21,24,26,31,34,37,38,42]
ANXIETY_ITEMS    = [2,4,7,9,15,19,20,23,25,28,30,36,40,41]
STRESS_ITEMS     = [1,6,8,11,12,14,18,22,27,29,32,33,35,39]

dep_cols    = [f'Q{i}A' for i in DEPRESSION_ITEMS if f'Q{i}A' in dass.columns]
anx_cols    = [f'Q{i}A' for i in ANXIETY_ITEMS if f'Q{i}A' in dass.columns]
stress_cols = [f'Q{i}A' for i in STRESS_ITEMS if f'Q{i}A' in dass.columns]

for col in dep_cols + anx_cols + stress_cols:
    dass[col] = (pd.to_numeric(dass[col], errors='coerce') - 1).clip(0, 3)

dass['DASS_Depression_raw'] = dass[dep_cols].sum(axis=1) * 2
dass['DASS_Anxiety_raw']    = dass[anx_cols].sum(axis=1) * 2
dass['DASS_Stress_raw']     = dass[stress_cols].sum(axis=1) * 2

def depression_cat(s): return 0 if s<=9 else 1 if s<=13 else 2 if s<=20 else 3 if s<=27 else 4
def anxiety_cat(s):    return 0 if s<=7 else 1 if s<=9  else 2 if s<=14 else 3 if s<=19 else 4
def to_3class(c):       return 0 if c==0 else 1 if c<=2 else 2

dass['depression_target'] = dass['DASS_Depression_raw'].apply(depression_cat).apply(to_3class)
dass['anxiety_target']    = dass['DASS_Anxiety_raw'].apply(anxiety_cat).apply(to_3class)

dass_clean = dass.dropna(subset=dep_cols + anx_cols + stress_cols).reset_index(drop=True)

X_anxiety = dass_clean[dep_cols + stress_cols]
y_anxiety = dass_clean['anxiety_target']

X_depression = dass_clean[anx_cols + stress_cols]
y_depression = dass_clean['depression_target']

scalers = {}
models_dict = {}

tasks = {
    'stress': (X_stress, y_stress),
    'anxiety': (X_anxiety, y_anxiety),
    'depression': (X_depression, y_depression)
}

for task_name, (X, y) in tasks.items():
    print(f"Training models for {task_name.upper()}...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    scalers[task_name] = scaler
    
    dt = DecisionTreeClassifier(max_depth=6, random_state=42).fit(X_tr_sc, y_tr)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(X_tr_sc, y_tr)
    xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42).fit(X_tr_sc, y_tr)
    lgb_m = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1).fit(X_tr_sc, y_tr)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42).fit(X_tr_sc, y_tr)
    
    models_dict[task_name] = {
        'Decision Tree': dt,
        'Random Forest': rf,
        'XGBoost': xgb_m,
        'LightGBM': lgb_m,
        'MLP Neural Net': mlp,
        'feature_names': list(X.columns)
    }

artifacts = {
    'scalers': scalers,
    'models': models_dict,
    'feature_names': {t: list(X.columns) for t, (X, y) in tasks.items()}
}

with open('models/unified_models.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(models_dict['stress']['XGBoost'], f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scalers['stress'], f)

print("✓ All models and scalers saved to models/unified_models.pkl successfully!")
