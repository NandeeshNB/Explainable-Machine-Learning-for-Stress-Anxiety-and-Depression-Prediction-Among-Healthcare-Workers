"""
PHASE 2 UPGRADE SCRIPT
======================
Adds cells for:
- Loading & auditing the DASS-42 dataset
- Loading & auditing the Healthcare Workforce Mental Health dataset
- Source-aware data audit summary

This inserts new cells AFTER the existing Cell 21 (end of phase 1 EDA / dept baseline).
"""

import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOK_PATH = 'phase1_2_stress_prediction.ipynb'

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Loaded notebook with {len(cells)} cells")

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

# ── Find insertion point ────────────────────────────────────────────────────
# Find cell 22 (the "---" separator between Phase 1 and Phase 2 sections)
# We insert the new Phase 2 dataset cells BEFORE the ML section
# Currently: Cell 22 is the "---" markdown before Phase 2 ML block
# We'll insert AFTER cell 21 (dept stress baseline) and BEFORE cell 22

INSERT_AFTER = 21   # after Department-wise Stress Baseline code cell

# ── Build new cells to insert ────────────────────────────────────────────────
new_cells = []

# ------------------------------------------------------------------
# Section header: new datasets
# ------------------------------------------------------------------
new_cells.append(md_cell("""\
---
### 1.11 Load DASS-42 Dataset (Psychological Component)

The **DASS-42** (Depression Anxiety Stress Scales - 42 items) dataset provides the psychological component of our framework.

- **Size:** ~39,775 records × 172 columns  
- **Structure:** Q1A–Q42A = answer (0–3), Q1I–Q42I = item position, Q1E–Q42E = elapsed time  
- **DASS-42 is NOT restricted to healthcare workers.** It is used here as the **psychological dimension** of the unified framework.  
- Subscale scores will be calculated and converted to risk categories.\
"""))

new_cells.append(code_cell("""\
# ── Load DASS-42 Dataset ─────────────────────────────────────────────────────
print("Loading DASS-42 Dataset...")
print("=" * 65)

dass_raw = pd.read_csv(
    'DASS42.csv',
    sep=None,           # auto-detect delimiter
    engine='python',
    on_bad_lines='skip'
)

print(f"  Shape          : {dass_raw.shape}")
print(f"  Rows           : {dass_raw.shape[0]:,}")
print(f"  Columns        : {dass_raw.shape[1]}")
print(f"  Missing values : {dass_raw.isnull().sum().sum():,}")

# ── Show column categories ────────────────────────────────────────────────────
answer_cols = [c for c in dass_raw.columns if c.endswith('A')]
item_cols   = [c for c in dass_raw.columns if c.endswith('I')]
time_cols   = [c for c in dass_raw.columns if c.endswith('E')]
meta_cols   = [c for c in dass_raw.columns
               if c not in answer_cols + item_cols + time_cols]

print(f"\\n  Answer columns (Q#A): {len(answer_cols)}")
print(f"  Item-order cols (Q#I): {len(item_cols)}")
print(f"  Elapsed-time  (Q#E)  : {len(time_cols)}")
print(f"  Metadata columns     : {len(meta_cols)}")
print(f"  Metadata: {meta_cols}")

print("\\n  Sample rows:")
print(dass_raw[answer_cols[:6] + meta_cols[:5]].head(3).to_string())
print("\\n✓ DASS-42 loaded")\
"""))

new_cells.append(md_cell("""\
### 1.12 DASS-42 Scoring & Target Label Creation

**Scoring Rules (multiply subscale raw sum × 2 to match published norms):**

| Dimension | Items (Q#A) | Normal | Mild | Moderate | Severe | Extremely Severe |
|---|---|---|---|---|---|---|
| Depression | 3,5,10,13,16,17,21,24,26,31,34,37,38,42 | 0–9 | 10–13 | 14–20 | 21–27 | ≥28 |
| Anxiety | 2,4,7,9,15,19,20,23,25,28,30,36,40,41 | 0–7 | 8–9 | 10–14 | 15–19 | ≥20 |
| Stress | 1,6,8,11,12,14,18,22,27,29,32,33,35,39 | 0–14 | 15–18 | 19–25 | 26–33 | ≥34 |

**Leakage Control:** When predicting Depression, the Depression raw score is **not** used as a feature.\
"""))

new_cells.append(code_cell("""\
# ── DASS-42 Subscale Scoring ──────────────────────────────────────────────────
print("Computing DASS-42 Subscale Scores...")
print("=" * 65)

# DASS-42 item mapping (1-indexed question numbers for each subscale)
DEPRESSION_ITEMS = [3,5,10,13,16,17,21,24,26,31,34,37,38,42]
ANXIETY_ITEMS    = [2,4,7,9,15,19,20,23,25,28,30,36,40,41]
STRESS_ITEMS     = [1,6,8,11,12,14,18,22,27,29,32,33,35,39]

# Build column names: Q3A, Q5A, etc.
dep_cols   = [f'Q{i}A' for i in DEPRESSION_ITEMS]
anx_cols   = [f'Q{i}A' for i in ANXIETY_ITEMS]
stress_cols= [f'Q{i}A' for i in STRESS_ITEMS]

# Keep only answer columns that exist in dataset
dep_cols   = [c for c in dep_cols if c in dass_raw.columns]
anx_cols   = [c for c in anx_cols if c in dass_raw.columns]
stress_cols= [c for c in stress_cols if c in dass_raw.columns]

print(f"  Depression items found: {len(dep_cols)}/14")
print(f"  Anxiety items found   : {len(anx_cols)}/14")
print(f"  Stress items found    : {len(stress_cols)}/14")

# Convert answer columns to numeric, coerce errors to NaN
for col in dep_cols + anx_cols + stress_cols:
    dass_raw[col] = pd.to_numeric(dass_raw[col], errors='coerce')

# Clamp to valid range [0, 3]
for col in dep_cols + anx_cols + stress_cols:
    dass_raw[col] = dass_raw[col].clip(0, 3)

# Compute subscale raw scores (sum * 2 per DASS-42 published scoring)
dass_raw['DASS_Depression_raw'] = dass_raw[dep_cols].sum(axis=1) * 2
dass_raw['DASS_Anxiety_raw']    = dass_raw[anx_cols].sum(axis=1) * 2
dass_raw['DASS_Stress_raw']     = dass_raw[stress_cols].sum(axis=1) * 2

# ── Create categorical severity labels ────────────────────────────────────────
def depression_cat(score):
    if score <= 9:  return 0   # Normal
    if score <= 13: return 1   # Mild
    if score <= 20: return 2   # Moderate
    if score <= 27: return 3   # Severe
    return 4                   # Extremely Severe

def anxiety_cat(score):
    if score <= 7:  return 0   # Normal
    if score <= 9:  return 1   # Mild
    if score <= 14: return 2   # Moderate
    if score <= 19: return 3   # Severe
    return 4                   # Extremely Severe

def stress_cat(score):
    if score <= 14: return 0   # Normal
    if score <= 18: return 1   # Mild
    if score <= 25: return 2   # Moderate
    if score <= 33: return 3   # Severe
    return 4                   # Extremely Severe

dass_raw['depression_cat_5'] = dass_raw['DASS_Depression_raw'].apply(depression_cat)
dass_raw['anxiety_cat_5']    = dass_raw['DASS_Anxiety_raw'].apply(anxiety_cat)
dass_raw['stress_cat_5']     = dass_raw['DASS_Stress_raw'].apply(stress_cat)

# ── Consolidate into 3 classes (Low / Moderate / High) for ML ────────────────
# 5-class → 3-class: 0=Normal → Low(0), 1-2=Mild/Moderate → Moderate(1), 3-4=Severe/ExSevere → High(2)
def to_3class(cat5):
    if cat5 == 0: return 0   # Low
    if cat5 <= 2: return 1   # Moderate
    return 2                  # High

dass_raw['depression_target'] = dass_raw['depression_cat_5'].apply(to_3class)
dass_raw['anxiety_target']    = dass_raw['anxiety_cat_5'].apply(to_3class)
dass_raw['dass_stress_target']= dass_raw['stress_cat_5'].apply(to_3class)

# ── Report distributions ──────────────────────────────────────────────────────
cat_labels = {0: 'Low', 1: 'Moderate', 2: 'High'}
for col, name in [('depression_target','Depression'),
                  ('anxiety_target','Anxiety'),
                  ('dass_stress_target','Stress (DASS)')]:
    dist = dass_raw[col].value_counts().sort_index()
    print(f"\\n  {name} Target Distribution:")
    for k, v in dist.items():
        print(f"    {cat_labels[k]:10s} ({k}): {v:6,} ({v/len(dass_raw)*100:.1f}%)")

print("\\n✓ DASS-42 scores and targets computed")\
"""))

new_cells.append(md_cell("""\
### 1.13 DASS-42 EDA\
"""))

new_cells.append(code_cell("""\
# ── DASS-42 Exploratory Data Analysis ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("DASS-42 Score Distributions", fontsize=15, fontweight='bold')

score_cols = [('DASS_Depression_raw', 'Depression Score', '#e74c3c'),
              ('DASS_Anxiety_raw',    'Anxiety Score',    '#e67e22'),
              ('DASS_Stress_raw',     'Stress Score',     '#9b59b6')]

for ax, (col, title, color) in zip(axes, score_cols):
    ax.hist(dass_raw[col].dropna(), bins=40, color=color, alpha=0.75, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    mean_val = dass_raw[col].mean()
    ax.axvline(mean_val, color='black', linestyle='--', label=f'Mean={mean_val:.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig('dass42_score_distributions.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ DASS-42 score distributions plotted")

# ── Class distribution bar chart ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("DASS-42 Target Class Distributions (3-class)", fontsize=15, fontweight='bold')

targets = [('depression_target', 'Depression', '#e74c3c'),
           ('anxiety_target',    'Anxiety',    '#e67e22'),
           ('dass_stress_target','Stress',     '#9b59b6')]
labels  = ['Low', 'Moderate', 'High']

for ax, (col, title, color) in zip(axes, targets):
    counts = dass_raw[col].value_counts().sort_index()
    bars   = ax.bar(labels[:len(counts)], counts.values, color=color, alpha=0.75, edgecolor='white')
    ax.set_title(f"{title} Target", fontweight='bold')
    ax.set_ylabel('Count')
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{count:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('dass42_class_distributions.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ DASS-42 class distributions plotted")\
"""))

new_cells.append(md_cell("""\
### 1.14 Load Healthcare Workforce Mental Health Dataset (Workforce Component)

The **Healthcare Workforce Mental Health Dataset** provides workforce-level context including workplace stressors, burnout frequency, job satisfaction, and turnover intention.

- **Size:** 5,000 records × 10 columns  
- **Note:** This dataset may be synthetic. Results derived from it must be presented as experimental evidence about the modeling framework, not clinical evidence.\
"""))

new_cells.append(code_cell("""\
# ── Load Healthcare Workforce Dataset ────────────────────────────────────────
print("Loading Healthcare Workforce Mental Health Dataset...")
print("=" * 65)

workforce_data = pd.read_csv('Healthcare Workforce Mental Health Dataset.csv')

print(f"  Shape          : {workforce_data.shape}")
print(f"  Rows           : {workforce_data.shape[0]:,}")
print(f"  Columns        : {workforce_data.shape[1]}")
print(f"  Missing values : {workforce_data.isnull().sum().sum()}")
print(f"\\n  Columns:")
for col in workforce_data.columns:
    dtype = workforce_data[col].dtype
    nunique = workforce_data[col].nunique()
    print(f"    {col:35s} | dtype={str(dtype):10s} | unique={nunique}")

print("\\n  Sample data:")
print(workforce_data.head(5).to_string())
print("\\n✓ Workforce dataset loaded")\
"""))

new_cells.append(md_cell("""\
### 1.15 Workforce Dataset EDA\
"""))

new_cells.append(code_cell("""\
# ── Workforce Dataset EDA ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Healthcare Workforce Mental Health Dataset — EDA", fontsize=15, fontweight='bold')

# 1. Stress Level distribution
ax = axes[0, 0]
workforce_data['Stress Level'].hist(bins=15, color='#e74c3c', alpha=0.75, ax=ax, edgecolor='white')
ax.set_title("Stress Level Distribution")
ax.set_xlabel("Stress Level (1-10)")
ax.set_ylabel("Count")

# 2. Burnout Frequency
ax = axes[0, 1]
burnout_counts = workforce_data['Burnout Frequency'].value_counts()
ax.bar(burnout_counts.index, burnout_counts.values, color='#e67e22', alpha=0.75, edgecolor='white')
ax.set_title("Burnout Frequency")
ax.set_xlabel("Frequency")
ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=30)

# 3. Job Satisfaction
ax = axes[0, 2]
workforce_data['Job Satisfaction'].hist(bins=5, color='#2ecc71', alpha=0.75, ax=ax, edgecolor='white')
ax.set_title("Job Satisfaction (1-5)")
ax.set_xlabel("Satisfaction Score")
ax.set_ylabel("Count")

# 4. Department Distribution
ax = axes[1, 0]
dept_counts = workforce_data['Department'].value_counts().head(10)
ax.barh(dept_counts.index, dept_counts.values, color='#3498db', alpha=0.75)
ax.set_title("Top Departments")
ax.set_xlabel("Count")

# 5. Turnover Intention
ax = axes[1, 1]
turnover = workforce_data['Turnover Intention'].value_counts()
ax.pie(turnover.values, labels=turnover.index, autopct='%1.1f%%',
       colors=['#2ecc71', '#e74c3c'], startangle=90)
ax.set_title("Turnover Intention")

# 6. Employee Type
ax = axes[1, 2]
emp_counts = workforce_data['Employee Type'].value_counts().head(8)
ax.barh(emp_counts.index, emp_counts.values, color='#9b59b6', alpha=0.75)
ax.set_title("Top Employee Types")
ax.set_xlabel("Count")

plt.tight_layout()
plt.savefig('workforce_eda.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ Workforce dataset EDA plots generated")\
"""))

new_cells.append(code_cell("""\
# ── Workforce Dataset Preprocessing ──────────────────────────────────────────
print("Preprocessing Healthcare Workforce Dataset...")
print("=" * 65)

wf = workforce_data.copy()

# Encode Burnout Frequency
burnout_map = {'Never': 0, 'Rarely': 1, 'Occasionally': 2, 'Often': 3, 'Always': 4}
wf['burnout_encoded'] = wf['Burnout Frequency'].map(burnout_map).fillna(2)

# Encode Access to EAPs
wf['eap_access'] = (wf['Access to EAPs'] == 'Yes').astype(int)

# Encode Turnover Intention
wf['turnover_flag'] = (wf['Turnover Intention'] == 'Yes').astype(int)

# Encode Workplace Factor
wf['workplace_factor_encoded'] = LabelEncoder().fit_transform(wf['Workplace Factor'].astype(str))

# Encode Department
wf['dept_workforce_encoded'] = LabelEncoder().fit_transform(wf['Department'].astype(str))

# Encode Employee Type
wf['employee_type_encoded'] = LabelEncoder().fit_transform(wf['Employee Type'].astype(str))

# Create stress category (1-3=Low, 4-6=Medium, 7-10=High)
wf['stress_category'] = pd.cut(wf['Stress Level'], bins=[0,3,6,10],
                                labels=[0,1,2], include_lowest=True).astype(int)

# Add source label
wf['data_source'] = 'workforce'

print(f"  Records after preprocessing: {len(wf):,}")
print(f"  Burnout encoded distribution:")
print(wf['burnout_encoded'].value_counts().sort_index().to_string())
print(f"\\n  Stress category distribution:")
print(wf['stress_category'].value_counts().sort_index().to_string())
print("\\n✓ Workforce dataset preprocessed")\
"""))

new_cells.append(md_cell("""\
### 1.16 Four-Dataset Audit Summary

A unified audit across all four data sources.\
"""))

new_cells.append(code_cell("""\
# ── Four-Dataset Unified Audit ────────────────────────────────────────────────
print("=" * 75)
print("  FOUR-DATASET AUDIT SUMMARY")
print("=" * 75)

datasets_info = {
    'Stress-Lysis': {
        'df': physio_data,
        'role': 'Physical / Environmental',
        'target': 'Stress_Level (0=Low, 1=Med, 2=High)',
        'synthetic': False
    },
    'Workplace Survey': {
        'df': workplace_data,
        'role': 'Occupational',
        'target': 'stress_score (continuous)',
        'synthetic': False
    },
    'DASS-42': {
        'df': dass_raw,
        'role': 'Psychological',
        'target': 'depression/anxiety/stress_target (0=Low, 1=Mod, 2=High)',
        'synthetic': False
    },
    'Healthcare Workforce': {
        'df': workforce_data,
        'role': 'Workforce-level',
        'target': 'Stress Level (1-10) → 3 classes',
        'synthetic': True
    }
}

for name, info in datasets_info.items():
    df = info['df']
    print(f"\\n  {'─'*70}")
    print(f"  Dataset      : {name}")
    print(f"  Role         : {info['role']}")
    print(f"  Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    missing = df.isnull().sum().sum()
    print(f"  Missing vals : {missing:,} ({missing/(df.shape[0]*df.shape[1])*100:.2f}%)")
    print(f"  Target       : {info['target']}")
    if info['synthetic']:
        print(f"  ⚠ SYNTHETIC  : Results from this dataset are experimental only")

print(f"\\n  {'─'*70}")
print(f"  NOTE: These datasets represent DIFFERENT populations.")
print(f"  They will NOT be row-concatenated as the same individuals.")
print(f"  Each provides a complementary dimension in the unified framework.")
print("=" * 75)
print("\\n✓ Phase 1 — All four datasets loaded and audited")\
"""))

# ── Insert new cells after INSERT_AFTER ──────────────────────────────────────
updated_cells = cells[:INSERT_AFTER + 1] + new_cells + cells[INSERT_AFTER + 1:]
nb['cells'] = updated_cells

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"✓ PHASE 2 UPDATE COMPLETE")
print(f"{'='*60}")
print(f"  Inserted {len(new_cells)} new cells after cell {INSERT_AFTER}")
print(f"  Total cells now: {len(updated_cells)}")
print(f"  Notebook saved to: {NOTEBOOK_PATH}")
