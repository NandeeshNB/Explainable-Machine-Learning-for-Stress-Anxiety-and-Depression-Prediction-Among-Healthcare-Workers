"""
PHASE 3 UPGRADE SCRIPT
======================
Adds cells for:
- Unified Feature Representation (source-labeled DataFrames)
- Three prediction task definitions (Stress, Anxiety, Depression)
- Data leakage controls
- Train/Val/Test split for each target
- Class imbalance analysis

This inserts new cells BEFORE the existing "Phase 2 ML" section (after old Cell 22,
which is now shifted by 13 cells → becomes cell 35 approx).
"""

import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOK_PATH = 'phase1_2_stress_prediction.ipynb'

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Loaded notebook with {len(cells)} cells")

# ── Find insertion point: the '---' cell before "Phase 2: ML" ────────────────
# Find the markdown cell that says "### 2.1 Prepare Data for Machine Learning"
target_text = "### 2.1 Prepare Data for Machine Learning"
insert_before = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown' and target_text in ''.join(cell['source']):
        insert_before = i
        break

if insert_before is None:
    # Fallback: find by "Phase 2" keyword
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown' and 'Phase 2' in ''.join(cell['source']) and 'Machine Learning' in ''.join(cell['source']):
            insert_before = i
            break

print(f"Inserting new cells before cell {insert_before}")

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

# ── New Phase 2 header ────────────────────────────────────────────────────────
# Update the existing "Phase 2" section label to be more descriptive
# And add new cells for unified representation BEFORE ML training

new_cells = []

new_cells.append(md_cell("""\
---
## Phase 2: Unified Feature Representation & Prediction Task Setup

### Objectives:
1. Create source-specific feature representations from each dataset
2. Build a unified analytical schema (without row-merging independent populations)
3. Define three prediction tasks: Stress, Anxiety, Depression
4. Implement strict data leakage controls
5. Perform Train/Validation/Test split for each target
6. Analyze class imbalance per target\
"""))

new_cells.append(md_cell("""\
### 2.1 Source-Specific Feature Representations

Each dataset is converted into a harmonized representation with a `data_source` label.  
**Leakage rule:** The subscale score being predicted is excluded from that target's features.\
"""))

new_cells.append(code_cell("""\
# ── Source A: Stress-Lysis (Physical/Environmental) ───────────────────────────
print("Building Source Representations...")
print("=" * 65)

# Use the already-processed physio_data + feature-engineered variables
# physio_data was loaded in Phase 1 with Humidity, Temperature, Step_count, Stress_Level

sl_repr = physio_data.copy()
sl_repr['data_source'] = 'stress_lysis'

# Add any derived features if not already present
if 'activity_level' not in sl_repr.columns:
    sl_repr['activity_level'] = pd.cut(
        sl_repr['Step_count'],
        bins=[0, 3000, 7000, sl_repr['Step_count'].max() + 1],
        labels=[0, 1, 2]
    ).astype(float)

if 'env_stress' not in sl_repr.columns:
    sl_repr['env_stress'] = (
        (sl_repr['Temperature'] - sl_repr['Temperature'].mean()) / sl_repr['Temperature'].std() +
        (sl_repr['Humidity']    - sl_repr['Humidity'].mean())    / sl_repr['Humidity'].std()
    )

sl_repr.rename(columns={'Stress_Level': 'stress_target'}, inplace=True)
# Add placeholder columns for anxiety/depression (NaN — not available in this source)
sl_repr['anxiety_target']    = np.nan
sl_repr['depression_target'] = np.nan

print(f"  Stress-Lysis representation : {sl_repr.shape}")
print(f"  Stress target distribution  : {sl_repr['stress_target'].value_counts().sort_index().to_dict()}")

# ── Source B: Workplace Survey (Occupational) ─────────────────────────────────
wp_repr = combined_data.copy() if 'combined_data' in dir() else workplace_data.copy()
wp_repr['data_source'] = 'workplace_survey'

# Normalize stress_score to 0-2 classes if not already done
if 'stress_target' not in wp_repr.columns:
    # Use stress_score to create Low/Medium/High
    wp_repr['stress_target'] = pd.cut(
        wp_repr['stress_score'] if 'stress_score' in wp_repr.columns else wp_repr.iloc[:, -1],
        bins=[-0.01, 33.33, 66.66, 101],
        labels=[0, 1, 2]
    ).astype(float)

wp_repr['anxiety_target']    = np.nan
wp_repr['depression_target'] = np.nan

print(f"\\n  Workplace Survey representation: {wp_repr.shape}")

# ── Source C: DASS-42 (Psychological) ────────────────────────────────────────
# dass_raw already has depression_target, anxiety_target, dass_stress_target
# Select the DASS answer columns as features (leakage-controlled below)
dass_feature_cols = dep_cols + anx_cols + stress_cols
dass_repr = dass_raw[dass_feature_cols + ['depression_target', 'anxiety_target', 'dass_stress_target', 'DASS_Depression_raw', 'DASS_Anxiety_raw', 'DASS_Stress_raw']].copy()
dass_repr['data_source'] = 'dass42'
# Rename DASS stress target to standard name
dass_repr.rename(columns={'dass_stress_target': 'stress_target'}, inplace=True)

# Drop rows with excessive missing answer values
missing_pct = dass_repr[dass_feature_cols].isnull().mean(axis=1)
dass_repr = dass_repr[missing_pct < 0.5].reset_index(drop=True)

print(f"\\n  DASS-42 representation : {dass_repr.shape}")
print(f"  Depression target dist :")
print(dass_repr['depression_target'].value_counts().sort_index().to_string())
print(f"  Anxiety target dist    :")
print(dass_repr['anxiety_target'].value_counts().sort_index().to_string())

# ── Source D: Healthcare Workforce (Workforce) ────────────────────────────────
wf_features = ['Stress Level', 'burnout_encoded', 'Job Satisfaction',
                'eap_access', 'turnover_flag', 'workplace_factor_encoded',
                'dept_workforce_encoded', 'employee_type_encoded',
                'Mental Health Absences', 'stress_category']
wf_features = [c for c in wf_features if c in wf.columns]

wf_repr = wf[wf_features + ['data_source']].copy()
wf_repr['stress_target']     = wf['stress_category']
wf_repr['anxiety_target']    = np.nan
wf_repr['depression_target'] = np.nan

print(f"\\n  Workforce representation  : {wf_repr.shape}")
print(f"  Stress category dist     :")
print(wf_repr['stress_target'].value_counts().sort_index().to_string())

print("\\n✓ All source representations built")\
"""))

new_cells.append(md_cell("""\
### 2.2 Define Three Prediction Datasets (Leakage-Controlled)

For each prediction task, we select only the relevant source and features:
- **Stress prediction** → Stress-Lysis features + Workplace features (no DASS stress score as input)
- **Anxiety prediction** → DASS-42 answer items (excluding anxiety subscale items as direct input; raw anxiety score excluded)
- **Depression prediction** → DASS-42 answer items (excluding depression subscale items as direct score; raw depression score excluded)\
"""))

new_cells.append(code_cell("""\
# ── Task 1: Stress Prediction Dataset ────────────────────────────────────────
print("Building Prediction Datasets...")
print("=" * 65)

# Stress: Use Stress-Lysis data (physical/environmental features)
stress_features = ['Humidity', 'Temperature', 'Step_count', 'activity_level', 'env_stress']
stress_features = [c for c in stress_features if c in sl_repr.columns]

X_stress = sl_repr[stress_features].copy()
y_stress  = sl_repr['stress_target'].astype(int)

# Remove NaN rows
valid_mask = y_stress.notna() & X_stress.notna().all(axis=1)
X_stress   = X_stress[valid_mask]
y_stress   = y_stress[valid_mask]

print(f"\\n  STRESS prediction dataset:")
print(f"    X shape : {X_stress.shape}")
print(f"    y dist  : {y_stress.value_counts().sort_index().to_dict()}")

# ── Task 2: Anxiety Prediction Dataset ────────────────────────────────────────
# Features: all DASS answer columns EXCEPT anxiety items
# (Leakage: anxiety item raw sum excluded; include stress + depression items as context)
anx_input_cols = dep_cols + stress_cols   # NO anxiety items to prevent leakage
anx_input_cols = [c for c in anx_input_cols if c in dass_repr.columns]

X_anxiety = dass_repr[anx_input_cols].copy()
y_anxiety  = dass_repr['anxiety_target'].astype(int)

valid_mask  = y_anxiety.notna() & X_anxiety.notna().all(axis=1)
X_anxiety   = X_anxiety[valid_mask]
y_anxiety   = y_anxiety[valid_mask]

print(f"\\n  ANXIETY prediction dataset:")
print(f"    X shape : {X_anxiety.shape}")
print(f"    y dist  : {y_anxiety.value_counts().sort_index().to_dict()}")

# ── Task 3: Depression Prediction Dataset ─────────────────────────────────────
# Features: all DASS answer columns EXCEPT depression items
dep_input_cols = anx_cols + stress_cols   # NO depression items to prevent leakage
dep_input_cols = [c for c in dep_input_cols if c in dass_repr.columns]

X_depression = dass_repr[dep_input_cols].copy()
y_depression  = dass_repr['depression_target'].astype(int)

valid_mask    = y_depression.notna() & X_depression.notna().all(axis=1)
X_depression  = X_depression[valid_mask]
y_depression  = y_depression[valid_mask]

print(f"\\n  DEPRESSION prediction dataset:")
print(f"    X shape : {X_depression.shape}")
print(f"    y dist  : {y_depression.value_counts().sort_index().to_dict()}")

print("\\n✓ Three prediction datasets defined with leakage controls")\
"""))

new_cells.append(md_cell("""\
### 2.3 Train / Validation / Test Split (70 / 15 / 15, Stratified)\
"""))

new_cells.append(code_cell("""\
# ── Stratified Train / Validation / Test Split ────────────────────────────────
print("Performing Stratified Train/Val/Test Splits (70/15/15)...")
print("=" * 65)

splits = {}

for task_name, X, y in [
    ('stress',     X_stress,     y_stress),
    ('anxiety',    X_anxiety,    y_anxiety),
    ('depression', X_depression, y_depression)
]:
    # Step 1: split off 30% as temp (to split into val+test)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    # Step 2: split temp into 50/50 → 15% val, 15% test of original
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )

    splits[task_name] = {
        'X_train': X_tr,   'y_train': y_tr,
        'X_val':   X_val,  'y_val':   y_val,
        'X_test':  X_te,   'y_test':  y_te
    }

    print(f"\\n  {task_name.upper()} splits:")
    print(f"    Train : {X_tr.shape[0]:5,} rows | {y_tr.value_counts().sort_index().to_dict()}")
    print(f"    Val   : {X_val.shape[0]:5,} rows | {y_val.value_counts().sort_index().to_dict()}")
    print(f"    Test  : {X_te.shape[0]:5,} rows | {y_te.value_counts().sort_index().to_dict()}")

# ── Standard scaler for each task ────────────────────────────────────────────
scalers = {}
for task_name in ['stress', 'anxiety', 'depression']:
    scaler = StandardScaler()
    splits[task_name]['X_train_sc'] = scaler.fit_transform(splits[task_name]['X_train'])
    splits[task_name]['X_val_sc']   = scaler.transform(splits[task_name]['X_val'])
    splits[task_name]['X_test_sc']  = scaler.transform(splits[task_name]['X_test'])
    scalers[task_name] = scaler

print("\\n✓ All splits and scalers ready")\
"""))

new_cells.append(md_cell("""\
### 2.4 Class Imbalance Analysis\
"""))

new_cells.append(code_cell("""\
# ── Class Imbalance Analysis ─────────────────────────────────────────────────
print("Class Imbalance Analysis (Training sets)...")
print("=" * 65)

task_labels = {
    'stress':     {0: 'Low', 1: 'Medium', 2: 'High'},
    'anxiety':    {0: 'Low', 1: 'Moderate', 2: 'High'},
    'depression': {0: 'Low', 1: 'Moderate', 2: 'High'}
}
task_colors = {
    'stress':     ['#2ecc71', '#f39c12', '#e74c3c'],
    'anxiety':    ['#3498db', '#e67e22', '#9b59b6'],
    'depression': ['#1abc9c', '#e74c3c', '#2c3e50']
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Training Set Class Distributions", fontsize=14, fontweight='bold')

for ax, task in zip(axes, ['stress', 'anxiety', 'depression']):
    y_tr = splits[task]['y_train']
    counts = y_tr.value_counts().sort_index()
    labels = [task_labels[task].get(k, str(k)) for k in counts.index]
    bars   = ax.bar(labels, counts.values,
                    color=task_colors[task][:len(counts)],
                    alpha=0.8, edgecolor='white', linewidth=1.5)
    ax.set_title(f"{task.capitalize()} (n={len(y_tr):,})", fontweight='bold')
    ax.set_ylabel("Count")
    for bar, cnt in zip(bars, counts.values):
        pct = cnt / len(y_tr) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(y_tr)*0.01,
                f'{cnt:,}\\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('class_imbalance_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

# ── Imbalance Ratio Report ────────────────────────────────────────────────────
print("\\nImbalance Ratio Summary:")
for task in ['stress', 'anxiety', 'depression']:
    y_tr   = splits[task]['y_train']
    counts = y_tr.value_counts().sort_index()
    ratio  = counts.max() / counts.min() if counts.min() > 0 else float('inf')
    print(f"  {task:12s}: majority/minority ratio = {ratio:.2f}x "
          f"{'→ SMOTE may help' if ratio > 1.5 else '→ balanced'}")

print("\\n✓ Class imbalance analysis complete")\
"""))

# ── Insert new cells BEFORE the ML section ───────────────────────────────────
updated_cells = cells[:insert_before] + new_cells + cells[insert_before:]
nb['cells'] = updated_cells

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"✓ PHASE 3 UPDATE COMPLETE")
print(f"{'='*60}")
print(f"  Inserted {len(new_cells)} new cells before cell {insert_before}")
print(f"  Total cells now: {len(updated_cells)}")
