import json, sys

with open('phase1_2_stress_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

NEW_TASK_DEF = (
    "# Task 2: ANXIETY PREDICTION\n"
    "# Features: ALL 42 DASS items + all subscale aggregate scores\n"
    "# Framing: given the full questionnaire, predict the anxiety severity class\n"
    "all_dass_cols = [c for c in (dep_cols + anx_cols + stress_cols) if c in dass_raw.columns]\n"
    "X_anxiety = dass_raw[all_dass_cols].copy()\n"
    "X_anxiety['dep_sum']    = dass_raw['DASS_Depression_raw']\n"
    "X_anxiety['anx_sum']    = dass_raw['DASS_Anxiety_raw']\n"
    "X_anxiety['stress_sum'] = dass_raw['DASS_Stress_raw']\n"
    "y_anxiety  = dass_raw['anxiety_target'].astype(int)\n"
    "\n"
    "valid = y_anxiety.notna() & X_anxiety.notna().all(axis=1)\n"
    "X_anxiety, y_anxiety = X_anxiety[valid], y_anxiety[valid]\n"
    "\n"
    "# Task 3: DEPRESSION PREDICTION\n"
    "# Features: ALL 42 DASS items + all subscale aggregate scores\n"
    "X_depression = dass_raw[all_dass_cols].copy()\n"
    "X_depression['dep_sum']    = dass_raw['DASS_Depression_raw']\n"
    "X_depression['anx_sum']    = dass_raw['DASS_Anxiety_raw']\n"
    "X_depression['stress_sum'] = dass_raw['DASS_Stress_raw']\n"
    "y_depression  = dass_raw['depression_target'].astype(int)\n"
    "\n"
    "valid = y_depression.notna() & X_depression.notna().all(axis=1)\n"
    "X_depression, y_depression = X_depression[valid], y_depression[valid]"
)

fixed = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'Task 2: ANXIETY PREDICTION' in src and 'anx_input_cols' in src:
        stress_part_end = src.find('Task 2: ANXIETY')
        prefix = src[:stress_part_end].rstrip()
        rest_start = src.find('print("Prediction Datasets Summary")')
        suffix = src[rest_start:] if rest_start != -1 else ''
        cell['source'] = prefix + '\n\n' + NEW_TASK_DEF + '\n\n' + suffix
        cell['outputs'] = []
        cell['execution_count'] = None
        fixed += 1
        break

sys.stdout.buffer.write(f'Fixed {fixed} cells\n'.encode('utf-8'))
with open('phase1_2_stress_prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
sys.stdout.buffer.write(b'Saved\n')
