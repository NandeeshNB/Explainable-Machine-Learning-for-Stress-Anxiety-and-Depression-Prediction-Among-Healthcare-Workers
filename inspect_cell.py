import json

with open('phase1_2_stress_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the task definition cell and show its content
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'Task 2: ANXIETY PREDICTION' in src and 'anx_input_cols' in src:
        print(f"Cell {i}:")
        print(repr(src[:2000]))
        break
