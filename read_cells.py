import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('phase1_2_stress_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
# Print first 5 cells fully
for i in range(5):
    cell = cells[i]
    src = ''.join(cell['source'])
    print(f'\n{"="*60}')
    print(f'Cell {i} [{cell["cell_type"].upper()}]')
    print('='*60)
    print(src)
