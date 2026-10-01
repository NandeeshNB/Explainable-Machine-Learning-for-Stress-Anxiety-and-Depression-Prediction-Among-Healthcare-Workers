import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('phase1_2_stress_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
summary = []
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    cell_type = cell['cell_type']
    first_line = src.split('\n')[0][:150] if src else ''
    summary.append(f'Cell {i} [{cell_type.upper()}]: {first_line}')

with open('notebook_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f'Total cells: {len(cells)}\n\n')
    f.write('\n'.join(summary))
print(f'Total cells: {len(cells)}')
print('Written to notebook_summary.txt')
