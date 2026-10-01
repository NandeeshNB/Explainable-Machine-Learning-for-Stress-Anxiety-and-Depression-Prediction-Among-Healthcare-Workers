import json, ast

with open('phase1_2_stress_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = cell['source']
    src_text = src if isinstance(src, str) else ''.join(src)
    try:
        ast.parse(src_text)
    except SyntaxError as e:
        errors.append(f'Cell {i}: {e}')

if errors:
    print('SYNTAX ERRORS:')
    for e in errors: print(' ', e)
else:
    print('All code cells parse cleanly - no syntax errors')
