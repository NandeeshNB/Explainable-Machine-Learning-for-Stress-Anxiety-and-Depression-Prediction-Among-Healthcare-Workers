import json, sys

with open('phase1_2_stress_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
executed   = [c for c in code_cells if c.get('execution_count') is not None]
errors     = [c for c in code_cells if any(o.get('output_type') == 'error' for o in c.get('outputs', []))]

out = f"Executed: {len(executed)}/{len(code_cells)}  |  Errors: {len(errors)}\n"
sys.stdout.buffer.write(out.encode('utf-8'))

for cell in executed:
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'BEST MODEL (by F1) PER TASK' in src or ('MODEL COMPARISON' in src and 'results_df' in src):
        for o in cell.get('outputs', []):
            if o.get('output_type') == 'stream':
                txt = ''.join(o.get('text', []))
                # Only print the comparison table
                if 'ANXIETY' in txt or 'DEPRESSION' in txt or 'BEST MODEL' in txt:
                    sys.stdout.buffer.write(txt[:3000].encode('utf-8'))
                    break
        break
