import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

print('=== DASS-42 Dataset ===')
try:
    dass = pd.read_csv('DASS42.csv', on_bad_lines='skip', sep=None, engine='python')
    print('Shape:', dass.shape)
    print('Columns:', list(dass.columns)[:30])
    print(dass.head(3).to_string())
except Exception as e:
    print('Error:', e)
    try:
        dass = pd.read_csv('DASS42.csv', on_bad_lines='skip', sep='\t')
        print('Tab-separated Shape:', dass.shape)
        print('Columns:', list(dass.columns)[:30])
        print(dass.head(3).to_string())
    except Exception as e2:
        print('Error2:', e2)

print()
print('=== Healthcare Workforce Mental Health Dataset ===')
wf = pd.read_csv('Healthcare Workforce Mental Health Dataset.csv')
print('Shape:', wf.shape)
print('Columns:', list(wf.columns))
print(wf.head(3).to_string())

print()
print('=== Stress-Lysis Dataset ===')
sl = pd.read_csv('Stress-Lysis.csv')
print('Shape:', sl.shape)
print('Columns:', list(sl.columns))
print(sl.head(3).to_string())
