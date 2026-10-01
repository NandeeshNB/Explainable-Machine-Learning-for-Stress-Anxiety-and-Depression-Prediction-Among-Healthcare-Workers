import sys
sys.stdout.reconfigure(encoding='utf-8')

# Check which packages are available
packages_to_check = [
    'xgboost', 'lightgbm', 'tensorflow', 'sklearn', 'pandas', 'numpy',
    'shap', 'imblearn', 'fairlearn', 'dice_ml', 'plotly', 'pdfplumber'
]

print(f"Python: {sys.executable}")
print("="*50)
for pkg in packages_to_check:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'available')
        print(f"  {pkg:20s}: {ver}")
    except ImportError:
        print(f"  {pkg:20s}: NOT INSTALLED")

print("="*50)
