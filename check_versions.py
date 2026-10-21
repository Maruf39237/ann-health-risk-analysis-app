from importlib.metadata import version, PackageNotFoundError

packages = [
    "streamlit",
    "pandas",
    "numpy",
    "joblib",
    "tensorflow",
    "matplotlib",
    "PyMuPDF",
    "scikit-learn",
    "xgboost",
    "catboost",
    "lightgbm",
    "imbalanced-learn",
    "seaborn",
    "jupyter",
    "ipykernel",
]

for package in packages:
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        print(f"{package}==NOT_INSTALLED")