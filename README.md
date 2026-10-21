# 🩺 ANN-Based Diabetes & Health Risk Prediction

An end-to-end **Machine Learning and Deep Learning project** that explores diabetes-related health indicators and predicts potential health risk using Artificial Neural Networks (ANNs) and gradient boosting models.

The project includes **exploratory data analysis (EDA), data preprocessing, model training and evaluation**, and an interactive **Streamlit web application** for health-risk prediction.

---

## 🚀 Live Demo

**Streamlit Cloud:** [Open the Live App](https://ann-health-risk-analysis-app-maruf39237.streamlit.app/)

---

## ✨ Features

- 📊 **Interactive Exploratory Data Analysis**
  - Health indicator distributions
  - Correlation and relationship analysis
  - Diabetes-related trends
  - Age, BMI, lifestyle, and socioeconomic insights

- 🧹 **Data Preparation and Preprocessing**
  - Dataset combining and preparation
  - Feature processing
  - Class imbalance investigation

- 🤖 **Model Training and Comparison**
  - Three ANN training approaches
  - XGBoost
  - CatBoost
  - LightGBM

- 🔮 **Interactive Health-Risk Prediction**
  - Enter health information
  - Load a random person from the available dataset
  - Generate predictions using the available models
  - Reveal the true status for supported sample profiles

- 📈 **Model Evaluation**
  - Accuracy, precision, recall, and F1-score
  - Confusion matrices
  - Saved model evaluation metrics

- 🖥️ **Multipage Streamlit Interface**
  - Home dashboard
  - Exploratory Data Analysis
  - Prediction
  - Report and insights

---

## 📁 Project Structure

```text
ANN-health-risk-prediction-project/
│
├── Assets/          # EDA visualizations and confusion matrices
├── Data/            # BRFSS health indicator datasets
├── DOC/             # EDA insights report (PDF)
├── Models/          # Trained models, preprocessing, and metrics
├── Notebooks/       # Data preparation, EDA, and experiments
│
├── Pages/
│   ├── home.py      # Home dashboard
│   ├── eda.py       # Exploratory data analysis
│   ├── prediction.py# Interactive prediction page
│   └── report.py    # Report and insights page
│
├── Structure/
│   └── structure.txt
│
├── app.py           # Main Streamlit entry point
├── requirements.txt # Python dependencies
├── runtime.txt      # Runtime configuration
├── .python-version  # Python version configuration
├── .gitignore
└── README.md
```

---

## 📓 Notebooks

The project includes four notebooks for data preparation, exploration, and model development.

| Notebook                           | Purpose                                     |
| ---------------------------------- | ------------------------------------------- |
| `dataset-combining.ipynb`          | Dataset preparation and combination         |
| `EDA.ipynb`                        | Exploratory data analysis and visualization |
| `ML-Algorithm.ipynb`               | Model development and experimentation       |
| `Models_Run_fair_comparison.ipynb` | Comparative model evaluation (ANN)          |

---

## 📊 Dataset

The `Data/` directory contains the following datasets:

- `diabetes_012_health_indicators_BRFSS2015.csv`
- `diabetes_012_health_indicators_BRFSS2021.csv`
- `diabetes_012_health_indicators_BRFSS2015_BRFSS2021.csv`

The project uses health indicators from the Behavioral Risk Factor Surveillance System (BRFSS) to investigate factors associated with diabetes and health risk.

Features include age, BMI, general health, physical activity, blood pressure, cholesterol, healthcare access, and lifestyle indicators.

See the notebooks and PDF report for details about the data preparation and analysis process.

---

## 🤖 Models Used

| Model             | Approach                                   |
| ----------------- | ------------------------------------------ |
| ANN — Raw Weights | Original class distribution                |
| ANN — Downsampled | Downsampling for class imbalance           |
| ANN — SMOTE       | Synthetic Minority Over-sampling Technique |
| XGBoost           | Gradient boosting                          |
| CatBoost          | Gradient boosting                          |
| LightGBM          | Gradient boosting                          |

Trained models and associated evaluation artifacts are stored in `Models/`.

Model performance should be compared using the same held-out test set. For At-Risk detection, recall, precision, F1-score, and PR-AUC are important in addition to accuracy.

---

## 🔬 Evaluation and Insights

The project includes saved evaluation metrics and confusion matrices for analyzing model performance.

Evaluation artifacts include:

- `home_metrics.json`
- `model_evaluations.json`
- `model_catboost_evaluations.json`
- `model_lightgbm_evaluations.json`
- `model_xgb_evaluations.json`

Confusion matrix visualizations are available in `Assets/`.

**Important:** A model with high accuracy may still miss many at-risk individuals, particularly when classes are imbalanced. Threshold selection should be based on validation data, with final performance reported on an untouched test set.

---

## 🚀 Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Maruf39237/ann-health-risk-analysis-app.git
cd ann-health-risk-analysis-app
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal to access the application.

---

## 🔮 How Prediction Works

The application takes health-related input features and uses the selected trained model to generate a prediction.

```text
User Input / Sample Profile
          ↓
Input Validation
          ↓
Feature Ordering and Preprocessing
          ↓
Selected Trained Model
          ↓
Risk Prediction and Probability
          ↓
Display Results
          ↓
Reveal True Status (when available)
```

The ANN models use the saved preprocessing artifact where required, while the boosting models use their corresponding prediction pipelines.

For random sample profiles, the displayed true status is available only when the application has access to the corresponding label.

---

## 🛠️ Technologies Used

| Category             | Technology                        |
| -------------------- | --------------------------------- |
| Programming Language | Python                            |
| Data Manipulation    | Pandas, NumPy                     |
| Visualization        | Matplotlib, Seaborn               |
| Deep Learning        | TensorFlow / Keras                |
| Machine Learning     | Scikit-learn                      |
| Gradient Boosting    | XGBoost, CatBoost, LightGBM       |
| Imbalanced Data      | SMOTE                             |
| Model Serialization  | Pickle, Keras, and JSON artifacts |
| Web Application      | Streamlit                         |
| Experimentation      | Jupyter Notebook                  |
| Version Control      | Git and GitHub                    |

See `requirements.txt` for the project's declared dependencies.

---

## ⚠️ Limitations and Disclaimer

- This project is intended for **educational and research purposes**.
- Predictions reflect patterns learned from the available dataset and may not generalize to every population.
- Model predictions are **not medical diagnoses** and should not replace professional medical advice.
- Performance metrics alone do not establish clinical usefulness or safety.
- Data quality, class imbalance, probability calibration, and the selected prediction threshold can affect results.
- Evaluation results depend on the data-splitting strategy and experimental setup.
- The application should not be used to make medical decisions without appropriate clinical validation.

---

## 👨‍💻 Author

**Mohi Uddin Maruf**

- **GitHub:** [Maruf39237](https://github.com/Maruf39237)
- **Repository:** [ANN Health Risk Analysis App](https://github.com/Maruf39237/ann-health-risk-analysis-app)
- **Email:** [maruf39237@gmail.com](mailto:maruf39237@gmail.com)

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the project's software, subject to the terms and conditions of the license.

See the [`LICENSE`](LICENSE) file for the full license text.

**Disclaimer:** The MIT License does not constitute medical approval or validation of this application. This project is intended for educational and research purposes only and is not a substitute for professional medical advice.
