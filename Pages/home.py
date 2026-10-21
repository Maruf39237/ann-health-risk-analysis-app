from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Health Analytics & Predictive Modeling Hub",
    page_icon="📊",
    # layout="wide",
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Models"
ASSET_PATH = ROOT / "Assets"

@st.cache_data(show_spinner="Loading precalculated metrics...")
def load_metrics_json():
    json_files = list(MODEL_DIR.glob("*.json"))
    if json_files:
        with open(json_files[0], "r") as f:
            return json.load(f)
    return {}

st.divider()
col1, col2 = st.columns([4, 2])

with col1:
    st.title("🩺Diabetes Health Indicators System🩺")
    st.markdown("#### Exploratory Data Analysis(EDA) of `BRFSS 2015` and `BRFSS 2021` Health Indicators")
    st.markdown("""##### Explore how diabetes status relates to the health conditions, lifestyle indicators, and demographic characteristics using a collection of pre-built EDA visualizations and multi-year survey data. 📜""")


with col2 :
    st.image("https://images.pexels.com/photos/7653096/pexels-photo-7653096.jpeg", width = 200, caption = "Every insight brings us one step closer to better health. 💡")

st.divider()

st.markdown(
    """
    ### 📚 Project Overview

    ##### This project presents a rigorous, data-driven exploration of diabetes-related health indicators using combined multi-year cohorts from the CDC Behavioral Risk Factor Surveillance System (`BRFSS 2015 & 2021`). It aims to uncover longitudinal patterns and complex relationships between diabetes status, chronic health conditions, lifestyle behaviors, and demographic characteristics.

    ##### Through an advanced collection of visual analyses, the project examines how clinical and socioeconomic factors—such as BMI, blood pressure, cholesterol, physical activity, age, education, and income—correlate with diabetes categories across survey cycles.


    ##### The application provides a structured platform to:
    ###### - ***Explore the Data :*** Inspect multi-year dataset statistics, data quality controls (handling duplicates and validation), and diabetes class distributions.
    ###### - ***Analyze Health Indicators :*** Examine individual clinical features and their statistical associations with target diabetes categories.
    ###### - ***Discover Patterns :*** Investigate behavioral, lifestyle, and socioeconomic gradients across populations.
    ###### - ***Visualize Findings :*** Present high-resolution exploratory insights through organized data visualizations and statistical asset cards.
    ###### - ***View Pre-Created Reports :*** Access pre-compiled reports based on the EDA and confusion matrices directly within the application to help users easily interpret all findings and insights.    
    ###### - ***Compare Predictive Models :*** Conduct performance comparisons of machine learning classifiers and deep learning neural network (`ANN`) pipelines tailored for highly imbalanced class distributions (incorporating SMOTE balancing and class weighting), complete with detailed classification reports and confusion matrices.
    

    ##### The overarching goal is to transform large-scale public health survey data into an interactive, transparent, and high-end analytical web application.
    """
)




st.divider()

st.markdown("### 📌 Project Highlights")
st.markdown(
    """
    <style>
    /* target only the numerical/value part inside the metric widget */
    div[data-testid="stMetricValue"] {
        color: #D4AF37 !important; /* Premium Metallic Gold */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)




metrics_data = load_metrics_json()

rows = metrics_data.get("total_records", metrics_data.get("rows", 453331))
cols = metrics_data.get("total_features", metrics_data.get("cols", 25))
missing = metrics_data.get("missing_values", metrics_data.get("missing", 0))
duplicate_rows = metrics_data.get("duplicate_rows", 36727)

st.session_state["dataset_available"] = True

# Safely load health status counts whether they are stored as strings or numbers
raw_target_counts = metrics_data.get(
    "health_status_counts",
    metrics_data.get(
        "target_counts",
        {
            "No Diabetes": 374597,
            "Diabetes": 68492,
            "Prediabetes": 10242,
        },
    ),
)

target_names_map = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
processed_counts = {}
for k, v in raw_target_counts.items():
  if str(k).isdigit():
    key_str = target_names_map.get(int(k), str(k))
  else:
    key_str = str(k)
  processed_counts[key_str] = int(v)

target_counts = pd.Series(processed_counts)
total_samples = rows

m1, m2, m3, m4 = st.columns([2, 1, 1, 2])
m1.metric(
    label="Cleaned Dataset Rows",
    value=f"{rows:,}",
    delta="-36,727 Duplicates",
    delta_color="off",
)

m2.metric(label="Features", value=f"{cols:,}")

m3.metric(label="Missing Values", value=f"{missing:,}")

m4.metric(
    label="Data Hygiene Status",
    value="100% Clean",
    delta="Ready for EDA",
    delta_color="normal",
)


with st.expander("📊 View Target Class Distribution & Percentages"):
  total_valid_samples = (
      target_counts.sum()
  )  # Ensures percentages strictly sum to 100.00%

  target_df = pd.DataFrame(
      {
          "Health Status": target_counts.index,
          "Patient Count": target_counts.values,
          "Percentage (%)": [
              (v / total_valid_samples) * 100 for v in target_counts.values
          ],
      }
  )
  target_df["Percentage (%)"] = target_df["Percentage (%)"].map("{:.2f}%".format)

  st.dataframe(
      target_df,
      hide_index=True,
      use_container_width=True,
      column_config={"Percentage (%)": st.column_config.Column(alignment="right")},
  )

  



st.markdown("### 🎯 Project Objectives")

st.markdown(
    """
<style>
.badge-card {
    background-color: #1E1E2E;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 15px;
}
.badge-title {
    font-size: 16px;
    font-weight: bold;
    color: #F5E0DC;
    margin-bottom: 10px;
}
.tag {
    display: inline-block;
    background: #313244;
    color: #89B4FA;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    margin: 3px;
    font-weight: 500;
}
</style>

<div style="display: flex; gap: 15px;">
    <div class="badge-card" style="flex: 1;">
        <div class="badge-title">🛠 Data Engineering & Hygiene</div>
        <span class="tag">BRFSS 2015 & 2021 Merged Cohort Dataset</span>
        <span class="tag">Duplicate Removal (36,727 dropped)</span>
        <span class="tag">Raw Dataset (490,058 rows, 23 columns)</span>
        <span class="tag">Data Hygiene Status 100% clean</span>
        <span class="tag">453,331 Clean Records</span>
        <span class="tag">0 Missing Values Audit</span>
    </div>
    <div class="badge-card" style="flex: 1;">
        <div class="badge-title">🔍 Epidemiological & Statistical EDA</div>
        <span class="tag">BMI & Weight Categories</span>
        <span class="tag">Pre-rendered High-Res Assets</span>
        <span class="tag">Blood Pressure & Cholesterol Risk</span>
        <span class="tag">Socioeconomic Trends (Income/Education)</span>
        <span class="tag">Mutual Information Ranking</span>
        <span class="tag">Correlation Heatmaps </span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)



st.markdown("### 🧾 Dataset Specification & Structure")
st.markdown(
    """
    | Dimension / Item | Specification & Details |
    |---|---|
    | **Combined Cohort** | BRFSS Diabetes Health Indicators (`2015 & 2021 Merged`) |
    | **Total Clean Records** | `453,331` rows after rigorous deduplication (`36,727` duplicates removed) |
    | **Source Context** | CDC Behavioral Risk Factor Surveillance System (Multi-Year Survey Cycles) |
    | **Target Field (`Diabetes_012`)** | `0`: No Diabetes, `1`: Prediabetes, `2`: Gestational / Established Diabetes |
    | **Feature Space** | 25 Attributes spanning clinical markers, BMI, lifestyle, and socioeconomic indicators |
    | **Analytical Scope** | Epidemiological Exploratory Data Analysis & Feature Relationship Modeling |
    """
)

st.info(
    """⚠️ **Analytical Note on Class Imbalance:** 
    The raw combined dataset exhibits a high degree of class imbalance across the target categories (`Diabetes_012`), with a dominant majority of non-diabetes records (`0`) compared to prediabetes (`1`) and active diabetes (`2`). """
    "Note: All exploratory data analysis (EDA) is performed on the cleaned raw dataset (`453,331 rows`)."
)



st.markdown("### 🔄 Comprehensive EDA Workflow")
st.code("""
    BRFSS 2015 + 2021 Raw CSVs
        ↓
    Multi-Year Data Merging & Year Annotation
        ↓
    Data Quality Audit (Missing Values & Duplicate Removal)
        ↓
    Univariate & Multivariate Feature Exploration
        ↓
    Socioeconomic & Epidemiological Pattern Analysis
        ↓
    Correlation Heatmaps & Mutual-Information Ranking
        ↓
    Interactive Streamlit Visualization & Asset Browsing
    """,
    language="text",
)




st.markdown("### 📊 What You Can Explore")
features_right, features_left = st.columns(2)

with features_right:
    st.markdown(
        """
        **📈 Relationships, Demographics & Models**        
        - At-risk population breakdowns: Age, income, binary features, and combined profiles (`AtRisk_by_Age.png`, `AtRisk_by_Age_Income.png`, `AtRisk_by_Binary_Features.png`, `AtRisk_by_Combined_Profile.png`, `AtRisk_by_General_Health.png`, `BMI_vs_Health_Status_Scatter_Trend.png`)
        - Comparative cross-feature heatmaps: Blood pressure (BP), cholesterol, and walking difficulty vs. diabetes status (`HighBP_vs_Diabetes_Heatmap.png`, `HighChol_vs_Diabetes_Heatmap.png`, `DiffWalk_vs_Diabetes_Heatmap.png`)
        - BMI vs. diabetes boxplots and heatmaps (`BMI_vs_Diabetes_Boxplot.png`, `BMI_vs_Diabetes_Heatmap.png`)
        - Demographic factors: Age, education, income, and sex distributions (`Age.png`, `Education.png`, `Income.png`, `Sex.png`)
        - Socioeconomic trends (`Socioeconomic_Diabetes_Trend.png`, `Socioeconomic_HeartDisease_Trend.png`)
        - Feature ranking, correlations, and predictors (`Correlation_Heatmap.png`, `Correlation_Heatmap_Predictors.png`, `Mutual_Information_Ranking.png`, `Mutual_Information_Health_Status.png`)
        - Model performance confusion matrices (`confusion_matrix_catboost.png`, `confusion_matrix_lightgbm.png`, `confusion_matrix_xgb.png`, `confusion_matrix_model1_raw.png`, `confusion_matrix_model2_balanced.png`, `confusion_matrix_model3_smote.png`)
        """
    )

    
with features_left:
    st.markdown(
        """
        **🩺 Health & Lifestyle Indicators**
        - High blood pressure & high cholesterol (`HighBP.png`, `HighChol.png`)
        - Cholesterol check frequency (`CholCheck.png`)
        - BMI distributions & clinical categories (`BMI.png`, `BMI_Categories.png`, `AtRisk_Percent_by_BMI.png`)
        - General, physical, and mental health days (`GenHlth.png`, `PhysHlth.png`, `MentHlth.png`)
        - Health status raw distributions & yearly comparisons (`Health_Status_Raw_Distribution.png`, `Health_Status_Raw_Distribution_Pie.png`, `Health_Status_by_Year.png`)
        - Behavioral lifestyle factors: Smoking, physical activity, diet, and alcohol consumption (`Smoker.png`, `PhysActivity.png`, `Fruits.png`, `Veggies.png`, `HvyAlcoholConsump.png`)
        - Clinical conditions: Stroke, heart disease/attack, and walking difficulty (`Stroke.png`, `HeartDiseaseorAttack.png`, `DiffWalk.png`, `AnyHealthcare.png`, `NoDocbcCost.png`)
        - Target class distribution (`Diabetes_012_Label.png`, `Diabetes_012_Label_Pie.png`)
        """
    )

    st.image("https://plus.unsplash.com/premium_photo-1778086819134-9d3a234956c7?q=80&w=880&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D", width = 270)






st.markdown("### 🛠 Technologies & Tools Used")
col1, col2 = st.columns(2)
with col1:
  st.markdown("""
        #### 📊 Data Manipulation & Preprocessing
        * **Python** - High-level, general-purpose programming language 
        * **Pandas & NumPy** – Data cleaning, merging, and vectorization
        * **Scikit-Learn** – Feature scaling, encoding, and data splitting
        * **Imbalanced-learn** – SMOTE oversampling for class imbalance

        #### 📈 Visualization & Asset Management
        * **Matplotlib & Seaborn** – Statistical plots and custom charting
        * **Static PNGs & JSON** – Pre-rendered visual assets for optimized loading
        """)

with col2:
  st.markdown("""
        #### 🧠 Machine Learning & Deep Learning
        * **TensorFlow & Keras** – Deep learning Artificial Neural Network (ANN) pipelines
        * **XGBoost, CatBoost, LightGBM** – Gradient-boosted classification models
        * **Scikit-Learn Metrics** – Confusion matrices and classification reports

        #### 🔀 Dashboard & Version Control
        * **Streamlit** – Interactive web application framework
        * **Git / GitHub** – Codebase and version management
        """)

  


st.markdown("### ⛓️ Project Structure")
st.code(
    """
    ANN- health-risk-prediction-project/
    │
    ├── Assets/
    │   ├── Age.png
    │   ├── AnyHealthcare.png
    │   ├── AtRisk_by_Age.png
    │   ├── AtRisk_by_Age_Income.png
    │   ├── AtRisk_by_Binary_Features.png
    │   ├── AtRisk_by_Combined_Profile.png
    │   ├── AtRisk_by_General_Health.png
    │   ├── AtRisk_Percent_by_BMI.png
    │   ├── BMI.png
    │   ├── BMI_Categories.png
    │   ├── BMI_vs_Diabetes_Boxplot.png
    │   ├── BMI_vs_Diabetes_Heatmap.png
    │   ├── BMI_vs_Health_Status_Scatter_Trend.png
    │   ├── CholCheck.png
    │   ├── confusion_matrix_catboost.png
    │   ├── confusion_matrix_lightgbm.png
    │   ├── confusion_matrix_model1_raw.png
    │   ├── confusion_matrix_model2_balanced.png
    │   ├── confusion_matrix_model3_smote.png
    │   ├── confusion_matrix_xgb.png
    │   ├── Correlation_Heatmap.png
    │   ├── Correlation_Heatmap_Predictors.png
    │   ├── Diabetes_012_Label.png
    │   ├── Diabetes_012_Label_Pie.png
    │   ├── DiffWalk.png
    │   ├── DiffWalk_vs_Diabetes_Heatmap.png
    │   ├── Education.png
    │   ├── Fruits.png
    │   ├── GenHlth.png
    │   ├── Health_Status_by_Year.png
    │   ├── Health_Status_Raw_Distribution.png
    │   ├── Health_Status_Raw_Distribution_Pie.png
    │   ├── HeartDiseaseorAttack.png
    │   ├── HighBP.png
    │   ├── HighBP_vs_Diabetes_Heatmap.png
    │   ├── HighChol.png
    │   ├── HighChol_vs_Diabetes_Heatmap.png
    │   ├── HvyAlcoholConsump.png
    │   ├── Income.png
    │   ├── MentHlth.png
    │   ├── Mutual_Information_Health_Status.png
    │   ├── Mutual_Information_Ranking.png
    │   ├── NoDocbcCost.png
    │   ├── PhysActivity.png
    │   ├── PhysHlth.png
    │   ├── Sex.png
    │   ├── Smoker.png
    │   ├── Socioeconomic_Diabetes_Trend.png
    │   ├── Socioeconomic_HeartDisease_Trend.png
    │   ├── Stroke.png
    │   └── Veggies.png
    |
    ├── Data/
    │   ├── diabetes_012_health_indicators_BRFSS2015.csv
    │   ├── diabetes_012_health_indicators_BRFSS2015_BRFSS2021.csv
    │   └── diabetes_012_health_indicators_BRFSS2021.csv
    │
    ├── DOC/
    │   └── Diabetes_EDA_Insights_Report_Final.pdf
    │
    ├── Models/
    │   ├── catboost_model.pkl
    │   ├── home_metrics.json
    │   ├── lightgbm_model.pkl
    │   ├── model_catboost_evaluations.json
    │   ├── model_evaluations.json
    │   ├── model_lightgbm_evaluations.json
    │   ├── model_xgb_evaluations.json
    │   ├── model1_raw_weights.keras
    │   ├── model2_downsampled.keras
    │   ├── model3_smote.keras
    │   ├── scaler_preprocessor.pkl
    │   └── xgb_model.pkl
    │
    ├── Notebooks/
    │   ├── dataset-combining.ipynb
    │   ├── EDA.ipynb
    │   ├── ML-Algorithm.ipynb
    │   └── Models_Run_fair_comparison.ipynb
    │
    ├── Pages/                    
    │   ├── eda.py
    │   ├── home.py                
    │   ├── prediction.py
    │   └── report.py
    │
    ├── Structure/
    │   └── structure.txt
    │
    ├── .gitignore
    ├── .python-version
    ├── app.py
    ├── README.md
    ├── requirements.txt
    └── runtime.txt
    """,
    language="text",
)



st.info(
    "💡 **Getting Started:** Use the sidebar navigation menu to switch between "
    "the **EDA Explorer**, **Machine Learning Models**, and **Deep Learning Inference** modules."
)





