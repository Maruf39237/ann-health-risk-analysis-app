import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import textwrap
from tensorflow import keras
import matplotlib.pyplot as plt

st.markdown("""
<style>
div.stButton > button {
    background-color: #dc2626;
    color: white;
    border: none;
    border-radius: 7px;
    padding: 0.35rem 1rem;
    min-height: 36px;
    height: 36px;
    font-size: 14px;
    font-weight: 600;
    width: 180px;
    transition: background-color 0.2s ease;
}

div.stButton > button:hover {
    background-color: #b91c1c;
    color: white;
    border: none;
}

div.stButton > button:focus:not(:active) {
    color: white;
    border: 1px solid #f87171;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.block-container { padding-top: 1.35rem; padding-bottom: 1.8rem; }
h1 { letter-spacing: -0.035em; font-weight: 750; }
h2, h3 { letter-spacing: -0.02em; }
[data-testid="stMetric"] {
  border: 1px solid rgba(120, 135, 155, 0.22);
  border-radius: 12px;
  padding: 0.65rem 0.8rem;
  background: rgba(120, 135, 155, 0.055);
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; }
[data-testid="stMetricValue"] { font-size: 1.45rem; }
[data-testid="stDataFrame"] { border: 1px solid rgba(120, 135, 155, 0.20); border-radius: 10px; }
div[data-testid="stExpander"] { border: 1px solid rgba(120, 135, 155, 0.20); border-radius: 10px; }
.stButton > button { border-radius: 9px; font-weight: 600; }
[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)





BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Models"
DATA_DIR  = BASE_DIR / "Data"

FEATURE_ORDER = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

BINARY = {
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"
}

AGE_MAP = {1:"18-24",2:"25-29",3:"30-34",4:"35-39",5:"40-44",6:"45-49",
           7:"50-54",8:"55-59",9:"60-64",10:"65-69",11:"70-74",12:"75-79",13:"80+"}
EDU_MAP = {1:"Never / kindergarten",2:"Grades 1-8",3:"Grades 9-11",
           4:"Grade 12 / GED",5:"College 1-3 yrs",6:"College 4+ yrs"}
INC_MAP = {1:"<$10k",2:"$10-15k",3:"$15-20k",4:"$20-25k",
           5:"$25-35k",6:"$35-50k",7:"$50-75k",8:"≥$75k"}
GEN_MAP = {1:"Excellent",2:"Very good",3:"Good",4:"Fair",5:"Poor"}




@st.cache_resource(show_spinner="Loading models…")
def load_models():
    models = {
        "ANN (SMOTE)": keras.models.load_model(
            MODEL_DIR / "model3_smote.h5", compile=False
        ),
        "ANN (Class Weights)": keras.models.load_model(
            MODEL_DIR / "model1_raw_weights.h5", compile=False
        ),
        "ANN (Downsampled)": keras.models.load_model(
            MODEL_DIR / "model2_downsampled.h5", compile=False
        ),
        "XGBoost": joblib.load(MODEL_DIR / "xgb_model.pkl"),
        "CatBoost": joblib.load(MODEL_DIR / "catboost_model.pkl"),
        "LightGBM": joblib.load(MODEL_DIR / "lightgbm_model.pkl"),
    }

    preprocessor = joblib.load(MODEL_DIR / "scaler_preprocessor.pkl")
    return models, preprocessor



@st.cache_data
def load_metrics():
    m = {}
    with open(MODEL_DIR / "model_evaluations.json") as f:
        ann = json.load(f)
        m["ANN (SMOTE)"] = ann["Model 3 (SMOTE)"]
        m["ANN (Class Weights)"] = ann["Model 1 (Raw + Class Weights)"]
        m["ANN (Downsampled)"] = ann["Model 2 (Downsampled 50:50)"]
        
    with open(MODEL_DIR / "model_xgb_evaluations.json") as f:
        m["XGBoost"] = json.load(f)
    with open(MODEL_DIR / "model_catboost_evaluations.json") as f:
        m["CatBoost"] = json.load(f)
    with open(MODEL_DIR / "model_lightgbm_evaluations.json") as f:
        m["LightGBM"] = json.load(f)
    return m

@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "diabetes_012_health_indicators_BRFSS2015_BRFSS2021.csv")

def to_int(val, is_binary=False, is_sex=False):
    if is_binary:
        return 1 if val == "Yes" else 0
    if is_sex:
        return 1 if val == "Male" else 0
    return val

def predict(user_dict, model_name, models, preprocessor):
    df = pd.DataFrame([user_dict])[FEATURE_ORDER]
    model = models[model_name]
    if model_name.startswith("ANN"):
        X = preprocessor.transform(df)
        proba = float(model.predict(X, verbose=0).ravel()[0])
    else:
        proba = float(model.predict_proba(df)[0][1])
    label = "At Risk" if proba >= 0.5 else "Healthy"
    return label, proba




# ----------
st.divider()
c1, c2 = st.columns(2)
with c1 : 
    st.title("Health Risk Prediction 🎯")
    st.markdown(
    """
    ##### Predict whether a person is **Healthy** or **At Risk** using six different models (three Neural Nets + three Gradient-Boosting models)
    """)

with c2 :
    st.image("https://plus.unsplash.com/premium_photo-1682309772037-8da49abb2b84?q=80&w=1212&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D", caption="Understanding risk starts with data.")

st.markdown(""" ##### All models were trained on the same stratified split of the imbalanced BRFSS(`2015 + 2021`) dataset (`~453k records, ~83% Healthy, ~17% At Risk`) so you can compare how different imbalance-handling strategies perform. This page is for educational and model-comparison purposes only, not for medical advice.""")

st.divider()





st.subheader("ℹ️ Model Specifications & Performance Details")
with st.expander("View Comprehensive Technical Specifications, Tables & Textual Breakdown", expanded=False):
    st.markdown("### Table 1: Technical Specifications and Imbalance Strategies 📋 ")
    
    spec_df = pd.DataFrame({
        "Model Name": [
            "1. ANN (Raw + Class Weights)",
            "2. ANN (Downsampled 50:50)",
            "3. ANN (SMOTE)",
            "4. XGBoost",
            "5. CatBoost",
            "6. LightGBM"
        ],
        "Algorithm Family": [
            "Multi-Layer Perceptron", "Multi-Layer Perceptron", "Multi-Layer Perceptron",
            "Gradient Boosted Trees", "Gradient Boosted Trees", "Gradient Boosted Trees"
        ],
        "Framework": [
            "TensorFlow / Keras", "TensorFlow / Keras", "TensorFlow / Keras + Imbalanced-learn",
            "xgboost", "catboost", "lightgbm"
        ],
        "Imbalance Handling": [
            "Computed Loss Class Weights (compute_class_weight)",
            "50:50 Random Undersampling (Training partition only)",
            "Synthetic Minority Over-sampling Technique (SMOTE)",
            "scale_pos_weight ratio tuning",
            "auto_class_weights='Balanced'",
            "class_weight='balanced'"
        ],
        "Key Architecture / Hyperparameters": [
            "Dense Layers, ReLU, Batch Normalization, Dropout, Adam Optimizer",
            "Dense Layers, ReLU, Batch Normalization, Dropout, Adam Optimizer",
            "Dense Layers, ReLU, Batch Normalization, Dropout, Adam Optimizer",
            "Gradient boosted decision trees, max depth optimized for tabular survey data",
            "Symmetric decision trees, ordered boosting to prevent target leakage",
            "Leaf-wise tree growth, histogram-based split optimization"
        ]
    })
    st.dataframe(spec_df, use_container_width=True, hide_index=True)

    st.markdown("### Table 2: Complete Performance Metrics Matrix 📊 ")
    
    
    evals = load_metrics()
    table_models = [
        ("1. ANN (Raw + Class Weights)", "ANN (Class Weights)"),
        ("2. ANN (Downsampled 50:50)", "ANN (Downsampled)"),
        ("3. ANN (SMOTE)", "ANN (SMOTE)"),
        ("4. XGBoost", "XGBoost"),
        ("5. CatBoost", "CatBoost"),
        ("6. LightGBM", "LightGBM"),
    ]
    metric_rows = []
    for display_name, key in table_models:
        report = evals[key]
        metric_rows.append({
            "Model": display_name,
            "Accuracy": f"{report.get('accuracy', 0):.1%}",
            "At-Risk Precision": f"{report.get('At Risk', {}).get('precision', 0):.3f}",
            "At-Risk Recall": f"{report.get('At Risk', {}).get('recall', 0):.3f}",
            "At-Risk F1": f"{report.get('At Risk', {}).get('f1-score', 0):.3f}",
            "Healthy Precision": f"{report.get('Healthy', {}).get('precision', 0):.3f}",
            "Healthy Recall": f"{report.get('Healthy', {}).get('recall', 0):.3f}",
            "Healthy F1": f"{report.get('Healthy', {}).get('f1-score', 0):.3f}",
            "Macro Avg F1": f"{report.get('macro avg', {}).get('f1-score', 0):.3f}",
        })
    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.caption("Values are read from the saved test-set classification reports; displayed values are rounded.")

    st.markdown(textwrap.dedent(
    """
    ### 📑 Detailed Textual Breakdown per Model

    1. **ANN (Raw + Class Weights)**
       - **Description:** A fully connected neural network trained directly on the original imbalanced tabular distribution.
       - **Mechanism:** To overcome the severe class imbalance, class weights are calculated inversely proportional to class frequencies and passed directly into the categorical cross-entropy loss function during gradient descent.
       - **Behavior:** Yields the highest At-Risk Recall (77%), making it the most aggressive model for capturing true positive pre-diabetic instances, though it yields slightly lower overall accuracy (70%) due to increased false positives among healthy cases.

    2. **ANN (Downsampled 50:50)**
       - **Description:** Identical neural network architecture trained on an artificially balanced dataset created by randomly downsampling the majority ("Healthy") class to match the minority class count.
       - **Mechanism:** Rigorously restricts the downsampling procedure exclusively to the training split (`xtrain_df1`) to prevent data leakage into validation and test folds.
       - **Behavior:** Achieves a balanced accuracy of 71% with a solid At-Risk Recall of 76% and a Healthy F1-Score of 0.80.

    3. **ANN (SMOTE)**
       - **Description:** Combines the standard multi-layer perceptron architecture with the Synthetic Minority Over-sampling Technique.
       - **Mechanism:** Synthesizes new feature vectors in the feature space for the minority class rather than dropping majority records.
       - **Behavior:** Reaches the highest overall Accuracy (72%) and the best balanced performance for the Healthy class (72% recall, 0.81 F1-score), while maintaining a competitive At-Risk F1-score of 0.48.

    4. **XGBoost (Extreme Gradient Boosting)**
       - **Description:** An optimized tree-based gradient boosting model built sequentially to minimize residual errors.
       - **Mechanism:** Utilizes the `scale_pos_weight` hyperparameter to adjust gradient boosting weights for the positive class.
       - **Behavior:** Delivers robust and stable tabular classification (71% accuracy, 76% recall) with rapid training times and minimal feature engineering requirements.

    5. **CatBoost (Categorical Boosting)**
       - **Description:** A gradient boosting framework designed for handling tabular structures using symmetric trees and ordered boosting.
       - **Mechanism:** Utilizes `auto_class_weights='Balanced'` to dynamically scale penalties during tree splitting.
       - **Behavior:** Matches XGBoost and LightGBM across all major evaluation metrics (71% accuracy, 0.48 At-Risk F1), providing strong resistance to overfitting on survey indicators.

    6. **LightGBM (Light Gradient Boosting Machine)**
       - **Description:** A high-speed, distributed gradient boosting algorithm utilizing leaf-wise tree growth.
       - **Mechanism:** Configured with `class_weight='balanced'` to assign higher loss penalties to misclassifications of the minority class during histogram-based split finding.
       - **Behavior:** Performs identically to the other gradient boosters, yielding 71% overall accuracy and reliable calibration for tabular patient assessments.
    """
    ))



df = load_data()
st.subheader("📁 Dataset preview")
# st.dataframe(df.head(), use_container_width=True)
st.dataframe(df.tail(), use_container_width=True)

models, preprocessor = load_models()
metrics = load_metrics()

st.subheader("Select model")
model_choice = st.selectbox(
    "Model",
    list(models.keys()),
    index=0,
    help="Neural models first (project focus)"
)

m = metrics[model_choice]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{m['accuracy']:.3f}")
c2.metric("At-Risk Precision", f"{m['At Risk']['precision']:.3f}")
c3.metric("At-Risk Recall", f"{m['At Risk']['recall']:.3f}")
c4.metric("At-Risk F1", f"{m['At Risk']['f1-score']:.3f}")

desc = {
    "ANN (SMOTE)": "Highest overall accuracy. Best single model for general use.",
    "ANN (Class Weights)": "Highest At-Risk recall among neural nets. Prefer when missing an At-Risk case is costly.",
    "ANN (Downsampled)": "Trained on balanced data. Middle ground.",
    "XGBoost": "Strongest and most reliable tree model.",
    "CatBoost": "Very close to XGBoost, robust on mixed features.",
    "LightGBM": "Fastest tree model, almost identical metrics."
}
st.info(desc[model_choice])




if "profile" not in st.session_state:
    st.session_state.profile = {f: 0 for f in FEATURE_ORDER}
    st.session_state.profile.update({
        "BMI": 28.0, "GenHlth": 1, "Age": 1, "Education": 1, "Income": 1
    })
if "true_label" not in st.session_state:
    st.session_state.true_label = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


st.caption("After editing the manual form, click **Use these values** to apply those inputs to the prediction profile.")
tab1, tab2, tab3 = st.tabs(["Manual Entry", "Load Random Person", "Paste Feature List"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.selectbox("Have you been told you have high blood pressure?", ["No","Yes"], key="HighBP")
        st.selectbox("Have you been told you have high cholesterol?", ["No","Yes"], key="HighChol")
        st.selectbox("Cholesterol check in the past 5 years?", ["No","Yes"], key="CholCheck")
        st.number_input("Body Mass Index (BMI)", 12.0, 60.0, 28.0, 0.1, key="BMI")
        st.selectbox("Smoked at least 100 cigarettes in your life?", ["No","Yes"], key="Smoker")
        st.selectbox("Ever told you had a stroke?", ["No","Yes"], key="Stroke")
        st.selectbox("Ever told you had coronary heart disease or heart attack?", ["No","Yes"], key="HeartDiseaseorAttack")
        st.selectbox("Physical activity in past 30 days (not job)?", ["No","Yes"], key="PhysActivity")
        st.selectbox("Consume fruit 1 or more times per day?", ["No","Yes"], key="Fruits")
        st.selectbox("Consume vegetables 1 or more times per day?", ["No","Yes"], key="Veggies")
        st.selectbox("Heavy alcohol drinker?", ["No","Yes"], key="HvyAlcoholConsump")
    with right:
        st.selectbox("Any kind of health care coverage?", ["No","Yes"], key="AnyHealthcare")
        st.selectbox("Needed doctor but could not because of cost (past 12 months)?", ["No","Yes"], key="NoDocbcCost")
        st.selectbox("General health", list(GEN_MAP.keys()), format_func=lambda x: GEN_MAP[x], key="GenHlth")
        st.slider("Days mental health not good (past 30 days)", 0, 30, 0, key="MentHlth")
        st.slider("Days physical health not good (past 30 days)", 0, 30, 0, key="PhysHlth")
        st.selectbox("Serious difficulty walking or climbing stairs?", ["No","Yes"], key="DiffWalk")
        st.selectbox("Sex", ["Female","Male"], key="Sex")
        st.selectbox("Age group", list(AGE_MAP.keys()), format_func=lambda x: AGE_MAP[x], key="Age")
        st.selectbox("Education level", list(EDU_MAP.keys()), format_func=lambda x: EDU_MAP[x], key="Education")
        st.selectbox("Income level", list(INC_MAP.keys()), format_func=lambda x: INC_MAP[x], key="Income")

    if st.button("Use these values", key="use_manual"):
        st.session_state.profile = {
            f: to_int(st.session_state[f], f in BINARY, f=="Sex") for f in FEATURE_ORDER
        }
        st.session_state.true_label = None
        st.session_state.last_prediction = None
        st.success("Profile loaded from manual entry")

with tab2:
    if st.button("Load random person"):
        row = df.sample(1).iloc[0]
        st.session_state.profile = {
            f: float(row[f]) if f == "BMI" else int(row[f]) for f in FEATURE_ORDER
        }
        st.session_state.true_label = "At Risk" if row["Health_Status"] == 1 else "Healthy"
        st.session_state.last_prediction = None
        st.success("Random person loaded – true status is hidden")

with tab3:
    st.caption("Paste a Python list or comma-separated values (exactly 21 numbers)")
    example = "[0, 1, 1, 28.5, 0, 0, 0, 1, 1, 1, 0, 1, 0, 3, 2, 5, 0, 0, 9, 5, 6]"
    text = st.text_area("Feature values", value=example, height=80)
    if st.button("Parse & load"):
        try:
            clean = text.strip().lstrip("[").rstrip("]")
            vals = [float(x.strip()) for x in clean.split(",")]
            if len(vals) != 21:
                st.error("Need exactly 21 values")
            else:
                st.session_state.profile = dict(zip(FEATURE_ORDER, vals))
                st.session_state.true_label = None
                st.session_state.last_prediction = None
                st.success("List loaded")
        except Exception:
            st.error("Could not parse. Check the format.")


if st.session_state.profile:
    with st.expander("Inspect loaded profile (raw values)", expanded=False):
        raw_df = pd.DataFrame([st.session_state.profile])[FEATURE_ORDER]
        st.dataframe(raw_df, use_container_width=True)

    if model_choice.startswith("ANN"):
        with st.expander("Preprocessed values (what the ANN actually sees)", expanded=False):
            raw_df = pd.DataFrame([st.session_state.profile])[FEATURE_ORDER]
            scaled_array = preprocessor.transform(raw_df)
            scaled_cols = (
                ['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income'] +
                ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
            )
            scaled_df = pd.DataFrame(scaled_array, columns=scaled_cols)
            scaled_df = scaled_df[FEATURE_ORDER]
            st.dataframe(scaled_df.round(4), use_container_width=True)
            st.caption("Only BMI, MentHlth, PhysHlth, GenHlth, Age, Education, Income are scaled. Binary features remain 0/1.")



st.divider()

_, predict_col, _ = st.columns([1, 1, 1])
with predict_col:
    if st.button("Predict", type="primary"):
        label, proba = predict(
            st.session_state.profile, model_choice, models, preprocessor
        )
        st.session_state.last_prediction = {
            "label": label, "proba": proba, "model": model_choice
        }


st.caption(
    "💡 Using a randomly loaded person? After predicting, "
    "click 'Reveal True Status' below to check the actual answer."
)


if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    label = res["label"]
    proba = res["proba"]
    confidence = proba if label == "At Risk" else (1 - proba)

    is_at_risk = label == "At Risk"
    status_color = "#b42332" if is_at_risk else "#16745b"
    st.markdown(
        f"""
        <div style="border:1px solid rgba(120,135,155,.28); border-left:5px solid {status_color};
                    border-radius:12px; padding:12px 16px; text-align:center; margin:4px 0 12px;
                    background:rgba(120,135,155,.045);">
            <div style="font-size:11px; font-weight:700; letter-spacing:1.2px; opacity:.72;">PREDICTED STATUS</div>
            <div style="font-size:24px; line-height:1.25; font-weight:750; color:{status_color}; margin:4px 0;">{label}</div>
            <div style="font-size:14px;">Model confidence <b>{confidence*100:.1f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if is_at_risk:
        st.warning(f"The model estimates a **{proba*100:.1f}%** probability of being **At Risk**.")
    else:
        st.success(f"The model estimates a **{(1-proba)*100:.1f}%** probability of being **Healthy**.")

    st.markdown("#### Probability of At Risk across models")
    all_probs = {
        name: (proba if name == res["model"] else predict(st.session_state.profile, name, models, preprocessor)[1])
        for name in models
    }

    comparison_rows = []
    for name, probability in all_probs.items():
        comparison_rows.append({
            "Model": name,
            "P(At Risk)": probability,
            "Predicted status": "At Risk" if probability >= 0.5 else "Healthy",
        })
    comparison_df = pd.DataFrame(comparison_rows)
    display_df = comparison_df.copy()
    display_df["P(At Risk)"] = display_df["P(At Risk)"].map(lambda value: f"{value*100:.1f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


    dark_theme = str(st.get_option("theme.base")).lower() == "dark"
    text_color = "#e2e8f0" if dark_theme else "#334155"
    grid_color = "#475569" if dark_theme else "#dbe3ee"
    selected_color = "#2563eb"
    other_color = "#aab7c8"
    risk_color = "#d9485f"
    healthy_color = "#2a9d8f"

    st.caption("The dashed 50% line is the current classification threshold used by all six models.")
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    values = comparison_df["P(At Risk)"].to_numpy(dtype=float) * 100
    names = comparison_df["Model"].tolist()
    bar_colors = [selected_color if name == res["model"] else other_color for name in names]
    bars = ax.barh(names, values, color=bar_colors, height=0.62, zorder=3)
    ax.axvline(50, color=risk_color, linestyle="--", linewidth=1.4, alpha=0.9, zorder=2)
    ax.set_xlim(0, 108)
    ax.set_xlabel("At-risk probability (%)", color=text_color, labelpad=8)
    ax.set_ylabel("")
    ax.set_title("Model probability comparison", loc="left", color=text_color, fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.grid(True, color=grid_color, linewidth=0.7, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=text_color, labelsize=9)
    ax.tick_params(axis="y", colors=text_color, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(min(value + 1.2, 103), bar.get_y() + bar.get_height()/2,
                f"{value:.1f}%", va="center", ha="left", fontsize=9,
                color=text_color, fontweight="bold")
    ax.invert_yaxis()  
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("#### Predicted target split")
    st.caption("This chart shows the selected model's two class probabilities for this profile—not the dataset's class distribution.")
    
    healthy_probability = max(0.0, 1.0 - proba)
    risk_probability = max(0.0, proba)
    target_values = [healthy_probability, risk_probability]
    healthy_color = "#2E8B57"   
    risk_color = "#9A3412"     

    fig, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=(5.8, 3),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.02},
    )
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    legend_ax.set_facecolor("none")
    ax.pie(
        target_values,
        colors=[healthy_color, risk_color],
        startangle=90,
        counterclock=False,
        radius=0.92,
        wedgeprops={"linewidth": 1.5, "edgecolor": "#FFFFFF" if not dark_theme else "#1F2937"},
    )
    ax.set_aspect("equal")
    ax.axis("off")

    
    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.scatter([0.08], [0.64], s=46, color=healthy_color, clip_on=False)
    legend_ax.text(0.19, 0.64, "Healthy", va="center", ha="left", color=text_color, fontsize=10)
    legend_ax.text(0.19, 0.49, f"{healthy_probability * 100:.1f}%", va="center", ha="left", color=text_color, fontsize=9)
    legend_ax.scatter([0.08], [0.28], s=46, color=risk_color, clip_on=False)
    legend_ax.text(0.19, 0.28, "At Risk", va="center", ha="left", color=text_color, fontsize=10)
    legend_ax.text(0.19, 0.13, f"{risk_probability * 100:.1f}%", va="center", ha="left", color=text_color, fontsize=9)

    fig.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)



if (
    st.session_state.true_label is not None
    and st.session_state.last_prediction is not None
):
    st.caption(
        "Now check whether the model's prediction matches "
        "this person's actual health status."
    )

    if st.button("🔍 Reveal True Status", type="primary"):
        true = st.session_state.true_label
        pred = st.session_state.last_prediction["label"]

        if true == pred:
            st.success(f"True status: **{true}** — Model was correct ✅")
        else:
            st.error(f"True status: **{true}** — Model was incorrect ❌")




            