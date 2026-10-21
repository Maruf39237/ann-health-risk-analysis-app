from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Diabetes EDA Explorer",
    page_icon="🧮",
    # layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "Assets"

st.divider()

col1, col2 = st.columns([2.5, 2.1])
with col1:
    st.title("Diabetes EDA Explorer 🧮")
    st.markdown("""
### - 453k+ Records Analyzed 📊
### - Biometric Risk Factors 🩺 
### - Exported EDA Visuals 📈  
""")

with col2:
    st.image(
        "https://images.unsplash.com/photo-1653912124332-615b347ab357?q=80&w=1074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D",
        caption="",
        width=300
    )
    st.markdown("""
    ###### Explore visual distributions of Blood Pressure (BP), Cholesterol, BMI, and habits from the combined `BRFSS records (2015 & 2021)`.
    """)

st.divider()

st.markdown(
    """
    ### **Module Overview & Epidemiological Insights:**
    
    - ##### This Exploratory Data Analysis module delivers an in-depth examination of public health survey patterns derived from 453,331 cleaned records across the combined BRFSS 2015 and 2021 cohorts. 
    - ##### It interactively visualizes critical biometric risk factors—such as body mass index (BMI) brackets, blood pressure, cholesterol levels, and walking difficulties and maps their direct statistical correlation with diabetes categories. 
    - ##### Through multi-variable heatmaps, mutual information feature rankings, and behavioral distribution plots, users can uncover how socioeconomic and lifestyle gradients influence population health outcomes. 
    - ##### The section also integrates baseline and advanced model confusion matrices to evaluate class-imbalanced predictions side-by-side with exploratory insights. 
    - ##### For exhaustive statistical narratives, granular multi-year breakdowns, and complete methodological documentation, please navigate to the dedicated **Report** page via the sidebar navigation menu.
    """
)

CHARTS = {
    "Target & Overview": {
        "Diabetes class distribution (Bar)": ("Diabetes_012_Label.png", "Distribution of No Diabetes, Prediabetes, and Diabetes (Bar chart)."),
        "Diabetes class distribution (Pie)": ("Diabetes_012_Label_Pie.png", "Distribution of No Diabetes, Prediabetes, and Diabetes (Pie chart)."),
        "Correlation heatmap": ("Correlation_Heatmap.png", "Pairwise correlations among numeric health indicators."),
        "Correlation heatmap (Predictors)": ("Correlation_Heatmap_Predictors.png", "Predictor-only correlation heatmap."),
        "Mutual information feature ranking": ("Mutual_Information_Ranking.png", "Feature ranking based on mutual information with the target."),
        "Mutual information health status": ("Mutual_Information_Health_Status.png", "Detailed mutual information scores for health status."),
    },
    "Health Conditions": {
        "High blood pressure": ("HighBP.png", "Distribution of high blood pressure status."),
        "High cholesterol": ("HighChol.png", "Distribution of high cholesterol status."),
        "Cholesterol check": ("CholCheck.png", "Whether respondents had a cholesterol check."),
        "Stroke history": ("Stroke.png", "Distribution of reported stroke history."),
        "Heart disease or attack": ("HeartDiseaseorAttack.png", "Distribution of reported heart disease or heart attack."),
        "General health": ("GenHlth.png", "Self-reported general health categories."),
        "Health status raw distribution": ("Health_Status_Raw_Distribution.png", "Raw distribution of health status metrics."),
        "Health status raw distribution (Pie)": ("Health_Status_Raw_Distribution_Pie.png", "Raw distribution of health status metrics (Pie chart)."),
        "Health status by year": ("Health_Status_by_Year.png", "Health status breakdown across survey years."),
        "Difficulty walking": ("DiffWalk.png", "Reported difficulty walking or climbing stairs."),
    },
    "BMI & Health Status": {
        "BMI distribution": ("BMI.png", "Distribution of body mass index."),
        "BMI categories": ("BMI_Categories.png", "Respondents grouped into BMI categories."),
        "BMI by diabetes category — boxplot": ("BMI_vs_Diabetes_Boxplot.png", "BMI distribution across the three diabetes categories."),
        "BMI category vs diabetes — heatmap": ("BMI_vs_Diabetes_Heatmap.png", "Diabetes class percentages by BMI category."),
        "BMI vs health status scatter trend": ("BMI_vs_Health_Status_Scatter_Trend.png", "Scatter trend analysis of BMI and health status."),
        
        "Physical health days": ("PhysHlth.png", "Number of days physical health was not good."),
        "Mental health days": ("MentHlth.png", "Number of days mental health was not good."),
    },
    "Lifestyle & Access": {
        "Smoking status": ("Smoker.png", "Distribution of reported smoking status."),
        "Physical activity": ("PhysActivity.png", "Distribution of physical activity."),
        "Fruit consumption": ("Fruits.png", "Reported fruit consumption."),
        "Vegetable consumption": ("Veggies.png", "Reported vegetable consumption."),
        "Heavy alcohol consumption": ("HvyAlcoholConsump.png", "Reported heavy alcohol consumption."),
        "Any healthcare coverage": ("AnyHealthcare.png", "Whether respondents reported healthcare coverage."),
        "Could not see doctor due to cost": ("NoDocbcCost.png", "Reported inability to see a doctor because of cost."),
    },
    "Risk & Prevalence Trends": {
        "High blood pressure vs diabetes": ("HighBP_vs_Diabetes_Heatmap.png", "Diabetes class proportions by high blood pressure status."),
        "High cholesterol vs diabetes": ("HighChol_vs_Diabetes_Heatmap.png", "Diabetes class proportions by high cholesterol status."),
        "Walking difficulty vs diabetes": ("DiffWalk_vs_Diabetes_Heatmap.png", "Diabetes class proportions by walking difficulty."),
        "At-risk percentage by age": ("AtRisk_by_Age.png", "At-Risk percentage in the sample categorized by age groups."),
        "At-risk percentage by age & income": ("AtRisk_by_Age_Income.png", "At-risk percentage cross-analyzed by age and income."),
        "At-risk percentage by binary features": ("AtRisk_by_Binary_Features.png", "At-risk trends across binary biometric indicators."),
        "At-risk percentage by combined profile": ("AtRisk_by_Combined_Profile.png", "At-risk metrics based on combined risk profiles."),
        "At-risk percentage by general health": ("AtRisk_by_General_Health.png", "At-risk breakdown by self-reported general health."),
        "At-risk percentage by BMI": ("AtRisk_Percent_by_BMI.png", "At-risk percentage plotted by BMI value."),
    },
    "Demographics & Socioeconomic": {
        "Age distribution": ("Age.png", "Distribution across age categories."),
        "Sex distribution": ("Sex.png", "Distribution by sex as encoded in the dataset."),
        "Education distribution": ("Education.png", "Distribution across education categories."),
        "Income distribution": ("Income.png", "Distribution across income categories."),
        "Socioeconomic diabetes trend": ("Socioeconomic_Diabetes_Trend.png", "Diabetes-related trend across age and income groups."),
        "Socioeconomic heart disease trend": ("Socioeconomic_HeartDisease_Trend.png", "Heart-disease trend across age and income groups."),
    },
    "Model Evaluation": {
        "Confusion matrix (CatBoost)": ("confusion_matrix_catboost.png", "Confusion matrix evaluation for CatBoost model."),
        "Confusion matrix (LightGBM)": ("confusion_matrix_lightgbm.png", "Confusion matrix evaluation for LightGBM model."),
        "Confusion matrix (XGBoost)": ("confusion_matrix_xgb.png", "Confusion matrix evaluation for XGBoost model."),
        "Confusion matrix (Raw model)": ("confusion_matrix_model1_raw.png", "Confusion matrix evaluation for baseline raw model."),
        "Confusion matrix (Balanced model)": ("confusion_matrix_model2_balanced.png", "Confusion matrix evaluation for class-balanced model."),
        "Confusion matrix (SMOTE model)": ("confusion_matrix_model3_smote.png", "Confusion matrix evaluation for SMOTE-resampled model."),
    }
}

file_by_title = {title: details[0] for group in CHARTS.values() for title, details in group.items()}
description_by_title = {title: details[1] for group in CHARTS.values() for title, details in group.items()}

def toggle_all(category_name, titles):
    val = st.session_state[f"select_all_{category_name}"]
    for title in titles:
        st.session_state[f"chk_{category_name}_{title}"] = val

tab_labels = list(CHARTS.keys())
tabs = st.tabs(tab_labels)

for tab, category in zip(tabs, tab_labels):
    with tab:
        category_charts = CHARTS[category]
        chart_titles = list(category_charts.keys())

        # Dynamically sync 'Select All' based on whether all individual checkboxes are currently true
        all_currently_checked = all(
            st.session_state.get(f"chk_{category}_{title}", False)
            for title in chart_titles
        )
        st.session_state[f"select_all_{category}"] = all_currently_checked
        
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader(f"{category} Visualizations")
        with c2:
            st.checkbox(
                "Select All",
                key=f"select_all_{category}",
                on_change=toggle_all,
                args=(category, chart_titles)
            )

        selected_titles = []
        for title in chart_titles:
            is_checked = st.checkbox(title, key=f"chk_{category}_{title}")
            if is_checked:
                selected_titles.append(title)

        st.divider()

        if not selected_titles:
            st.info("Please select at least one visualization above.")
        else:
            for title in selected_titles:
                filename = file_by_title[title]
                image_path = ASSET_DIR / filename
                if image_path.is_file():
                    st.markdown(f"### {title}")
                    st.image(str(image_path), use_container_width=True)
                    st.write(description_by_title[title])
                    st.divider()







