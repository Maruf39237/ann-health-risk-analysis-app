from pathlib import Path
import streamlit as st
import pymupdf as fitz
import base64

st.set_page_config(
    page_title="Project Insights & Final Report",
    page_icon="📄",
)

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "DOC"
PDF_PATH = DOC_DIR / "Diabetes_EDA_Insights_Report_Final.pdf"

st.divider()

col1, col2 = st.columns([3, 2.4])
with col1:
    st.markdown("# Final Exploratory Data Analysis (EDA) Insights Report📄")
    st.markdown("""
##### • Comprehensive Documentation
##### • Multi-Year BRFSS Cohorts
##### • Executive Summary & Findings  
""")

with col2:
    st.image(
        "https://www.diabetesselfmanagement.com/wp-content/uploads/2023/02/dsm-fight-off-type-2-diabetes-shutterstock-shutterstock-1822915058-1.jpg",
        caption="",
        width=350
    )
    st.markdown("""
    #### Access the complete pre-compiled analytical report summarizing all findings from 453k+ records.
    """)

st.divider()

report_tab1, report_tab2, report_tab3 = st.tabs([
    "Executive Hub & Download", 
    "Structured Core Findings", 
    "Interactive Page-by-Page Reader"
])

with report_tab1:
    st.subheader("Executive Summary & Document Hub")
    st.markdown("""
    This section provides direct access to the finalized, production-grade analytical report generated from the comprehensive Exploratory Data Analysis of the combined `BRFSS 2015 & 2021` dataset (`453,331` records).
    """)
    
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    
    
    st.markdown("""
    **Report Specifications:**
    - **File Name:** `Diabetes_EDA_Insights_Report_Final.pdf`
    - **Target Audience:** Healthcare Analysts, Data Scientists, and Epidemiologists
    - **Coverage:** Data hygiene audit, biometric risk gradients, mutual information ranking, and model evaluation metrics.
    """)

    st.success("✅ **Final Report Status:** Verified and ready for download.")
    st.download_button(
        label="📥 Download Full PDF Report",
        data=pdf_bytes,
        file_name="Diabetes_EDA_Insights_Report_Final.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Analyzed Records", "453,331", "100% Clean Data")
    m2.metric("Survey Cycles", "2015 & 2021", "Multi-Year Cohort")
    m3.metric("Document Type", "PDF Report", "Polished / Refined")
    st.divider()

with report_tab2:
    st.subheader("Structured Core Findings & Insights")
    st.markdown("A synthesized breakdown of the key chapters included in the final report:")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("""
        #### 1. Data Hygiene & Preprocessing Audit
        - Rigorous handling of multi-year CDC BRFSS schemas.
        - Successful identification and removal of **36,727 duplicate records** from the raw combined dataset (490,058 initial rows).
        - Zero missing values confirmed post-cleaning across all 25 core analytical attributes.
        
        #### 2. Biometric Risk Factors & BMI Gradients
        - Confirmed sharp non-linear escalation of diabetes prevalence in higher BMI categories (Overweight and Obese tiers).
        - High Blood Pressure (`HighBP`) and High Cholesterol (`HighChol`) identified as dominant co-morbid indicators showing powerful mutual information scores.
        """)
    with col_f2:
        st.markdown("""
        #### 3. Socioeconomic & Lifestyle Gradients
        - Clear stratification across income and education brackets, highlighting vulnerable demographic segments.
        - Behavioral factors such as physical activity (`PhysActivity`) and walking difficulty (`DiffWalk`) exhibit strong predictive divergence between control and target classes.
        
        #### 4. Model Evaluation & Class Imbalance
        - Side-by-side comparative analysis of gradient-boosted trees (XGBoost, CatBoost, LightGBM) and deep learning Artificial Neural Networks (ANN).
        - Evaluation of SMOTE oversampling and class weighting techniques to address severe target imbalance (`Diabetes_012`).
        """)

with report_tab3:
    st.subheader("Interactive Page-by-Page Reader")
    st.markdown("Browse through the report pages seamlessly using the page navigation selector below:")
    
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    
    col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
    with col_p2:
        page_num = st.number_input(
            f"Select Page (1 to {total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )
    
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=150)
    st.image(pix.tobytes("png"), caption=f"Page {page_num} of {total_pages}", use_container_width=True)