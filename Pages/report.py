from pathlib import Path
import streamlit as st
import fitz  
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
    "📖 Original PDF Viewer"
])

with report_tab1:
    st.subheader("Executive Summary & Document Hub")
    st.markdown("""
    This section provides direct access to the finalized, production-grade analytical report generated from the comprehensive Exploratory Data Analysis of the combined `BRFSS 2015 & 2021` dataset (`453,331` records).
    """)
    
    if PDF_PATH.is_file():
        with open(PDF_PATH, "rb") as f:
            pdf_bytes = f.read()
    else:
        pdf_bytes = b""
    
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
    st.subheader("📖 Original PDF Viewer")
    st.caption("High-quality page rendering • Works on laptop & mobile")

    if PDF_PATH.is_file():
        import fitz
        import base64

        doc = fitz.open(PDF_PATH)
        total_pages = len(doc)

        # Centered page selector (works well on mobile too)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page_num = st.number_input(
                f"Page (1 – {total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="pdf_page_selector"
            )

        # Render page
        page = doc[page_num - 1]
        zoom = 4.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

        # Responsive bordered container
        bordered_html = f"""
        <div style="
            border: 2px solid #cbd5e1;
            border-radius: 10px;
            padding: 10px;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
            width: 100%;
            max-width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        ">
            <img src="data:image/png;base64,{b64_img}" 
                 style="
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                 " />
        </div>
        """
        st.markdown(bordered_html, unsafe_allow_html=True)

        doc.close()
    else:
        st.error(f"PDF not found: {PDF_PATH}")        

