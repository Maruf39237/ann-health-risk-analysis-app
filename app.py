from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Diabetes Health Indicators | EDA",
    page_icon="🩺",
    # layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

st.markdown(
    """
    <style>
    /* Target the text inside the modern navigation items */
    [data-testid="stSidebarNavItems"] span {
        font-size: 20px !important;
        font-weight: bold;
    }
    
    /* Optional: Make the emojis larger to match the new text size */
    [data-testid="stSidebarNavItems"] span img,
    [data-testid="stSidebarNavItems"] span emoji {
        transform: scale(1.3);
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

home_page = st.Page(str(ROOT / "Pages" / "home.py"), title="Home", icon="🏠", default=True)
eda_page = st.Page(str(ROOT / "Pages" / "eda.py"), title="EDA Explorer", icon="📊")
prediction_page = st.Page(str(ROOT / "Pages" / "prediction.py"), title="Make Prediction", icon="📍")
report_page = st.Page(str(ROOT / "Pages" / "report.py"), title="Report", icon="📄")

pg = st.navigation([home_page, eda_page, prediction_page, report_page], position="hidden")

st.sidebar.image("https://images.pexels.com/photos/6942252/pexels-photo-6942252.jpeg", use_container_width=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True) 

st.sidebar.page_link(home_page, label="Home", icon="🏠")
st.sidebar.page_link(eda_page, label="EDA Explorer", icon="📊")
st.sidebar.page_link(report_page, label="Report", icon="📄")
st.sidebar.page_link(prediction_page, label="Make Prediction", icon="📍")

pg.run()