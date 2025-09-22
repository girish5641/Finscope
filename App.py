
import streamlit as st
from Home import home_page
from Portfolio import portfolio_optimizer_page
from BSM import bsm_analysis_simple
from Option import run_option_analysis_dashboard

# Custom CSS styling
st.markdown(
    """
    <style>
    /* Main sidebar container */
    [data-testid="stSidebar"] {
        background-color: #f0f4f8;
        padding: 20px 15px !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1) !important;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h2 {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #34495e !important;
        margin-bottom: 20px !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    /* Selectbox label */
    [data-testid="stSidebar"] label {
        font-weight: 600 !important;
        font-size: 16px !important;
        color: #2c3e50 !important;
        margin-bottom: 10px !important;
        display: block;
    }

    /* Selectbox dropdown */
    [data-testid="stSelectbox"] div[role="button"] {
        border-radius: 8px !important;
        border: 1.5px solid #2980b9 !important;
        background-color: white !important;
        font-size: 16px !important;
        color: #2c3e50 !important;
        padding: 8px 12px !important;
    }

    /* Hover effect on selectbox */
    [data-testid="stSelectbox"] div[role="button"]:hover {
        border-color: #1abc9c !important;
        box-shadow: 0 0 6px rgba(26, 188, 156, 0.3) !important;
    }
    
    /* Dropdown menu */
    [role="listbox"] {
        border-radius: 8px !important;
        border: 1.5px solid #2980b9 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* Dropdown menu items */
    [role="option"] {
        padding: 8px 12px !important;
    }
    
    /* Hover effect on dropdown items */
    [role="option"]:hover {
        background-color: #f0f4f8 !important;
        color: #1abc9c !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Select a page", 
        ["Home", "Portfolio Optimizer", "BSM Model", "Option Pricing"],
        label_visibility="collapsed"
    )

# Page routing
if page == "Home":
    home_page()
elif page == "Portfolio Optimizer":
    portfolio_optimizer_page()
elif page == "BSM Model":
    st.header("Black-Scholes-Merton Model Analysis")
    col1, col2 = st.columns(2)
    with col1:
        K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
        T = st.number_input("Maturity (years)", value=1.0, step=0.1, min_value=0.1)
    with col2:
        r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01, format="%.2f")
        sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01, min_value=0.01, format="%.2f")
    
    if st.button("Run BSM Analysis and Plots"):
        bsm_analysis_simple(K=K, T=T, r=r, sigma=sigma)
elif page == "Option Pricing":
    run_option_analysis_dashboard()

