import streamlit as st
import pandas as pd

# Import use case modules
from modules import reorder_prediction, cross_region, duplicates

# Set Streamlit page configuration
st.set_page_config(
    page_title="Procurement Intelligence Dashboard",
    layout="wide"
)

# Title of the dashboard
st.title("📦 Procurement Intelligence Dashboard")

# Load the cleansed data
@st.cache_data
def load_data():
    return pd.read_csv("data/Cleansed_PO_Data.csv")

df = load_data()

# Sidebar navigation
st.sidebar.title("Use Cases")
page = st.sidebar.radio("Choose an analysis module:", [
    "Smart Reorder Point Prediction",
    "Cross-Region Vendor Optimization",
    "Duplicate Vendor/Item Detection"
])

# Route to appropriate module
if page == "Smart Reorder Point Predicti
