import streamlit as st
import pandas as pd
import sys
import os

# Import all modules
import contracting_opportunities
import seasonal_price_optimization
import spend_categorization_anomaly
import lot_size_optimization
import cross_region
import duplicates
import reorder_prediction

# Page configuration
st.set_page_config(
    page_title="Procurement Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-status {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Procurement Analytics Platform</h1>
        <p>Advanced AI-Powered Procurement Intelligence & Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload and module selection
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Procurement Data", 
            type=['csv', 'xlsx'],
            help="Upload your procurement data file (CSV or Excel format)"
        )
        
        if uploaded_file is not None:
            # Load data
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File loaded: {uploaded_file.name}")
                st.info(f"📊 {len(df)} records loaded")
                
                # Data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Module selection
                st.header("🔧 Analytics Modules")
                
                modules = {
                    "📋 Data Overview": "overview",
                    "🤝 Contracting Opportunities": "contracting",
                    "🌟 Seasonal Price Optimization": "seasonal",
                    "📊 Spend Categorization & Anomaly": "spend_anomaly",
                    "📦 LOT Size Optimization": "lot_size",
                    "🌍 Cross-Region Analysis": "cross_region",
                    "🔍 Duplicate Detection": "duplicates",
                    "📈 Reorder Prediction": "reorder"
                }
                
                # Module status
                st.markdown("""
                <div class="module-status">
                    <h4>📊 Analytics Modules Status</h4>
                    <p><strong>Available Modules:</strong> 7/7</p>
                </div>
                """, unsafe_allow_html=True)
                
                selected_module = st.selectbox(
                    "Select Analytics Module",
                    list(modules.keys()),
                    index=0
                )
                
                # Display selected module
                if selected_module == "📋 Data Overview":
                    show_data_overview(df)
                elif modules[selected_module] == "contracting":
                    contracting_opportunities.display(df)
                elif modules[selected_module] == "seasonal":
                    seasonal_price_optimization.display(df)
                elif modules[selected_module] == "spend_anomaly":
                    spend_categorization_anomaly.display(df)
                elif modules[selected_module] == "lot_size":
                    lot_size_optimization.display(df)
                elif modules[selected_module] == "cross_region":
                    cross_region.display(df)
                elif modules[selected_module] == "duplicates":
                    duplicates.display(df)
                elif modules[selected_module] == "reorder":
                    reorder_prediction.display(df)
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.info("Please ensure your file has the correct format and required columns.")
        
        else:
            # Show welcome message when no file is uploaded
            st.info("👆 Please upload a procurement data file to begin analysis")
            show_welcome_screen()

def show_data_overview(df):
    """Display data overview and basic statistics"""
    st.header("📋 Data Overview")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        unique_vendors = df['Vendor Name'].nunique() if 'Vendor Name' in df.columns else 0
        st.metric("Unique Vendors", f"{unique_vendors:,}")
    with col3:
        unique_items = df['Item'].nunique() if 'Item' in df.columns else 0
        st.metric("Unique Items", f"{unique_items:,}")
    with col4:
        if 'Line Total' in df.columns:
            total_spend = df['Line Total'].sum()
        elif 'Unit Price' in df.columns and 'Qty Delivered' in df.columns:
            total_spend = (df['Unit Price'] * df['Qty Delivered']).sum()
        else:
            total_spend = 0
        st.metric("Total Spend", f"${total_spend:,.0f}")
    
    # Column information
    st.subheader("📊 Dataset Information")
    
    col_info = []
    for col in df.columns:
        col_info.append({
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Unique Values': df[col].nunique()
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True)
    
    # Data sample
    st.subheader("📋 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

def show_welcome_screen():
    """Show welcome screen with module descriptions"""
    st.header("🚀 Welcome to Procurement Analytics Platform")
    
    st.markdown("""
    ## 📊 Available Analytics Modules:
    
    ### 🤝 **Contracting Opportunities**
    - Identify optimal contracting opportunities
    - Vendor performance analysis
    - Contract savings calculation
    - Implementation roadmap
    
    ### 🌟 **Seasonal Price Optimization**
    - Analyze seasonal price patterns
    - Optimal purchase timing recommendations
    - Potential savings calculator
    
    ### 📊 **Spend Categorization & Anomaly Detection**
    - AI-powered spend categorization
    - Anomaly detection using machine learning
    - Spend insights and recommendations
    
    ### 📦 **LOT Size Optimization**
    - Economic Order Quantity (EOQ) analysis
    - Inventory optimization
    - Cost reduction opportunities
    
    ### 🌍 **Cross-Region Analysis**
    - Compare pricing across regions
    - Vendor optimization opportunities
    - Regional spend analysis
    
    ### 🔍 **Duplicate Detection**
    - Identify duplicate vendors/items
    - Fuzzy matching algorithms
    - Data quality improvements
    
    ### 📈 **Reorder Prediction**
    - Smart reorder point calculation
    - Demand forecasting
    - Inventory planning
    """)
    
    st.info("👆 Upload your procurement data file to start analyzing!")

if __name__ == "__main__":
    main()
