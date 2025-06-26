import streamlit as st
import pandas as pd
import numpy as np

# Import all modules
import contracting_opportunities
import cross_region
import duplicates
import lot_size_optimization
import reorder_prediction
import seasonal_price_optimization
import spend_categorization_anomaly

def main():
    st.set_page_config(
        page_title="Procurement Analytics Suite", 
        page_icon="📊", 
        layout="wide"
    )
    
    st.title("📊 Procurement Analytics Suite")
    st.markdown("Advanced procurement analysis and optimization tools")
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your procurement data", 
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.sidebar.success("✅ Data loaded successfully!")
            st.sidebar.metric("Total Records", len(df))
            
            # Module selection
            st.sidebar.header("🔧 Analysis Modules")
            
            analysis_modules = {
                "🤝 Contracting Opportunities": "contracting_opportunities",
                "🌍 Cross-Region Optimization": "cross_region", 
                "🔍 Duplicate Detection": "duplicates",
                "📦 LOT Size Optimization": "lot_size_optimization",
                "📈 Reorder Prediction": "reorder_prediction",
                "🌟 Seasonal Price Optimization": "seasonal_price_optimization",
                "📊 Spend Categorization & Anomaly Detection": "spend_categorization_anomaly"
            }
            
            selected_module = st.sidebar.selectbox(
                "Select Analysis Module",
                options=list(analysis_modules.keys())
            )
            
            # Run selected module
            module_name = analysis_modules[selected_module]
            
            if module_name == "contracting_opportunities":
                contracting_opportunities.display(df)
            elif module_name == "cross_region":
                cross_region.display(df)
            elif module_name == "duplicates":
                duplicates.display(df)
            elif module_name == "lot_size_optimization":
                lot_size_optimization.display(df)
            elif module_name == "reorder_prediction":
                reorder_prediction.display(df)
            elif module_name == "seasonal_price_optimization":
                seasonal_price_optimization.display(df)
            elif module_name == "spend_categorization_anomaly":
                spend_categorization_anomaly.display(df)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload your procurement data file to get started")

if __name__ == "__main__":
    main()
