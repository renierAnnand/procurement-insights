import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import all modules
import contracting_opportunities
import cross_region
import duplicates
import lot_size_optimization
import reorder_prediction
import seasonal_price_optimization
import spend_categorization_anomaly

def load_and_clean_data(uploaded_file):
    """Load and clean the uploaded data file"""
    try:
        # Read Excel file
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Please upload an Excel (.xlsx, .xls) or CSV file")
            return None
        
        # Data cleaning and transformation
        if not df.empty:
            # Calculate actual unit price from Line Total / Qty Delivered
            df['Actual_Unit_Price'] = np.where(
                (df['Qty Delivered'] > 0) & (df['Line Total'] > 0),
                df['Line Total'] / df['Qty Delivered'],
                0
            )
            
            # Use Ordered Date as Creation Date (since Creation Date is corrupted)
            if 'Ordered Date' in df.columns:
                df['Creation Date'] = pd.to_datetime(df['Ordered Date'], errors='coerce')
            
            # Set currency to SAR (since Unit Price column contains currency codes)
            df['Currency_Clean'] = 'SAR'
            
            # Replace the Unit Price column with calculated actual unit price
            df['Unit Price'] = df['Actual_Unit_Price']
            
            # Clean any missing or invalid data
            df = df.dropna(subset=['Vendor Name', 'Item'])
            df = df[df['Line Total'] > 0]
            df = df[df['Qty Delivered'] > 0]
            
            # Convert dates properly
            date_columns = ['Ordered Date', 'Delivered date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
        else:
            st.error("The uploaded file is empty")
            return None
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

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
        type=['csv', 'xlsx', 'xls'],
        help="Upload Excel or CSV file with procurement data"
    )
    
    if uploaded_file is not None:
        # Load and clean data
        with st.spinner("Loading and processing data..."):
            df = load_and_clean_data(uploaded_file)
        
        if df is not None:
            # Data overview
            st.sidebar.success(f"✅ Data loaded successfully!")
            st.sidebar.metric("Total Records", len(df))
            st.sidebar.metric("Date Range", f"{df['Creation Date'].min().date()} to {df['Creation Date'].max().date()}")
            
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
                options=list(analysis_modules.keys()),
                index=0
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
        
    else:
        # Show sample data requirements
        st.info("👆 Please upload your procurement data file to get started")
        
        st.subheader("📋 Required Data Format")
        st.markdown("""
        Your data file should contain the following columns:
        
        **Essential columns:**
        - `Business Unit`: Organization unit
        - `Vendor Name`: Supplier name
        - `Item`: Item/product code
        - `Item Description`: Product description
        - `Line Total`: Total amount for the line
        - `Qty Delivered`: Quantity delivered
        - `Ordered Date`: Order placement date
        - `W/H`: Warehouse location
        
        **Optional columns:**
        - `Qty Rejected`: Rejected quantity
        - `Delivered date`: Delivery date
        - `Purchase Order`: PO number
        - `Category`: Item category
        """)
        
        # Sample data preview
        with st.expander("📊 View Sample Data Format"):
            sample_data = {
                'Business Unit': ['Company A', 'Company A', 'Company B'],
                'Vendor Name': ['Supplier 1', 'Supplier 2', 'Supplier 1'],
                'Item': ['ITEM001', 'ITEM002', 'ITEM003'],
                'Item Description': ['Product A', 'Product B', 'Product C'],
                'Line Total': [1000.00, 750.50, 1200.00],
                'Qty Delivered': [10, 5, 8],
                'Ordered Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
                'W/H': ['Warehouse_1', 'Warehouse_2', 'Warehouse_1']
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()
