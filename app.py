import streamlit as st
import pandas as pd
import numpy as np
import re

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

# Custom CSS
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
    .sidebar-header {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .module-status {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

def clean_numeric_column(series, column_name):
    """Clean numeric columns by removing non-numeric characters and converting to float"""
    try:
        series = series.astype(str)
        series = series.str.replace('$', '', regex=False)
        series = series.str.replace(',', '', regex=False)
        series = series.str.replace('%', '', regex=False)
        series = series.str.replace(' ', '', regex=False)
        series = series.replace(['N/A', 'NA', 'n/a', 'na', 'NULL', 'null', '', 'TBD', 'tbd'], np.nan)
        series = pd.to_numeric(series, errors='coerce')
        
        if 'qty' in column_name.lower() or 'quantity' in column_name.lower():
            series = series.clip(lower=0)
            
        return series
    except Exception as e:
        st.warning(f"Warning cleaning {column_name}: {str(e)}")
        return series

def clean_date_column(series):
    """Clean date columns"""
    try:
        return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
    except Exception as e:
        st.warning(f"Warning cleaning date column: {str(e)}")
        return series

def validate_and_clean_data(df):
    """Comprehensive data validation and cleaning"""
    
    original_rows = len(df)
    issues_found = []
    df_clean = df.copy()
    
    # Clean column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # Column name mappings
    column_mappings = {
        'vendor': 'Vendor Name', 'vendor_name': 'Vendor Name', 'supplier': 'Vendor Name',
        'item_id': 'Item', 'item_code': 'Item', 'product': 'Item',
        'unit_price': 'Unit Price', 'price': 'Unit Price', 'cost': 'Unit Price',
        'quantity': 'Qty Delivered', 'qty': 'Qty Delivered', 'amount': 'Qty Delivered',
        'date': 'Creation Date', 'order_date': 'Creation Date', 'purchase_date': 'Creation Date',
        'total': 'Line Total', 'line_total': 'Line Total'
    }
    
    # Apply column mappings
    for old_name, new_name in column_mappings.items():
        for col in df_clean.columns:
            if old_name.lower() in col.lower() and new_name not in df_clean.columns:
                df_clean = df_clean.rename(columns={col: new_name})
                break
    
    # Clean numeric columns
    numeric_columns = ['Unit Price', 'Qty Delivered', 'Line Total', 'Qty Rejected']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = clean_numeric_column(df_clean[col], col)
            null_count = df_clean[col].isnull().sum()
            if null_count > 0:
                issues_found.append(f"Found {null_count} invalid values in {col}")
    
    # Clean date columns
    date_columns = ['Creation Date', 'Delivery Date', 'Order Date']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = clean_date_column(df_clean[col])
    
    # Calculate Line Total if missing
    if 'Line Total' not in df_clean.columns and 'Unit Price' in df_clean.columns and 'Qty Delivered' in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    # Remove rows with critical missing data
    critical_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered']
    for col in critical_columns:
        if col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[col])
    
    # Remove invalid values
    if 'Unit Price' in df_clean.columns:
        df_clean = df_clean[df_clean['Unit Price'] > 0]
    if 'Qty Delivered' in df_clean.columns:
        df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    return df_clean, issues_found

def check_required_columns(df):
    """Check if the dataframe has minimum required columns"""
    required_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

def main():
    # Main header - OUTSIDE sidebar
    st.markdown("""
    <div class="main-header">
        <h1>📊 Procurement Analytics Platform</h1>
        <p>Advanced AI-Powered Procurement Intelligence & Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'selected_module' not in st.session_state:
        st.session_state.selected_module = "📋 Data Overview"
    
    # SIDEBAR - Only for file upload and module selection
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>📁 Data Upload</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Procurement Data", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload your procurement data file (CSV or Excel format)"
        )
        
        # File processing
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
                else:
                    df_raw = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded: {uploaded_file.name}")
                st.info(f"📊 {len(df_raw):,} records")
                
                # Clean and validate data
                df_clean, issues = validate_and_clean_data(df_raw)
                
                # Check required columns
                has_required, missing = check_required_columns(df_clean)
                
                if has_required and len(df_clean) > 0:
                    st.session_state.df = df_clean
                    
                    st.markdown("""
                    <div class="module-status">
                        ✅ Data Ready for Analysis
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Module selection
                    st.markdown("### 🔧 Analytics Modules")
                    
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
                    
                    st.session_state.selected_module = st.selectbox(
                        "Select Module",
                        list(modules.keys()),
                        index=list(modules.keys()).index(st.session_state.selected_module) 
                        if st.session_state.selected_module in modules.keys() else 0,
                        key="module_selector"
                    )
                    
                    st.markdown(f"""
                    <div class="module-status">
                        📊 Available Modules: 7/7<br>
                        🎯 Selected: {st.session_state.selected_module.split(' ', 1)[1]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error("❌ Data validation failed")
                    if missing:
                        st.error(f"Missing: {', '.join(missing)}")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        else:
            st.info("👆 Upload a file to begin")
    
    # MAIN CONTENT AREA - Display selected module content
    if st.session_state.df is not None:
        df = st.session_state.df
        selected_module = st.session_state.selected_module
        
        # Map selection to module
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
        
        # Display the selected module in MAIN AREA
        try:
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
            st.error(f"❌ Error in {selected_module}: {str(e)}")
            st.info("This might be due to missing columns or data format issues.")
    
    else:
        # Welcome screen in MAIN AREA when no data is loaded
        show_welcome_screen()

def show_data_overview(df):
    """Display data overview and basic statistics in MAIN area"""
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
    
    # Data quality summary
    st.subheader("📊 Data Quality Summary")
    
    quality_info = []
    for col in df.columns:
        quality_info.append({
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': f"{df[col].count():,}",
            'Null Count': f"{df[col].isnull().sum():,}",
            'Unique Values': f"{df[col].nunique():,}",
            'Completeness': f"{(df[col].count() / len(df) * 100):.1f}%"
        })
    
    quality_df = pd.DataFrame(quality_info)
    st.dataframe(quality_df, use_container_width=True)
    
    # Data sample
    st.subheader("📋 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

def show_welcome_screen():
    """Show welcome screen in MAIN area"""
    st.header("🚀 Welcome to Procurement Analytics Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 📋 File Format Requirements:
        
        **Required Columns:**
        - **Vendor Name** (text): Supplier/vendor name
        - **Unit Price** (number): Price per unit (no $ symbols)
        - **Qty Delivered** (number): Quantity delivered
        
        **Optional Columns:**
        - **Item** (text/number): Item ID or name
        - **Creation Date** (date): Order date
        - **Line Total** (number): Total amount
        - **Item Description** (text): Item description
        
        ## ⚠️ Common Issues to Avoid:
        - $ symbols in price columns
        - Commas in numbers (1,000)
        - Text in numeric columns ("N/A", "TBD")
        - Inconsistent date formats
        """)
    
    with col2:
        st.markdown("""
        ## 📊 Available Analytics Modules:
        
        ### 🤝 **Contracting Opportunities**
        Identify optimal contracting opportunities based on spend analysis.
        
        ### 🌟 **Seasonal Price Optimization**
        Analyze seasonal price patterns to optimize purchase timing.
        
        ### 📊 **Spend Categorization & Anomaly Detection**
        AI-powered spend categorization and anomaly detection.
        
        ### 📦 **LOT Size Optimization**
        Economic Order Quantity (EOQ) analysis for inventory optimization.
        
        ### 🌍 **Cross-Region Analysis**
        Compare pricing and performance across different regions.
        
        ### 🔍 **Duplicate Detection**
        Identify potential duplicate vendors and items.
        
        ### 📈 **Reorder Prediction**
        Smart reorder point calculation and demand forecasting.
        """)
    
    st.info("👈 Upload your procurement data file in the sidebar to start analyzing!")

if __name__ == "__main__":
    main()
