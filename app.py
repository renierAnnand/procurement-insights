import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO
from math import sqrt

# Configure the page
st.set_page_config(
    page_title="Procurement Analytics Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe imports with error handling
def safe_import_modules():
    """Safely import modules and track which ones are available"""
    available_modules = {}
    
    # Try importing each module
    try:
        import lot_size_optimization
        available_modules['lot_size'] = lot_size_optimization
        st.sidebar.success("✅ LOT Size Optimization loaded")
    except Exception as e:
        st.sidebar.error(f"❌ LOT Size Optimization failed: {str(e)}")
        available_modules['lot_size'] = None
    
    try:
        import contracting_opportunities
        available_modules['contracting'] = contracting_opportunities
        st.sidebar.success("✅ Contracting Opportunities loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Contracting Opportunities failed: {str(e)}")
        available_modules['contracting'] = None
    
    try:
        import seasonal_price_optimization
        available_modules['seasonal'] = seasonal_price_optimization
        st.sidebar.success("✅ Seasonal Price Optimization loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Seasonal Price Optimization failed: {str(e)}")
        available_modules['seasonal'] = None
    
    try:
        import spend_categorization_anomaly
        available_modules['spend_anomaly'] = spend_categorization_anomaly
        st.sidebar.success("✅ Spend Analysis loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Spend Analysis failed: {str(e)}")
        available_modules['spend_anomaly'] = None
    
    try:
        import duplicates
        available_modules['duplicates'] = duplicates
        st.sidebar.success("✅ Duplicate Detection loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Duplicate Detection failed: {str(e)}")
        available_modules['duplicates'] = None
    
    try:
        import cross_region
        available_modules['cross_region'] = cross_region
        st.sidebar.success("✅ Cross-Region Analysis loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Cross-Region Analysis failed: {str(e)}")
        available_modules['cross_region'] = None
    
    try:
        import reorder_prediction
        available_modules['reorder'] = reorder_prediction
        st.sidebar.success("✅ Reorder Prediction loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Reorder Prediction failed: {str(e)}")
        available_modules['reorder'] = None
    
    return available_modules

# Built-in LOT Size Optimization (backup implementation)
def builtin_lot_size_optimization(df):
    """Built-in LOT Size Optimization - backup implementation"""
    st.header("📦 LOT Size Optimization (Built-in)")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management.")
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("This module requires: Item, Unit Price, and Qty Delivered columns")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        holding_cost_rate = st.slider("Holding Cost Rate (%)", 5, 30, 15) / 100
    with col2:
        ordering_cost = st.number_input("Ordering Cost ($)", 50, 500, 100)
    with col3:
        working_days = st.number_input("Working Days/Year", 200, 365, 250)
    
    # Item selection
    items = sorted(df_clean['Item'].unique())
    selected_item = st.selectbox("Select Item for EOQ Analysis", items)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item]
        
        # Calculate demand and costs
        annual_demand = item_data['Qty Delivered'].sum()
        avg_unit_cost = item_data['Unit Price'].mean()
        holding_cost = avg_unit_cost * holding_cost_rate
        
        # EOQ calculation
        if annual_demand > 0 and holding_cost > 0:
            eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
            current_avg_order = item_data['Qty Delivered'].mean()
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Annual Demand", f"{annual_demand:,.0f}")
            with col2:
                st.metric("Optimal Order Qty (EOQ)", f"{eoq:.0f}")
            with col3:
                st.metric("Current Avg Order", f"{current_avg_order:.0f}")
            with col4:
                potential_savings = abs(current_avg_order - eoq) * 10  # Simplified calculation
                st.metric("Potential Savings", f"${potential_savings:,.0f}")

# Built-in Spend Analysis (backup implementation)
def builtin_spend_analysis(df):
    """Built-in Spend Analysis - backup implementation"""
    st.header("📊 Spend Analysis (Built-in)")
    st.markdown("Basic spend categorization and analysis.")
    
    # Calculate line total if missing
    if 'Line Total' not in df.columns:
        df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    
    # Vendor analysis
    vendor_spend = df.groupby('Vendor Name')['Line Total'].sum().sort_values(ascending=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Spend", f"${df['Line Total'].sum():,.0f}")
    with col2:
        st.metric("Top Vendor", vendor_spend.index[0])
    with col3:
        st.metric("Vendor Count", len(vendor_spend))
    
    # Top vendors chart
    fig = px.bar(
        x=vendor_spend.head(10).values,
        y=vendor_spend.head(10).index,
        orientation='h',
        title="Top 10 Vendors by Spend"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Spend by month
    if 'Creation Date' in df.columns:
        df['Month'] = pd.to_datetime(df['Creation Date']).dt.to_period('M')
        monthly_spend = df.groupby('Month')['Line Total'].sum()
        
        fig = px.line(
            x=monthly_spend.index.astype(str),
            y=monthly_spend.values,
            title="Monthly Spend Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

def load_sample_data():
    """Generate comprehensive sample procurement data"""
    np.random.seed(42)
    
    vendors = [
        'Tech Solutions Inc', 'Office Supplies Corp', 'Industrial Materials Ltd',
        'Global Electronics', 'Professional Services LLC', 'Equipment Rental Co',
        'Quality Components', 'Logistics Partners', 'Manufacturing Supplies',
        'Digital Services Group'
    ]
    
    items = [
        'Laptop Computer', 'Office Chair', 'Steel Rod', 'Printer Cartridge',
        'Consulting Services', 'Forklift Rental', 'Electronic Component',
        'Shipping Service', 'Raw Material', 'Software License'
    ]
    
    n_records = 500
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    data = {
        'Vendor Name': np.random.choice(vendors, n_records),
        'Item': np.random.choice(range(1000, 2000), n_records),
        'Item Description': np.random.choice(items, n_records),
        'Unit Price': np.random.uniform(5, 500, n_records),
        'Qty Delivered': np.random.randint(1, 100, n_records),
        'Creation Date': pd.to_datetime(np.random.choice(pd.date_range(start_date, end_date), n_records)),
        'W/H': np.random.choice(['Warehouse A', 'Warehouse B', 'Warehouse C'], n_records),
        'Category': np.random.choice(['IT', 'Office', 'Manufacturing', 'Services'], n_records)
    }
    
    df = pd.DataFrame(data)
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    df['Qty Rejected'] = np.random.randint(0, df['Qty Delivered'] // 10 + 1, n_records)
    
    return df

def display_dashboard_overview(df):
    """Display high-level dashboard metrics"""
    st.header("📊 Procurement Analytics Dashboard")
    st.markdown("**Comprehensive procurement insights and optimization platform**")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_spend = df['Line Total'].sum()
        st.metric("Total Spend", f"${total_spend:,.0f}")
    
    with col2:
        unique_vendors = df['Vendor Name'].nunique()
        st.metric("Active Vendors", unique_vendors)
    
    with col3:
        unique_items = df['Item'].nunique()
        st.metric("Unique Items", unique_items)
    
    with col4:
        avg_order_value = df['Line Total'].mean()
        st.metric("Avg Order Value", f"${avg_order_value:.0f}")
    
    with col5:
        total_orders = len(df)
        st.metric("Total Orders", total_orders)
    
    # Quick insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Top vendors by spend
        top_vendors = df.groupby('Vendor Name')['Line Total'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_vendors.values,
            y=top_vendors.index,
            orientation='h',
            title="Top 10 Vendors by Spend"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly spend trend
        if 'Creation Date' in df.columns:
            monthly_spend = df.groupby(df['Creation Date'].dt.to_period('M'))['Line Total'].sum()
            
            fig = px.line(
                x=monthly_spend.index.astype(str),
                y=monthly_spend.values,
                title="Monthly Spend Trend"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.title("🛒 Procurement Analytics Platform")
    st.markdown("**Advanced procurement insights and optimization suite**")
    
    # Check module availability
    with st.sidebar:
        st.header("🔧 Module Status")
        available_modules = safe_import_modules()
        
        available_count = sum(1 for module in available_modules.values() if module is not None)
        total_count = len(available_modules)
        st.info(f"✅ {available_count}/{total_count} modules loaded successfully")
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Data source selection
        data_source = st.radio(
            "Choose data source:",
            ["📊 Sample Data", "📤 Upload CSV File"]
        )
        
        # Load data
        if data_source == "📊 Sample Data":
            df = load_sample_data()
            st.success(f"✅ Loaded {len(df)} sample records")
            
        else:  # Upload CSV
            uploaded_file = st.file_uploader(
                "Upload your procurement data (CSV)",
                type=['csv'],
                help="CSV should contain: Vendor Name, Item, Unit Price, Qty Delivered, Creation Date"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} records")
                    st.write("**Columns:**", list(df.columns))
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    df = load_sample_data()
                    st.info("Using sample data instead")
            else:
                df = load_sample_data()
                st.info("Using sample data")
        
        # Navigation menu
        st.header("🎯 Analytics Modules")
        
        # Build module list based on availability
        module_options = ["📊 Dashboard Overview"]
        
        if available_modules.get('lot_size'):
            module_options.append("📦 LOT Size Optimization")
        else:
            module_options.append("📦 LOT Size (Built-in)")
        
        if available_modules.get('contracting'):
            module_options.append("🤝 Contracting Opportunities")
        
        if available_modules.get('seasonal'):
            module_options.append("🌟 Seasonal Price Optimization")
        
        if available_modules.get('spend_anomaly'):
            module_options.append("📊 Spend Analysis & Anomalies")
        else:
            module_options.append("📊 Spend Analysis (Built-in)")
        
        if available_modules.get('duplicates'):
            module_options.append("🔍 Duplicate Detection")
        
        if available_modules.get('cross_region'):
            module_options.append("🌍 Cross-Region Analysis")
        
        if available_modules.get('reorder'):
            module_options.append("📈 Reorder Prediction")
        
        module_options.append("📋 Data Overview")
        
        selected_module = st.selectbox("Select Module:", module_options)
    
    # Main content area
    if df is not None and len(df) > 0:
        try:
            if selected_module == "📊 Dashboard Overview":
                display_dashboard_overview(df)
                
            elif selected_module == "📦 LOT Size Optimization":
                if available_modules.get('lot_size'):
                    available_modules['lot_size'].display(df)
                else:
                    st.error("LOT Size Optimization module not available")
                    
            elif selected_module == "📦 LOT Size (Built-in)":
                builtin_lot_size_optimization(df)
                
            elif selected_module == "🤝 Contracting Opportunities":
                if available_modules.get('contracting'):
                    available_modules['contracting'].display(df)
                else:
                    st.error("Contracting Opportunities module not available")
                    
            elif selected_module == "🌟 Seasonal Price Optimization":
                if available_modules.get('seasonal'):
                    available_modules['seasonal'].display(df)
                else:
                    st.error("Seasonal Price Optimization module not available")
                    
            elif selected_module == "📊 Spend Analysis & Anomalies":
                if available_modules.get('spend_anomaly'):
                    available_modules['spend_anomaly'].display(df)
                else:
                    st.error("Spend Analysis module not available")
                    
            elif selected_module == "📊 Spend Analysis (Built-in)":
                builtin_spend_analysis(df)
                
            elif selected_module == "🔍 Duplicate Detection":
                if available_modules.get('duplicates'):
                    available_modules['duplicates'].display(df)
                else:
                    st.error("Duplicate Detection module not available")
                    
            elif selected_module == "🌍 Cross-Region Analysis":
                if available_modules.get('cross_region'):
                    available_modules['cross_region'].display(df)
                else:
                    st.error("Cross-Region Analysis module not available")
                    
            elif selected_module == "📈 Reorder Prediction":
                if available_modules.get('reorder'):
                    available_modules['reorder'].display(df)
                else:
                    st.error("Reorder Prediction module not available")
                    
            elif selected_module == "📋 Data Overview":
                st.header("📊 Data Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Unique Vendors", df['Vendor Name'].nunique())
                with col3:
                    st.metric("Total Spend", f"${df['Line Total'].sum():,.0f}")
                with col4:
                    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Data Quality", f"{completeness:.1f}%")
                
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(20), use_container_width=True)
                
                # Column information
                st.subheader("📊 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': [str(df[col].dtype) for col in df.columns],
                    'Non-Null': [df[col].count() for col in df.columns],
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in module execution: {str(e)}")
            st.info("Please check your data format and try again.")
    
    else:
        st.error("No data available. Please load data first.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Procurement Analytics Platform** | Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
