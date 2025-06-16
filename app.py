import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Import the enhanced modules
import cross_region
import duplicates
import reorder_prediction

# Page configuration
st.set_page_config(
    page_title="Procurement Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f6;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None):
    """Load and cache the procurement data"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Try to load available files
            possible_files = [
                'Combined_Structured_PO_Data 1.csv',
                'Cleansed_PO_Data_Model_Ready.csv'
            ]
            df = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    st.sidebar.info(f"📁 Loaded: {file_path}")
                    break
            
            if df is None:
                return None
        
        # Data preprocessing
        df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce')
        df['Approved Date'] = pd.to_datetime(df['Approved Date'], errors='coerce')
        df['PO Receipt Date'] = pd.to_datetime(df['PO Receipt Date'], errors='coerce')
        
        # Clean numeric columns
        numeric_cols = ['Unit Price', 'Qty Delivered', 'Qty Ordered', 'Line Total']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def show_overview_dashboard(df):
    """Display the overview dashboard with key metrics"""
    st.markdown('<h1 class="main-header">📊 Procurement Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_pos = len(df)
        st.metric("Total POs", f"{total_pos:,}")
    
    with col2:
        total_value = df['Line Total'].sum() if 'Line Total' in df.columns else 0
        st.metric("Total Value (SAR)", f"{total_value:,.0f}")
    
    with col3:
        unique_vendors = df['Vendor Name'].nunique()
        st.metric("Active Vendors", f"{unique_vendors:,}")
    
    with col4:
        unique_items = df['Item'].nunique()
        st.metric("Unique Items", f"{unique_items:,}")
    
    with col5:
        avg_lead_time = df['lead_time_days'].mean() if 'lead_time_days' in df.columns else 0
        st.metric("Avg Lead Time", f"{avg_lead_time:.1f} days")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 PO Value Trend")
        if 'Creation Date' in df.columns and 'Line Total' in df.columns:
            monthly_data = df.groupby(df['Creation Date'].dt.to_period('M'))['Line Total'].sum().reset_index()
            monthly_data['Creation Date'] = monthly_data['Creation Date'].astype(str)
            
            fig = px.line(monthly_data, x='Creation Date', y='Line Total',
                         title="Monthly PO Value Trend",
                         labels={'Line Total': 'Total Value (SAR)', 'Creation Date': 'Month'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Top Vendors by Value")
        if 'Vendor Name' in df.columns and 'Line Total' in df.columns:
            top_vendors = df.groupby('Vendor Name')['Line Total'].sum().nlargest(10).reset_index()
            
            fig = px.bar(top_vendors, x='Line Total', y='Vendor Name',
                        orientation='h', title="Top 10 Vendors by Total Value",
                        labels={'Line Total': 'Total Value (SAR)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 PO Status Distribution")
        if 'PO Status' in df.columns:
            status_counts = df['PO Status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        title="Distribution of PO Status")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Regional Distribution")
        if 'W/H' in df.columns:
            warehouse_counts = df['W/H'].value_counts().head(10)
            fig = px.bar(x=warehouse_counts.index, y=warehouse_counts.values,
                        title="Top 10 Warehouses by PO Count",
                        labels={'x': 'Warehouse', 'y': 'Number of POs'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Sidebar for navigation
    st.sidebar.title("🚀 Navigation")
    
    # Data loading section
    st.sidebar.subheader("📁 Data Loading")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None:
        st.error("⚠️ No data loaded. Please upload a CSV file or ensure 'Cleansed_PO_Data_Model_Ready.csv' exists in the current directory.")
        st.info("💡 Upload your procurement data CSV file using the sidebar to get started.")
        return
    
    # Data info
    st.sidebar.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Navigation menu
    page = st.sidebar.radio(
        "Select Analysis",
        ["🏠 Overview Dashboard", "📈 Smart Reorder Predictions", "🔄 Cross-Region Vendor Optimization", "🔍 Duplicate Detection"]
    )
    
    # Filter options
    st.sidebar.subheader("🔧 Data Filters")
    
    # Date range filter
    if 'Creation Date' in df.columns:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['Creation Date'].min(), df['Creation Date'].max()),
            min_value=df['Creation Date'].min(),
            max_value=df['Creation Date'].max()
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Creation Date'] >= pd.Timestamp(start_date)) & 
                   (df['Creation Date'] <= pd.Timestamp(end_date))]
    
    # Vendor filter
    if 'Vendor Name' in df.columns:
        selected_vendors = st.sidebar.multiselect(
            "Select Vendors",
            options=sorted(df['Vendor Name'].dropna().unique()),
            default=None
        )
        if selected_vendors:
            df = df[df['Vendor Name'].isin(selected_vendors)]
    
    # Warehouse filter
    if 'W/H' in df.columns:
        selected_warehouses = st.sidebar.multiselect(
            "Select Warehouses",
            options=sorted(df['W/H'].dropna().unique()),
            default=None
        )
        if selected_warehouses:
            df = df[df['W/H'].isin(selected_warehouses)]
    
    # Display selected page
    if page == "🏠 Overview Dashboard":
        show_overview_dashboard(df)
    elif page == "📈 Smart Reorder Predictions":
        reorder_prediction.display(df)
    elif page == "🔄 Cross-Region Vendor Optimization":
        cross_region.display(df)
    elif page == "🔍 Duplicate Detection":
        duplicates.display(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Procurement Analytics Dashboard v1.0**")
    st.sidebar.markdown("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
