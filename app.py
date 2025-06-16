import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None):
    """Load and cache the procurement data"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Uploaded file loaded successfully!")
            # Store filename in session state for reference
            st.session_state['data_source'] = uploaded_file.name
        else:
            # Try to load available files including the new dataset
            possible_files = [
                'PO_Model_Optimized_Large.csv',  # New comprehensive dataset
                'Combined_Structured_PO_Data 1.csv',
                'Cleansed_PO_Data_Model_Ready.csv',
                'procurement_data.csv',
                'po_data.csv'
            ]
            df = None
            loaded_file = None
            
            for file_path in possible_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    loaded_file = file_path
                    st.session_state['data_source'] = file_path
                    break
            
            if df is None:
                return None
            else:
                st.sidebar.info(f"📁 Auto-loaded: {loaded_file}")
        
        # Data preprocessing and validation with enhanced column mapping
        original_rows = len(df)
        
        # Map common column variations to standard names
        column_mappings = {
            # Date columns
            'Creation Date': ['Creation Date', 'creation_date', 'create_date', 'po_date'],
            'Approved Date': ['Approved Date', 'approved_date', 'approval_date'],
            'PO Receipt Date': ['PO Receipt Date', 'receipt_date', 'received_date'],
            'Requested Delivery Date': ['Requested Delivery Date', 'delivery_date'],
            'Promised Delivery Date': ['Promised Delivery Date', 'promised_date'],
            
            # Vendor columns
            'Vendor Name': ['Vendor Name', 'vendor_name', 'supplier_name', 'supplier'],
            'Vendor No': ['Vendor No', 'vendor_no', 'vendor_id', 'supplier_id'],
            
            # Item columns  
            'Item': ['Item', 'item', 'item_code', 'product_code'],
            'Item Description': ['Item Description', 'item_description', 'description', 'product_desc'],
            'Product Family': ['Product Family', 'product_family', 'category', 'product_category'],
            'Sub Product': ['Sub Product', 'sub_product', 'subcategory'],
            
            # Quantity columns
            'Qty Delivered': ['Qty Delivered', 'qty_delivered', 'quantity_delivered', 'delivered_qty'],
            'Qty Ordered': ['Qty Ordered', 'qty_ordered', 'quantity_ordered', 'ordered_qty'],
            'Qty Accepted': ['Qty Accepted', 'qty_accepted', 'accepted_qty'],
            'Qty Remaining': ['Qty Remaining', 'qty_remaining', 'remaining_qty'],
            
            # Financial columns
            'Unit Price': ['Unit Price', 'unit_price', 'price', 'cost'],
            'Line Total': ['Line Total', 'line_total', 'total', 'amount'],
            'Total In SAR': ['Total In SAR', 'total_sar', 'sar_total'],
            'Price In SAR': ['Price In SAR', 'price_sar', 'sar_price'],
            
            # Location/Regional columns
            'DEP': ['DEP', 'department', 'dept'],
            'SEC': ['SEC', 'section'],
            'W/H': ['W/H', 'warehouse', 'wh'],
            'China/Non-China': ['China/Non-China', 'region', 'location_type'],
            
            # Other important columns
            'Buyer': ['Buyer', 'buyer', 'purchaser'],
            'PO Status': ['PO Status', 'status', 'po_status'],
            'UOM': ['UOM', 'unit_of_measure', 'uom']
        }
        
        # Apply column mappings
        for standard_name, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in df.columns and standard_name not in df.columns:
                    df[standard_name] = df[possible_name]
                    break
        
        # Convert date columns with enhanced error handling
        date_columns = ['Creation Date', 'Approved Date', 'PO Receipt Date', 'Requested Delivery Date', 'Promised Delivery Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean numeric columns with enhanced error handling
        numeric_columns = ['Unit Price', 'Qty Delivered', 'Qty Ordered', 'Line Total', 'Total In SAR', 'Price In SAR', 'Qty Accepted', 'Qty Remaining']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate Line Total if missing (try multiple quantity columns)
        if 'Line Total' not in df.columns or df['Line Total'].isna().all():
            if 'Unit Price' in df.columns:
                qty_col = None
                for qty_name in ['Qty Delivered', 'Qty Accepted', 'Qty Ordered']:
                    if qty_name in df.columns and not df[qty_name].isna().all():
                        qty_col = qty_name
                        break
                
                if qty_col:
                    df['Line Total'] = df['Unit Price'] * df[qty_col]
                    st.sidebar.info(f"📊 Calculated Line Total using {qty_col}")
        
        # Data quality summary with comprehensive checks
        essential_columns = ['Vendor Name', 'Unit Price']
        qty_columns = ['Qty Delivered', 'Qty Ordered', 'Qty Accepted']
        
        # Find the best quantity column
        best_qty_col = None
        for qty_col in qty_columns:
            if qty_col in df.columns:
                non_null_count = df[qty_col].notna().sum()
                if non_null_count > 0:
                    best_qty_col = qty_col
                    break
        
        if best_qty_col:
            essential_columns.append(best_qty_col)
        
        clean_rows = len(df.dropna(subset=essential_columns, how='any'))
        data_quality = (clean_rows / original_rows) * 100 if original_rows > 0 else 0
        
        # Store data quality info in session state
        st.session_state['data_quality'] = {
            'total_rows': original_rows,
            'clean_rows': clean_rows,
            'quality_percentage': data_quality,
            'columns_available': list(df.columns),
            'key_columns_present': [col for col in essential_columns if col in df.columns]
        }
        
        st.sidebar.metric("Data Quality", f"{data_quality:.1f}%", f"{clean_rows:,} / {original_rows:,} rows")
        
        # Show available key columns
        key_cols_available = len([col for col in ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date', 'Line Total'] if col in df.columns])
        st.sidebar.metric("Key Columns Available", f"{key_cols_available}/6")
        
        return df
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {str(e)}")
        st.sidebar.info("💡 Check that your CSV file has the expected columns")
        return None

def apply_filters(df):
    """Apply filters to the dataframe based on sidebar selections"""
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Date range filter
    if 'Creation Date' in df.columns and df['Creation Date'].notna().any():
        date_min = df['Creation Date'].min()
        date_max = df['Creation Date'].max()
        
        date_range = st.sidebar.date_input(
            "📅 Date Range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['Creation Date'] >= pd.Timestamp(start_date)) &
                (filtered_df['Creation Date'] <= pd.Timestamp(end_date))
            ]
    
    # Vendor filter
    if 'Vendor Name' in df.columns:
        vendors = sorted(df['Vendor Name'].dropna().unique())
        selected_vendors = st.sidebar.multiselect(
            "🏢 Select Vendors",
            vendors,
            default=vendors[:10] if len(vendors) > 10 else vendors
        )
        if selected_vendors:
            filtered_df = filtered_df[filtered_df['Vendor Name'].isin(selected_vendors)]
    
    # Product Family filter
    if 'Product Family' in df.columns:
        families = sorted(df['Product Family'].dropna().unique())
        selected_families = st.sidebar.multiselect(
            "📦 Product Families",
            families,
            default=families
        )
        if selected_families:
            filtered_df = filtered_df[filtered_df['Product Family'].isin(selected_families)]
    
    # Department filter
    if 'DEP' in df.columns:
        departments = sorted(df['DEP'].dropna().unique())
        selected_depts = st.sidebar.multiselect(
            "🏭 Departments",
            departments,
            default=departments
        )
        if selected_depts:
            filtered_df = filtered_df[filtered_df['DEP'].isin(selected_depts)]
    
    # Buyer filter
    if 'Buyer' in df.columns:
        buyers = sorted(df['Buyer'].dropna().unique())
        selected_buyers = st.sidebar.multiselect(
            "👤 Buyers",
            buyers,
            default=buyers
        )
        if selected_buyers:
            filtered_df = filtered_df[filtered_df['Buyer'].isin(selected_buyers)]
    
    return filtered_df

def display_key_metrics(df):
    """Display key procurement metrics"""
    if df is None or df.empty:
        st.warning("No data available for metrics calculation")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_orders = len(df)
        st.metric("📋 Total Orders", f"{total_orders:,}")
    
    with col2:
        if 'Line Total' in df.columns:
            total_value = df['Line Total'].sum()
            st.metric("💰 Total Value", f"${total_value:,.0f}")
        else:
            st.metric("💰 Total Value", "N/A")
    
    with col3:
        unique_vendors = df['Vendor Name'].nunique() if 'Vendor Name' in df.columns else 0
        st.metric("🏢 Unique Vendors", f"{unique_vendors:,}")
    
    with col4:
        if 'Unit Price' in df.columns:
            avg_unit_price = df['Unit Price'].mean()
            st.metric("💵 Avg Unit Price", f"${avg_unit_price:.2f}")
        else:
            st.metric("💵 Avg Unit Price", "N/A")
    
    with col5:
        if 'Qty Delivered' in df.columns:
            total_qty = df['Qty Delivered'].sum()
            st.metric("📦 Total Quantity", f"{total_qty:,.0f}")
        elif 'Qty Ordered' in df.columns:
            total_qty = df['Qty Ordered'].sum()
            st.metric("📦 Total Quantity", f"{total_qty:,.0f}")
        else:
            st.metric("📦 Total Quantity", "N/A")

def show_overview_dashboard(df):
    """Display overview dashboard with key visualizations"""
    st.header("📊 Overview Dashboard")
    
    if df is None or df.empty:
        st.warning("No data available for dashboard")
        return
    
    # Key metrics
    display_key_metrics(df)
    
    st.markdown("---")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Vendor analysis
        if 'Vendor Name' in df.columns and 'Line Total' in df.columns:
            vendor_analysis = df.groupby('Vendor Name')['Line Total'].sum().sort_values(ascending=False).head(10)
            
            fig_vendor = px.bar(
                x=vendor_analysis.values,
                y=vendor_analysis.index,
                orientation='h',
                title="🏢 Top 10 Vendors by Total Value",
                labels={'x': 'Total Value ($)', 'y': 'Vendor Name'}
            )
            fig_vendor.update_layout(height=400)
            st.plotly_chart(fig_vendor, use_container_width=True)
    
    with col2:
        # Product family analysis
        if 'Product Family' in df.columns and 'Line Total' in df.columns:
            family_analysis = df.groupby('Product Family')['Line Total'].sum().sort_values(ascending=False)
            
            fig_family = px.pie(
                values=family_analysis.values,
                names=family_analysis.index,
                title="📦 Spending by Product Family"
            )
            fig_family.update_layout(height=400)
            st.plotly_chart(fig_family, use_container_width=True)
    
    # Time series analysis
    if 'Creation Date' in df.columns and 'Line Total' in df.columns:
        st.subheader("📈 Spending Trends Over Time")
        
        # Monthly aggregation
        df_time = df.copy()
        df_time['Year_Month'] = df_time['Creation Date'].dt.to_period('M')
        monthly_spending = df_time.groupby('Year_Month')['Line Total'].sum()
        
        fig_time = px.line(
            x=monthly_spending.index.astype(str),
            y=monthly_spending.values,
            title="Monthly Spending Trend",
            labels={'x': 'Month', 'y': 'Total Spending ($)'}
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)

def show_vendor_analysis(df):
    """Display detailed vendor analysis"""
    st.header("🏢 Vendor Analysis")
    
    if df is None or df.empty or 'Vendor Name' not in df.columns:
        st.warning("No vendor data available for analysis")
        return
    
    # Vendor performance metrics
    agg_dict = {
        'Line Total': ['sum', 'mean', 'count'],
        'Unit Price': 'mean'
    }
    
    # Add quantity column based on availability
    if 'Qty Delivered' in df.columns:
        agg_dict['Qty Delivered'] = 'sum'
    elif 'Qty Ordered' in df.columns:
        agg_dict['Qty Ordered'] = 'sum'
    
    vendor_metrics = df.groupby('Vendor Name').agg(agg_dict).round(2)
    
    vendor_metrics.columns = ['Total Value', 'Avg Order Value', 'Order Count', 'Avg Unit Price', 'Total Quantity']
    vendor_metrics = vendor_metrics.sort_values('Total Value', ascending=False)
    
    # Top vendors summary
    st.subheader("📋 Top Vendors Summary")
    st.dataframe(vendor_metrics.head(20), use_container_width=True)
    
    # Vendor comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Order frequency
        order_freq = df['Vendor Name'].value_counts().head(15)
        fig_freq = px.bar(
            x=order_freq.values,
            y=order_freq.index,
            orientation='h',
            title="📊 Order Frequency by Vendor",
            labels={'x': 'Number of Orders', 'y': 'Vendor Name'}
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col2:
        # Average order value
        avg_order_value = df.groupby('Vendor Name')['Line Total'].mean().sort_values(ascending=False).head(15)
        fig_avg = px.bar(
            x=avg_order_value.values,
            y=avg_order_value.index,
            orientation='h',
            title="💰 Average Order Value by Vendor",
            labels={'x': 'Average Order Value ($)', 'y': 'Vendor Name'}
        )
        st.plotly_chart(fig_avg, use_container_width=True)

def show_product_analysis(df):
    """Display detailed product analysis"""
    st.header("📦 Product Analysis")
    
    if df is None or df.empty:
        st.warning("No product data available for analysis")
        return
    
    # Product family analysis
    if 'Product Family' in df.columns:
        st.subheader("📊 Product Family Analysis")
        
        agg_dict = {
            'Line Total': ['sum', 'mean', 'count'],
            'Unit Price': 'mean'
        }
        
        # Add quantity column based on availability
        if 'Qty Delivered' in df.columns:
            agg_dict['Qty Delivered'] = 'sum'
        elif 'Qty Ordered' in df.columns:
            agg_dict['Qty Ordered'] = 'sum'
        
        family_metrics = df.groupby('Product Family').agg(agg_dict).round(2)
        
        family_metrics.columns = ['Total Value', 'Avg Order Value', 'Order Count', 'Avg Unit Price', 'Total Quantity']
        family_metrics = family_metrics.sort_values('Total Value', ascending=False)
        
        st.dataframe(family_metrics, use_container_width=True)
        
        # Family spending trend
        if 'Creation Date' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly trend by family
                df_trend = df.copy()
                df_trend['Year_Month'] = df_trend['Creation Date'].dt.to_period('M')
                family_trend = df_trend.groupby(['Year_Month', 'Product Family'])['Line Total'].sum().reset_index()
                family_trend['Year_Month'] = family_trend['Year_Month'].astype(str)
                
                fig_trend = px.line(
                    family_trend,
                    x='Year_Month',
                    y='Line Total',
                    color='Product Family',
                    title="📈 Monthly Spending by Product Family"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Top items analysis
                if 'Item Description' in df.columns:
                    top_items = df.groupby('Item Description')['Line Total'].sum().sort_values(ascending=False).head(10)
                    
                    fig_items = px.bar(
                        x=top_items.values,
                        y=top_items.index,
                        orientation='h',
                        title="🔝 Top 10 Items by Value",
                        labels={'x': 'Total Value ($)', 'y': 'Item Description'}
                    )
                    st.plotly_chart(fig_items, use_container_width=True)

def show_geographical_analysis(df):
    """Display geographical and departmental analysis"""
    st.header("🌍 Geographical & Departmental Analysis")
    
    if df is None or df.empty:
        st.warning("No geographical data available for analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Department analysis
        if 'DEP' in df.columns:
            dept_analysis = df.groupby('DEP').agg({
                'Line Total': ['sum', 'count'],
                'Vendor Name': 'nunique'
            }).round(2)
            dept_analysis.columns = ['Total Spending', 'Order Count', 'Unique Vendors']
            dept_analysis = dept_analysis.sort_values('Total Spending', ascending=False)
            
            st.subheader("🏭 Department Analysis")
            st.dataframe(dept_analysis, use_container_width=True)
            
            # Department spending chart
            fig_dept = px.bar(
                x=dept_analysis.index,
                y=dept_analysis['Total Spending'],
                title="💰 Spending by Department",
                labels={'x': 'Department', 'y': 'Total Spending ($)'}
            )
            st.plotly_chart(fig_dept, use_container_width=True)
    
    with col2:
        # Regional analysis
        if 'China/Non-China' in df.columns:
            region_analysis = df.groupby('China/Non-China').agg({
                'Line Total': ['sum', 'count'],
                'Vendor Name': 'nunique'
            }).round(2)
            region_analysis.columns = ['Total Spending', 'Order Count', 'Unique Vendors']
            
            st.subheader("🌏 Regional Analysis")
            st.dataframe(region_analysis, use_container_width=True)
            
            # Regional pie chart
            fig_region = px.pie(
                values=region_analysis['Total Spending'],
                names=region_analysis.index,
                title="🥧 Spending by Region"
            )
            st.plotly_chart(fig_region, use_container_width=True)
        
        # Warehouse analysis
        if 'W/H' in df.columns:
            wh_analysis = df.groupby('W/H')['Line Total'].sum().sort_values(ascending=False).head(10)
            
            fig_wh = px.bar(
                x=wh_analysis.index,
                y=wh_analysis.values,
                title="🏪 Top Warehouses by Spending",
                labels={'x': 'Warehouse', 'y': 'Total Spending ($)'}
            )
            st.plotly_chart(fig_wh, use_container_width=True)

def show_advanced_analytics(df):
    """Display advanced analytics and insights"""
    st.header("🔬 Advanced Analytics")
    
    if df is None or df.empty:
        st.warning("No data available for advanced analytics")
        return
    
    # Price trend analysis
    if 'Creation Date' in df.columns and 'Unit Price' in df.columns:
        st.subheader("📈 Price Trend Analysis")
        
        price_trend = df.groupby(df['Creation Date'].dt.to_period('M'))['Unit Price'].mean()
        
        fig_price_trend = px.line(
            x=price_trend.index.astype(str),
            y=price_trend.values,
            title="Average Unit Price Trend Over Time",
            labels={'x': 'Month', 'y': 'Average Unit Price ($)'}
        )
        st.plotly_chart(fig_price_trend, use_container_width=True)
    
    # Delivery performance analysis
    if 'Requested Delivery Date' in df.columns and 'PO Receipt Date' in df.columns:
        st.subheader("🚚 Delivery Performance Analysis")
        
        df_delivery = df.dropna(subset=['Requested Delivery Date', 'PO Receipt Date'])
        df_delivery['Delivery_Days'] = (df_delivery['PO Receipt Date'] - df_delivery['Requested Delivery Date']).dt.days
        
        # On-time delivery rate
        on_time = (df_delivery['Delivery_Days'] <= 0).sum()
        total_deliveries = len(df_delivery)
        on_time_rate = (on_time / total_deliveries) * 100 if total_deliveries > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ On-Time Delivery Rate", f"{on_time_rate:.1f}%")
        with col2:
            avg_delay = df_delivery['Delivery_Days'].mean()
            st.metric("⏱️ Average Delivery Delay", f"{avg_delay:.1f} days")
        
        # Delivery performance by vendor
        if 'Vendor Name' in df.columns:
            vendor_delivery = df_delivery.groupby('Vendor Name').agg({
                'Delivery_Days': ['mean', 'count']
            }).round(1)
            vendor_delivery.columns = ['Avg Delivery Days', 'Order Count']
            vendor_delivery = vendor_delivery[vendor_delivery['Order Count'] >= 5].sort_values('Avg Delivery Days')
            
            st.subheader("🏢 Vendor Delivery Performance")
            st.dataframe(vendor_delivery.head(20), use_container_width=True)

def show_data_explorer(df):
    """Display data explorer with raw data and search functionality"""
    st.header("🔍 Data Explorer")
    
    if df is None or df.empty:
        st.warning("No data available to explore")
        return
    
    # Search functionality
    search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
    
    # Column selection
    available_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "📋 Select columns to display:",
        available_columns,
        default=available_columns[:10] if len(available_columns) > 10 else available_columns
    )
    
    # Apply search filter
    display_df = df.copy()
    if search_term:
        mask = False
        for col in df.select_dtypes(include=['object']).columns:
            mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
        display_df = df[mask]
    
    # Display filtered data
    if selected_columns:
        display_df = display_df[selected_columns]
    
    st.write(f"📊 Showing {len(display_df):,} rows")
    st.dataframe(display_df, use_container_width=True, height=600)
    
    # Download functionality
    if st.button("📥 Download Filtered Data as CSV"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Click to Download",
            data=csv,
            file_name=f"filtered_procurement_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main application function"""
    # App title and description
    st.markdown('<div class="main-header">📊 Procurement Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Comprehensive analysis of procurement and purchase order data with advanced insights**")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # File upload
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your procurement data (CSV)",
        type=['csv'],
        help="Upload a CSV file with procurement/PO data"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None:
        st.error("❌ No data available. Please upload a CSV file or ensure data files exist in the directory.")
        st.info("💡 **Expected data columns:** Vendor Name, Item, Unit Price, Qty Delivered, Creation Date, Line Total")
        return
    
    # Apply filters
    st.sidebar.subheader("🔧 Filters")
    filtered_df = apply_filters(df)
    
    # Navigation
    st.sidebar.subheader("📑 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "📊 Overview Dashboard",
            "🏢 Vendor Analysis", 
            "📦 Product Analysis",
            "🌍 Geographical Analysis",
            "🔬 Advanced Analytics",
            "🔍 Data Explorer"
        ]
    )
    
    # Data info sidebar
    st.sidebar.subheader("ℹ️ Data Information")
    if 'data_quality' in st.session_state:
        quality_info = st.session_state['data_quality']
        st.sidebar.info(f"**Source:** {st.session_state.get('data_source', 'Unknown')}")
        st.sidebar.info(f"**Columns:** {len(quality_info['columns_available'])}")
        
        # Show filtered data info
        st.sidebar.info(f"**Filtered Rows:** {len(filtered_df):,} / {len(df):,}")
    
    # Display selected page
    if page == "📊 Overview Dashboard":
        show_overview_dashboard(filtered_df)
    elif page == "🏢 Vendor Analysis":
        show_vendor_analysis(filtered_df)
    elif page == "📦 Product Analysis":
        show_product_analysis(filtered_df)
    elif page == "🌍 Geographical Analysis":
        show_geographical_analysis(filtered_df)
    elif page == "🔬 Advanced Analytics":
        show_advanced_analytics(filtered_df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("*💡 Dashboard built with Streamlit for comprehensive procurement data analysis*")

if __name__ == "__main__":
    main()
