import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import re
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from math import sqrt
warnings.filterwarnings('ignore')

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
def generate_sample_data(num_records=1000):
    """Generate comprehensive sample procurement data with multiple currencies for testing"""
    np.random.seed(42)
    
    # Sample data parameters with multiple currencies
    vendors = [
        "Global Tech Solutions", "Premium Office Supplies", "Industrial Materials Corp",
        "Elite Manufacturing", "Smart Logistics Ltd", "Quality Parts Inc",
        "Advanced Systems", "Reliable Services", "Premier Components", "Efficient Operations",
        "Innovation Partners", "Strategic Suppliers", "Excellence Group", "Dynamic Solutions",
        "Professional Resources"
    ]
    
    items = list(range(1, 51))  # Item IDs 1-50
    
    # Multi-currency setup
    currencies = ['USD', 'EUR', 'GBP', 'SAR', 'AED', 'QAR', 'INR', 'CNY', 'JPY']
    currency_weights = [0.30, 0.15, 0.10, 0.25, 0.08, 0.04, 0.03, 0.03, 0.02]
    
    item_descriptions = [
        "High-performance laptop computer", "Ergonomic office chair with lumbar support",
        "Stainless steel fabrication material", "Electronic circuit board components",
        "Professional cleaning and janitorial supplies", "Personal protective equipment",
        "Enterprise network router and switches", "Premium office stationery set",
        "Industrial maintenance and repair tools", "Scientific laboratory equipment",
        "Enterprise software licensing", "High-quality printing and copying materials",
        "Heavy-duty industrial machinery", "Protective packaging materials",
        "Professional transportation and logistics services"
    ] * 4  # Repeat to have enough descriptions
    
    product_families = [
        "IT & Technology", "Office Supplies", "Raw Materials", "Electronics",
        "Facilities", "Safety", "Infrastructure", "Stationery",
        "Maintenance & Repair", "Laboratory", "Software", "Marketing",
        "Manufacturing", "Packaging", "Services"
    ]
    
    departments = ["IT", "Operations", "Facilities", "R&D", "Marketing", "HR", "Finance", "Production"]
    sections = ["SEC-A", "SEC-B", "SEC-C", "SEC-D", "SEC-E"]
    warehouses = ["WH-001", "WH-002", "WH-003", "WH-004", "WH-005"]
    buyers = ["John Smith", "Sarah Johnson", "Mike Chen", "Lisa Brown", "David Wilson"]
    regions = ["China", "Non-China"]
    statuses = ["Approved", "Delivered", "Pending", "Completed"]
    
    # Generate dates over the last 2 years
    start_date = datetime.now() - timedelta(days=730)
    end_date = datetime.now()
    
    data = []
    
    for i in range(num_records):
        # Create seasonal price variations
        creation_date = start_date + timedelta(days=np.random.randint(0, 730))
        month = creation_date.month
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * month / 12)  # +/- 10% seasonal variation
        
        # Select currency for this transaction
        currency = np.random.choice(currencies, p=currency_weights)
        
        # Base unit price with currency-specific ranges
        if currency == 'USD':
            base_price = np.random.uniform(5, 500) * seasonal_factor
            price_format = f"${base_price:.2f}" if np.random.random() < 0.4 else base_price
        elif currency == 'EUR':
            base_price = np.random.uniform(4, 450) * seasonal_factor
            price_format = f"€{base_price:.2f}" if np.random.random() < 0.4 else base_price
        elif currency == 'GBP':
            base_price = np.random.uniform(3, 400) * seasonal_factor
            price_format = f"£{base_price:.2f}" if np.random.random() < 0.4 else base_price
        elif currency == 'SAR':
            base_price = np.random.uniform(20, 1800) * seasonal_factor
            price_format = f"{base_price:.2f} SAR" if np.random.random() < 0.3 else base_price
        elif currency == 'AED':
            base_price = np.random.uniform(18, 1700) * seasonal_factor
            price_format = f"{base_price:.2f} AED" if np.random.random() < 0.3 else base_price
        elif currency == 'INR':
            base_price = np.random.uniform(400, 40000) * seasonal_factor
            price_format = f"₹{base_price:.2f}" if np.random.random() < 0.3 else base_price
        elif currency == 'CNY':
            base_price = np.random.uniform(35, 3500) * seasonal_factor
            price_format = f"¥{base_price:.2f}" if np.random.random() < 0.3 else base_price
        elif currency == 'JPY':
            base_price = np.random.uniform(500, 70000) * seasonal_factor
            price_format = f"¥{base_price:.0f}" if np.random.random() < 0.3 else base_price
        else:
            base_price = np.random.uniform(10, 1000) * seasonal_factor
            price_format = base_price
        
        # Quantity with some correlation to price (higher price = lower quantity typically)
        qty_delivered = max(1, int(np.random.exponential(50) * (100 / (base_price / 10 if base_price > 10 else 1))))
        qty_ordered = max(qty_delivered, qty_delivered + np.random.randint(0, 10))
        
        # Generate delivery dates
        delivery_delay = np.random.randint(-5, 30)  # Can be early or late
        po_receipt_date = creation_date + timedelta(days=delivery_delay)
        
        # Approval date (before creation)
        approved_date = creation_date - timedelta(days=np.random.randint(1, 10))
        
        # Requested vs promised delivery
        requested_delivery = creation_date + timedelta(days=np.random.randint(7, 21))
        promised_delivery = requested_delivery + timedelta(days=np.random.randint(-2, 7))
        
        vendor_name = np.random.choice(vendors)
        item_id = np.random.choice(items)
        
        record = {
            'Creation Date': creation_date,
            'Approved Date': approved_date,
            'PO Receipt Date': po_receipt_date,
            'Requested Delivery Date': requested_delivery,
            'Promised Delivery Date': promised_delivery,
            'Vendor Name': vendor_name,
            'Vendor No': f"V{hash(vendor_name) % 10000:04d}",
            'Item': item_id,
            'Item Description': np.random.choice(item_descriptions),
            'Product Family': np.random.choice(product_families),
            'Sub Product': f"Sub-{np.random.choice(['A', 'B', 'C', 'D'])}",
            'Qty Delivered': qty_delivered,
            'Qty Ordered': qty_ordered,
            'Qty Accepted': max(0, qty_delivered - np.random.randint(0, 3)),
            'Qty Remaining': max(0, qty_ordered - qty_delivered),
            'Unit Price': price_format,  # This might contain currency symbols/text
            'Currency': currency if np.random.random() < 0.7 else None,  # 70% have explicit currency column
            'Line Total': None,  # Will be calculated after currency conversion
            'Total In SAR': None,  # Will be calculated after currency conversion
            'Price In SAR': None,  # Will be calculated after currency conversion
            'DEP': np.random.choice(departments),
            'SEC': np.random.choice(sections),
            'W/H': np.random.choice(warehouses),
            'China/Non-China': np.random.choice(regions),
            'Buyer': np.random.choice(buyers),
            'PO Status': np.random.choice(statuses),
            'UOM': np.random.choice(['Each', 'Box', 'Kg', 'Meter', 'Liter'])
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

class CurrencyConverter:
    """Advanced currency converter for procurement analytics"""
    
    def __init__(self):
        self.base_currency = 'SAR'
        self.exchange_rates = {}
        self.last_updated = None
        self.currency_symbols = {
            'USD': ['
        
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
        st.sidebar.info("💡 Generating sample data for demonstration")
        df = generate_sample_data()
        st.session_state['data_source'] = 'Generated Sample Data (Error Recovery)'
        return df

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

# ADVANCED MODULES INTEGRATION

def show_overview_dashboard(df):
    """Display overview dashboard with key visualizations"""
    st.header("📊 Overview Dashboard")
    
    if df is None or df.empty:
        st.warning("No data available for dashboard")
        return
    
    # Currency conversion summary if available
    if 'currency_conversions' in st.session_state and st.session_state['currency_conversions'] > 0:
        st.info(f"💱 **Multi-Currency Data Detected:** {st.session_state['currency_conversions']} values automatically converted to SAR using current exchange rates.")
    
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
                title="🏢 Top 10 Vendors by Total Value (SAR)",
                labels={'x': 'Total Value (SAR)', 'y': 'Vendor Name'}
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
                title="📦 Spending by Product Family (SAR)"
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
            title="Monthly Spending Trend (SAR)",
            labels={'x': 'Month', 'y': 'Total Spending (SAR)'}
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Currency breakdown if conversions were made
    if 'currency_conversions' in st.session_state and st.session_state['currency_conversions'] > 0:
        st.subheader("💱 Currency Conversion Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            converter = CurrencyConverter()
            converter.get_exchange_rates()
            
            # Show exchange rates used
            rates_data = {
                'Currency': ['USD', 'EUR', 'GBP', 'AED', 'QAR', 'INR'],
                'Rate to SAR': [
                    converter.exchange_rates.get('USD', 0),
                    converter.exchange_rates.get('EUR', 0),
                    converter.exchange_rates.get('GBP', 0),
                    converter.exchange_rates.get('AED', 0),
                    converter.exchange_rates.get('QAR', 0),
                    converter.exchange_rates.get('INR', 0)
                ]
            }
            
            rates_df = pd.DataFrame(rates_data)
            st.dataframe(rates_df.style.format({'Rate to SAR': '{:.4f}'}), use_container_width=True)
        
        with col2:
            st.metric("Conversion Method", "Real-time Exchange Rates")
            st.metric("Base Currency", "Saudi Riyal (SAR)")
            st.metric("Last Updated", datetime.now().strftime('%Y-%m-%d %H:%M'))
            
            if st.button("🔄 Refresh Exchange Rates"):
                st.rerun()

def show_contracting_opportunities(df):
    """Contracting Opportunities Module"""
    st.header("🤝 Contracting Opportunities")
    st.markdown("Identify optimal contracting opportunities based on spend analysis, vendor performance, and demand predictability.")
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Configuration parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_spend = st.number_input("Min Annual Spend Threshold", min_value=0, value=50000, step=10000)
    with col2:
        min_frequency = st.number_input("Min Annual Order Frequency", min_value=1, value=6, step=1)
    with col3:
        analysis_period = st.selectbox("Analysis Period", ["All Data", "Last 12 Months", "Last 6 Months"])
    
    # Filter data by period
    if analysis_period == "Last 12 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=365)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    elif analysis_period == "Last 6 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=180)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    else:
        analysis_df = df_clean
    
    if st.button("🔍 Identify Contract Opportunities", type="primary"):
        with st.spinner("Analyzing contract opportunities..."):
            contract_opportunities = []
            
            # Analyze vendor-item combinations
            vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
            
            for (vendor, item), group_data in vendor_item_combinations:
                # Calculate metrics
                total_spend = (group_data['Unit Price'] * group_data['Qty Delivered']).sum()
                order_frequency = len(group_data)
                
                # Calculate time span
                date_range = group_data['Creation Date'].max() - group_data['Creation Date'].min()
                months_span = date_range.days / 30 if date_range.days > 0 else 1
                monthly_frequency = order_frequency / months_span
                
                # Demand predictability
                monthly_demand = group_data.groupby(group_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
                demand_cv = monthly_demand.std() / monthly_demand.mean() if len(monthly_demand) > 1 and monthly_demand.mean() > 0 else 1
                demand_predictability = max(0, 1 - demand_cv)
                
                # Contract suitability score
                spend_score = min(total_spend / min_spend, 1.0) if min_spend > 0 else 1.0
                frequency_score = min(monthly_frequency / (min_frequency / 12), 1.0)
                
                suitability_score = (spend_score * 0.4 + frequency_score * 0.3 + demand_predictability * 0.3)
                
                # Contract recommendation
                if suitability_score >= 0.7 and total_spend >= min_spend:
                    recommendation = "High Priority"
                elif suitability_score >= 0.5 and total_spend >= min_spend * 0.5:
                    recommendation = "Medium Priority"
                elif suitability_score >= 0.3:
                    recommendation = "Low Priority"
                else:
                    recommendation = "Not Suitable"
                
                if recommendation != "Not Suitable":
                    item_desc = group_data['Item Description'].iloc[0] if 'Item Description' in group_data.columns else f"Item {item}"
                    
                    contract_opportunities.append({
                        'Vendor Name': vendor,
                        'Item': item,
                        'Item Description': item_desc[:50] + "..." if len(item_desc) > 50 else item_desc,
                        'Annual Spend': total_spend,
                        'Order Frequency': order_frequency,
                        'Monthly Frequency': monthly_frequency,
                        'Demand Predictability': demand_predictability,
                        'Suitability Score': suitability_score,
                        'Contract Priority': recommendation,
                        'Avg Unit Price': group_data['Unit Price'].mean(),
                    })
            
            if contract_opportunities:
                opportunities_df = pd.DataFrame(contract_opportunities)
                opportunities_df = opportunities_df.sort_values(['Suitability Score', 'Annual Spend'], ascending=[False, False])
                
                # Summary metrics
                total_contract_spend = opportunities_df['Annual Spend'].sum()
                high_priority_count = len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])
                avg_suitability = opportunities_df['Suitability Score'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Opportunities", len(opportunities_df))
                with col2:
                    st.metric("High Priority Items", high_priority_count)
                with col3:
                    st.metric("Total Contract Spend", f"${total_contract_spend:,.0f}")
                with col4:
                    st.metric("Avg Suitability Score", f"{avg_suitability:.2f}")
                
                # Priority distribution
                priority_counts = opportunities_df['Contract Priority'].value_counts()
                
                fig = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title="Contract Priority Distribution",
                    color_discrete_map={
                        'High Priority': '#ff6b6b',
                        'Medium Priority': '#ffd93d',
                        'Low Priority': '#6bcf7f'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Contract Opportunities Details")
                
                st.dataframe(
                    opportunities_df.style.format({
                        'Annual Spend': '{:,.0f}',
                        'Monthly Frequency': '{:.1f}',
                        'Demand Predictability': '{:.2f}',
                        'Suitability Score': '{:.2f}',
                        'Avg Unit Price': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No contract opportunities found with the current criteria.")

def show_lot_size_optimization(df):
    """LOT Size Optimization Module"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management.")
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
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
    
    # Calculate EOQ for all items
    optimization_results = []
    
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        if len(item_data) >= 3:  # Need minimum data points
            annual_demand = item_data['Qty Delivered'].sum()
            avg_unit_cost = item_data['Unit Price'].mean()
            current_avg_order = item_data['Qty Delivered'].mean()
            
            holding_cost = avg_unit_cost * holding_cost_rate
            
            if annual_demand > 0 and holding_cost > 0:
                eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                def total_cost(order_qty):
                    if order_qty <= 0:
                        return float('inf')
                    ordering_cost_total = (annual_demand / order_qty) * ordering_cost
                    holding_cost_total = (order_qty / 2) * holding_cost
                    return ordering_cost_total + holding_cost_total
                
                eoq_cost = total_cost(eoq)
                current_cost = total_cost(current_avg_order)
                potential_savings = current_cost - eoq_cost
                
                optimization_results.append({
                    'Item': item,
                    'Annual Demand': annual_demand,
                    'Current Avg Order': current_avg_order,
                    'Optimal EOQ': eoq,
                    'Current Cost': current_cost,
                    'EOQ Cost': eoq_cost,
                    'Potential Savings': potential_savings,
                    'Savings %': (potential_savings / current_cost * 100) if current_cost > 0 else 0
                })
    
    if optimization_results:
        results_df = pd.DataFrame(optimization_results)
        results_df = results_df.sort_values('Potential Savings', ascending=False)
        
        # Summary metrics
        total_savings = results_df['Potential Savings'].sum()
        avg_savings_pct = results_df['Savings %'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Potential Savings", f"${total_savings:,.0f}")
        with col2:
            st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
        with col3:
            st.metric("Items Analyzed", len(results_df))
        
        # Top opportunities
        st.subheader("🎯 Top Optimization Opportunities")
        
        display_df = results_df.head(15)[['Item', 'Current Avg Order', 'Optimal EOQ', 'Potential Savings', 'Savings %']]
        
        st.dataframe(
            display_df.style.format({
                'Current Avg Order': '{:.0f}',
                'Optimal EOQ': '{:.0f}',
                'Potential Savings': '${:,.0f}',
                'Savings %': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        # Visualization
        fig = px.bar(results_df.head(10), 
                    x='Potential Savings', 
                    y='Item',
                    orientation='h',
                    title="Top 10 Items by Savings Potential")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need more data to perform EOQ optimization.")

def show_seasonal_price_optimization(df):
    """Seasonal Price Optimization Module"""
    st.header("🌟 Seasonal Price Optimization")
    st.markdown("Optimize purchase timing based on seasonal price patterns for maximum cost savings.")
    
    # Basic data validation
    required_columns = ['Creation Date', 'Unit Price', 'Item']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean and prepare data
    df_clean = df.copy()
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date', 'Unit Price', 'Item'])
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Add date components
    df_clean['Year'] = df_clean['Creation Date'].dt.year
    df_clean['Month'] = df_clean['Creation Date'].dt.month
    df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name()
    
    # Item selection
    items = sorted(df_clean['Item'].unique())
    selected_item = st.selectbox("Select Item for Analysis", items)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item]
        
        if len(item_data) < 5:
            st.warning(f"Not enough data points for {selected_item} (need at least 5)")
        else:
            # Monthly price trends
            monthly_prices = item_data.groupby('Month_Name')['Unit Price'].agg(['mean', 'std', 'count']).reset_index()
            
            # Order months correctly
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_prices['Month_Order'] = monthly_prices['Month_Name'].apply(
                lambda x: month_order.index(x) if x in month_order else 12
            )
            monthly_prices = monthly_prices.sort_values('Month_Order')
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_price = item_data['Unit Price'].mean()
                st.metric("Average Price", f"${avg_price:.2f}")
            with col2:
                min_month = monthly_prices.loc[monthly_prices['mean'].idxmin(), 'Month_Name']
                min_price = monthly_prices['mean'].min()
                st.metric("Cheapest Month", min_month, f"${min_price:.2f}")
            with col3:
                max_month = monthly_prices.loc[monthly_prices['mean'].idxmax(), 'Month_Name']
                max_price = monthly_prices['mean'].max()
                st.metric("Most Expensive Month", max_month, f"${max_price:.2f}")
            
            # Price trend chart
            fig = px.line(monthly_prices, x='Month_Name', y='mean',
                         title=f"Monthly Price Trends - Item {selected_item}",
                         labels={'mean': 'Average Unit Price ($)', 'Month_Name': 'Month'})
            fig.update_traces(line=dict(width=3))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(df):
    """Spend Categorization & Anomaly Detection Module"""
    st.header("📊 Spend Categorization & Anomaly Detection")
    st.markdown("AI-powered spend categorization and anomaly detection for complete spend visibility.")
    
    # Basic data validation
    required_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    # Calculate line total if missing
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Anomaly Detection
    contamination_rate = st.slider("Anomaly Detection Sensitivity (%)", 1, 10, 5) / 100
    
    if st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies using AI..."):
            
            # Prepare features for anomaly detection
            features = ['Unit Price', 'Qty Delivered', 'Line Total']
            feature_data = df_clean[features].fillna(df_clean[features].median())
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_features)
            
            # Add anomaly labels
            df_clean['Is_Anomaly'] = anomaly_labels == -1
            
            # Anomaly summary
            total_anomalies = df_clean['Is_Anomaly'].sum()
            anomaly_spend = df_clean[df_clean['Is_Anomaly']]['Line Total'].sum()
            total_spend = df_clean['Line Total'].sum()
            anomaly_percent = (anomaly_spend / total_spend * 100) if total_spend > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Detected", total_anomalies)
            with col2:
                st.metric("Anomaly Spend", f"${anomaly_spend:,.0f}")
            with col3:
                st.metric("Anomaly %", f"{anomaly_percent:.1f}%")
            
            if total_anomalies > 0:
                # Anomaly details
                st.subheader("🚨 Detected Anomalies")
                
                anomaly_data = df_clean[df_clean['Is_Anomaly']].sort_values('Line Total', ascending=False)
                display_cols = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Line Total']
                
                # Show available columns only
                available_cols = [col for col in display_cols if col in anomaly_data.columns]
                
                st.dataframe(
                    anomaly_data[available_cols].head(20).style.format({
                        'Unit Price': '${:.2f}',
                        'Qty Delivered': '{:.1f}',
                        'Line Total': '${:,.0f}'
                    }),
                    use_container_width=True
                )
                
                # Anomaly visualization
                fig = px.scatter(df_clean, 
                                x='Unit Price', 
                                y='Qty Delivered',
                                color='Is_Anomaly',
                                size='Line Total',
                                title="Anomaly Detection: Price vs Quantity",
                                labels={'Is_Anomaly': 'Anomaly'},
                                color_discrete_map={True: 'red', False: 'blue'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No significant anomalies detected with current sensitivity settings.")

def show_cross_region_optimization(df):
    """Cross-Region Vendor Optimization"""
    st.header("🌍 Cross-Region Vendor Optimization")
    
    if 'Item' not in df.columns or 'W/H' not in df.columns:
        st.error("This module requires 'Item' and 'W/H' (Warehouse) columns")
        return
    
    # Select item to analyze across regions/vendors
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    filtered = df[df["Item"] == item]
    
    if len(filtered) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by vendor and warehouse to compare pricing
    result = (
        filtered.groupby(["Vendor Name", "W/H"])["Unit Price"]
        .mean()
        .reset_index()
        .sort_values(by="Unit Price")
    )
    
    # Show results
    st.write("Average Unit Price by Vendor and Warehouse:")
    st.dataframe(result, use_container_width=True)
    
    # Visualization
    if len(result) > 0:
        fig = px.bar(result, x='Unit Price', y='Vendor Name', color='W/H',
                    title=f"Price Comparison for Item {item}",
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)

def show_reorder_prediction(df):
    """Smart Reorder Point Prediction"""
    st.header("📈 Smart Reorder Point Prediction")
    
    if 'Item' not in df.columns or 'Creation Date' not in df.columns or 'Qty Delivered' not in df.columns:
        st.error("This module requires 'Item', 'Creation Date', and 'Qty Delivered' columns")
        return
    
    # User selects an item to analyze
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    
    # Filter the dataset for that item
    item_df = df[df["Item"] == item].copy()
    
    if len(item_df) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by month and sum quantity delivered
    item_df["Month"] = pd.to_datetime(item_df["Creation Date"]).dt.to_period("M")
    demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)
    
    # Calculate reorder point using basic statistical method
    if len(demand_by_month) > 0:
        reorder_point = demand_by_month.mean() + demand_by_month.std()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Monthly Demand", f"{demand_by_month.mean():.1f}")
        with col2:
            st.metric("Demand Variability", f"{demand_by_month.std():.1f}")
        with col3:
            st.metric("Suggested Reorder Point", f"{reorder_point:.1f}")
        
        # Chart
        fig = px.line(x=demand_by_month.index.astype(str), y=demand_by_month.values,
                     title=f"Monthly Demand Pattern - Item {item}",
                     labels={'x': 'Month', 'y': 'Quantity Delivered'})
        fig.add_hline(y=reorder_point, line_dash="dash", line_color="red",
                     annotation_text=f"Reorder Point: {reorder_point:.1f}")
        st.plotly_chart(fig, use_container_width=True)

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
    st.markdown('<div class="main-header">📊 Procurement Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("**Advanced procurement analytics with AI-powered insights and optimization tools**")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Platform Controls")
    
    # Currency conversion settings
    st.sidebar.subheader("💱 Currency Conversion")
    enable_currency = st.sidebar.checkbox("Enable Multi-Currency Conversion", value=True, 
                                         help="Convert all amounts to Saudi Riyals (SAR)")
    
    if enable_currency:
        show_conversion_details = st.sidebar.checkbox("Show Conversion Details", value=False)
    
    # File upload
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your procurement data (CSV)",
        type=['csv'],
        help="Upload a CSV file with procurement/PO data"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("❌ No data available. Please upload a CSV file or check your data files.")
        return
    
    # Apply filters
    st.sidebar.subheader("🔧 Filters")
    filtered_df = apply_filters(df)
    
    # Navigation
    st.sidebar.subheader("📑 Analytics Modules")
    page = st.sidebar.selectbox(
        "Choose a module:",
        [
            "📊 Overview Dashboard",
            "🤝 Contracting Opportunities", 
            "📦 LOT Size Optimization",
            "🌟 Seasonal Price Optimization",
            "🚨 Anomaly Detection",
            "🌍 Cross-Region Optimization",
            "📈 Reorder Prediction",
            "🔍 Data Explorer"
        ]
    )
    
    # Data info sidebar
    st.sidebar.subheader("ℹ️ Data Information")
    if 'data_quality' in st.session_state:
        quality_info = st.session_state['data_quality']
        st.sidebar.info(f"**Source:** {st.session_state.get('data_source', 'Unknown')}")
        st.sidebar.info(f"**Columns:** {len(quality_info['columns_available'])}")
        
        # Show currency conversion info if available
        if 'currency_conversions' in st.session_state:
            conversions = st.session_state['currency_conversions']
            st.sidebar.success(f"💱 **Currency Conversions:** {conversions} values converted to SAR")
        
        # Show filtered data info
        st.sidebar.info(f"**Filtered Rows:** {len(filtered_df):,} / {len(df):,}")
        
        # Show exchange rate info if available
        if enable_currency and show_conversion_details:
            st.sidebar.subheader("💱 Exchange Rates Used")
            converter = CurrencyConverter()
            converter.get_exchange_rates()
            
            # Show a few key rates
            key_currencies = ['USD', 'EUR', 'GBP', 'AED', 'QAR']
            for curr in key_currencies:
                if curr in converter.exchange_rates:
                    rate = converter.exchange_rates[curr]
                    st.sidebar.text(f"{curr}: {rate:.3f} SAR")
    
    else:
        st.sidebar.info("**Source:** No data loaded")
    # Display selected page
    if page == "📊 Overview Dashboard":
        show_overview_dashboard(filtered_df)
    elif page == "🤝 Contracting Opportunities":
        show_contracting_opportunities(filtered_df)
    elif page == "📦 LOT Size Optimization":
        show_lot_size_optimization(filtered_df)
    elif page == "🌟 Seasonal Price Optimization":
        show_seasonal_price_optimization(filtered_df)
    elif page == "🚨 Anomaly Detection":
        show_anomaly_detection(filtered_df)
    elif page == "🌍 Cross-Region Optimization":
        show_cross_region_optimization(filtered_df)
    elif page == "📈 Reorder Prediction":
        show_reorder_prediction(filtered_df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("*🚀 Advanced Procurement Analytics Platform - Built with Streamlit & AI*")

if __name__ == "__main__":
    main(), 'USD', 'US
        
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
        st.sidebar.info("💡 Generating sample data for demonstration")
        df = generate_sample_data()
        st.session_state['data_source'] = 'Generated Sample Data (Error Recovery)'
        return df

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

# ADVANCED MODULES INTEGRATION

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

def show_contracting_opportunities(df):
    """Contracting Opportunities Module"""
    st.header("🤝 Contracting Opportunities")
    st.markdown("Identify optimal contracting opportunities based on spend analysis, vendor performance, and demand predictability.")
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Configuration parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_spend = st.number_input("Min Annual Spend Threshold", min_value=0, value=50000, step=10000)
    with col2:
        min_frequency = st.number_input("Min Annual Order Frequency", min_value=1, value=6, step=1)
    with col3:
        analysis_period = st.selectbox("Analysis Period", ["All Data", "Last 12 Months", "Last 6 Months"])
    
    # Filter data by period
    if analysis_period == "Last 12 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=365)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    elif analysis_period == "Last 6 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=180)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    else:
        analysis_df = df_clean
    
    if st.button("🔍 Identify Contract Opportunities", type="primary"):
        with st.spinner("Analyzing contract opportunities..."):
            contract_opportunities = []
            
            # Analyze vendor-item combinations
            vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
            
            for (vendor, item), group_data in vendor_item_combinations:
                # Calculate metrics
                total_spend = (group_data['Unit Price'] * group_data['Qty Delivered']).sum()
                order_frequency = len(group_data)
                
                # Calculate time span
                date_range = group_data['Creation Date'].max() - group_data['Creation Date'].min()
                months_span = date_range.days / 30 if date_range.days > 0 else 1
                monthly_frequency = order_frequency / months_span
                
                # Demand predictability
                monthly_demand = group_data.groupby(group_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
                demand_cv = monthly_demand.std() / monthly_demand.mean() if len(monthly_demand) > 1 and monthly_demand.mean() > 0 else 1
                demand_predictability = max(0, 1 - demand_cv)
                
                # Contract suitability score
                spend_score = min(total_spend / min_spend, 1.0) if min_spend > 0 else 1.0
                frequency_score = min(monthly_frequency / (min_frequency / 12), 1.0)
                
                suitability_score = (spend_score * 0.4 + frequency_score * 0.3 + demand_predictability * 0.3)
                
                # Contract recommendation
                if suitability_score >= 0.7 and total_spend >= min_spend:
                    recommendation = "High Priority"
                elif suitability_score >= 0.5 and total_spend >= min_spend * 0.5:
                    recommendation = "Medium Priority"
                elif suitability_score >= 0.3:
                    recommendation = "Low Priority"
                else:
                    recommendation = "Not Suitable"
                
                if recommendation != "Not Suitable":
                    item_desc = group_data['Item Description'].iloc[0] if 'Item Description' in group_data.columns else f"Item {item}"
                    
                    contract_opportunities.append({
                        'Vendor Name': vendor,
                        'Item': item,
                        'Item Description': item_desc[:50] + "..." if len(item_desc) > 50 else item_desc,
                        'Annual Spend': total_spend,
                        'Order Frequency': order_frequency,
                        'Monthly Frequency': monthly_frequency,
                        'Demand Predictability': demand_predictability,
                        'Suitability Score': suitability_score,
                        'Contract Priority': recommendation,
                        'Avg Unit Price': group_data['Unit Price'].mean(),
                    })
            
            if contract_opportunities:
                opportunities_df = pd.DataFrame(contract_opportunities)
                opportunities_df = opportunities_df.sort_values(['Suitability Score', 'Annual Spend'], ascending=[False, False])
                
                # Summary metrics
                total_contract_spend = opportunities_df['Annual Spend'].sum()
                high_priority_count = len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])
                avg_suitability = opportunities_df['Suitability Score'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Opportunities", len(opportunities_df))
                with col2:
                    st.metric("High Priority Items", high_priority_count)
                with col3:
                    st.metric("Total Contract Spend", f"${total_contract_spend:,.0f}")
                with col4:
                    st.metric("Avg Suitability Score", f"{avg_suitability:.2f}")
                
                # Priority distribution
                priority_counts = opportunities_df['Contract Priority'].value_counts()
                
                fig = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title="Contract Priority Distribution",
                    color_discrete_map={
                        'High Priority': '#ff6b6b',
                        'Medium Priority': '#ffd93d',
                        'Low Priority': '#6bcf7f'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Contract Opportunities Details")
                
                st.dataframe(
                    opportunities_df.style.format({
                        'Annual Spend': '{:,.0f}',
                        'Monthly Frequency': '{:.1f}',
                        'Demand Predictability': '{:.2f}',
                        'Suitability Score': '{:.2f}',
                        'Avg Unit Price': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No contract opportunities found with the current criteria.")

def show_lot_size_optimization(df):
    """LOT Size Optimization Module"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management.")
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
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
    
    # Calculate EOQ for all items
    optimization_results = []
    
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        if len(item_data) >= 3:  # Need minimum data points
            annual_demand = item_data['Qty Delivered'].sum()
            avg_unit_cost = item_data['Unit Price'].mean()
            current_avg_order = item_data['Qty Delivered'].mean()
            
            holding_cost = avg_unit_cost * holding_cost_rate
            
            if annual_demand > 0 and holding_cost > 0:
                eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                def total_cost(order_qty):
                    if order_qty <= 0:
                        return float('inf')
                    ordering_cost_total = (annual_demand / order_qty) * ordering_cost
                    holding_cost_total = (order_qty / 2) * holding_cost
                    return ordering_cost_total + holding_cost_total
                
                eoq_cost = total_cost(eoq)
                current_cost = total_cost(current_avg_order)
                potential_savings = current_cost - eoq_cost
                
                optimization_results.append({
                    'Item': item,
                    'Annual Demand': annual_demand,
                    'Current Avg Order': current_avg_order,
                    'Optimal EOQ': eoq,
                    'Current Cost': current_cost,
                    'EOQ Cost': eoq_cost,
                    'Potential Savings': potential_savings,
                    'Savings %': (potential_savings / current_cost * 100) if current_cost > 0 else 0
                })
    
    if optimization_results:
        results_df = pd.DataFrame(optimization_results)
        results_df = results_df.sort_values('Potential Savings', ascending=False)
        
        # Summary metrics
        total_savings = results_df['Potential Savings'].sum()
        avg_savings_pct = results_df['Savings %'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Potential Savings", f"${total_savings:,.0f}")
        with col2:
            st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
        with col3:
            st.metric("Items Analyzed", len(results_df))
        
        # Top opportunities
        st.subheader("🎯 Top Optimization Opportunities")
        
        display_df = results_df.head(15)[['Item', 'Current Avg Order', 'Optimal EOQ', 'Potential Savings', 'Savings %']]
        
        st.dataframe(
            display_df.style.format({
                'Current Avg Order': '{:.0f}',
                'Optimal EOQ': '{:.0f}',
                'Potential Savings': '${:,.0f}',
                'Savings %': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        # Visualization
        fig = px.bar(results_df.head(10), 
                    x='Potential Savings', 
                    y='Item',
                    orientation='h',
                    title="Top 10 Items by Savings Potential")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need more data to perform EOQ optimization.")

def show_seasonal_price_optimization(df):
    """Seasonal Price Optimization Module"""
    st.header("🌟 Seasonal Price Optimization")
    st.markdown("Optimize purchase timing based on seasonal price patterns for maximum cost savings.")
    
    # Basic data validation
    required_columns = ['Creation Date', 'Unit Price', 'Item']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean and prepare data
    df_clean = df.copy()
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date', 'Unit Price', 'Item'])
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Add date components
    df_clean['Year'] = df_clean['Creation Date'].dt.year
    df_clean['Month'] = df_clean['Creation Date'].dt.month
    df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name()
    
    # Item selection
    items = sorted(df_clean['Item'].unique())
    selected_item = st.selectbox("Select Item for Analysis", items)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item]
        
        if len(item_data) < 5:
            st.warning(f"Not enough data points for {selected_item} (need at least 5)")
        else:
            # Monthly price trends
            monthly_prices = item_data.groupby('Month_Name')['Unit Price'].agg(['mean', 'std', 'count']).reset_index()
            
            # Order months correctly
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_prices['Month_Order'] = monthly_prices['Month_Name'].apply(
                lambda x: month_order.index(x) if x in month_order else 12
            )
            monthly_prices = monthly_prices.sort_values('Month_Order')
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_price = item_data['Unit Price'].mean()
                st.metric("Average Price", f"${avg_price:.2f}")
            with col2:
                min_month = monthly_prices.loc[monthly_prices['mean'].idxmin(), 'Month_Name']
                min_price = monthly_prices['mean'].min()
                st.metric("Cheapest Month", min_month, f"${min_price:.2f}")
            with col3:
                max_month = monthly_prices.loc[monthly_prices['mean'].idxmax(), 'Month_Name']
                max_price = monthly_prices['mean'].max()
                st.metric("Most Expensive Month", max_month, f"${max_price:.2f}")
            
            # Price trend chart
            fig = px.line(monthly_prices, x='Month_Name', y='mean',
                         title=f"Monthly Price Trends - Item {selected_item}",
                         labels={'mean': 'Average Unit Price ($)', 'Month_Name': 'Month'})
            fig.update_traces(line=dict(width=3))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(df):
    """Spend Categorization & Anomaly Detection Module"""
    st.header("📊 Spend Categorization & Anomaly Detection")
    st.markdown("AI-powered spend categorization and anomaly detection for complete spend visibility.")
    
    # Basic data validation
    required_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    # Calculate line total if missing
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Anomaly Detection
    contamination_rate = st.slider("Anomaly Detection Sensitivity (%)", 1, 10, 5) / 100
    
    if st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies using AI..."):
            
            # Prepare features for anomaly detection
            features = ['Unit Price', 'Qty Delivered', 'Line Total']
            feature_data = df_clean[features].fillna(df_clean[features].median())
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_features)
            
            # Add anomaly labels
            df_clean['Is_Anomaly'] = anomaly_labels == -1
            
            # Anomaly summary
            total_anomalies = df_clean['Is_Anomaly'].sum()
            anomaly_spend = df_clean[df_clean['Is_Anomaly']]['Line Total'].sum()
            total_spend = df_clean['Line Total'].sum()
            anomaly_percent = (anomaly_spend / total_spend * 100) if total_spend > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Detected", total_anomalies)
            with col2:
                st.metric("Anomaly Spend", f"${anomaly_spend:,.0f}")
            with col3:
                st.metric("Anomaly %", f"{anomaly_percent:.1f}%")
            
            if total_anomalies > 0:
                # Anomaly details
                st.subheader("🚨 Detected Anomalies")
                
                anomaly_data = df_clean[df_clean['Is_Anomaly']].sort_values('Line Total', ascending=False)
                display_cols = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Line Total']
                
                # Show available columns only
                available_cols = [col for col in display_cols if col in anomaly_data.columns]
                
                st.dataframe(
                    anomaly_data[available_cols].head(20).style.format({
                        'Unit Price': '${:.2f}',
                        'Qty Delivered': '{:.1f}',
                        'Line Total': '${:,.0f}'
                    }),
                    use_container_width=True
                )
                
                # Anomaly visualization
                fig = px.scatter(df_clean, 
                                x='Unit Price', 
                                y='Qty Delivered',
                                color='Is_Anomaly',
                                size='Line Total',
                                title="Anomaly Detection: Price vs Quantity",
                                labels={'Is_Anomaly': 'Anomaly'},
                                color_discrete_map={True: 'red', False: 'blue'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No significant anomalies detected with current sensitivity settings.")

def show_cross_region_optimization(df):
    """Cross-Region Vendor Optimization"""
    st.header("🌍 Cross-Region Vendor Optimization")
    
    if 'Item' not in df.columns or 'W/H' not in df.columns:
        st.error("This module requires 'Item' and 'W/H' (Warehouse) columns")
        return
    
    # Select item to analyze across regions/vendors
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    filtered = df[df["Item"] == item]
    
    if len(filtered) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by vendor and warehouse to compare pricing
    result = (
        filtered.groupby(["Vendor Name", "W/H"])["Unit Price"]
        .mean()
        .reset_index()
        .sort_values(by="Unit Price")
    )
    
    # Show results
    st.write("Average Unit Price by Vendor and Warehouse:")
    st.dataframe(result, use_container_width=True)
    
    # Visualization
    if len(result) > 0:
        fig = px.bar(result, x='Unit Price', y='Vendor Name', color='W/H',
                    title=f"Price Comparison for Item {item}",
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)

def show_reorder_prediction(df):
    """Smart Reorder Point Prediction"""
    st.header("📈 Smart Reorder Point Prediction")
    
    if 'Item' not in df.columns or 'Creation Date' not in df.columns or 'Qty Delivered' not in df.columns:
        st.error("This module requires 'Item', 'Creation Date', and 'Qty Delivered' columns")
        return
    
    # User selects an item to analyze
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    
    # Filter the dataset for that item
    item_df = df[df["Item"] == item].copy()
    
    if len(item_df) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by month and sum quantity delivered
    item_df["Month"] = pd.to_datetime(item_df["Creation Date"]).dt.to_period("M")
    demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)
    
    # Calculate reorder point using basic statistical method
    if len(demand_by_month) > 0:
        reorder_point = demand_by_month.mean() + demand_by_month.std()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Monthly Demand", f"{demand_by_month.mean():.1f}")
        with col2:
            st.metric("Demand Variability", f"{demand_by_month.std():.1f}")
        with col3:
            st.metric("Suggested Reorder Point", f"{reorder_point:.1f}")
        
        # Chart
        fig = px.line(x=demand_by_month.index.astype(str), y=demand_by_month.values,
                     title=f"Monthly Demand Pattern - Item {item}",
                     labels={'x': 'Month', 'y': 'Quantity Delivered'})
        fig.add_hline(y=reorder_point, line_dash="dash", line_color="red",
                     annotation_text=f"Reorder Point: {reorder_point:.1f}")
        st.plotly_chart(fig, use_container_width=True)

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
    st.markdown('<div class="main-header">📊 Procurement Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("**Advanced procurement analytics with AI-powered insights and optimization tools**")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Platform Controls")
    
    # File upload
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your procurement data (CSV)",
        type=['csv'],
        help="Upload a CSV file with procurement/PO data"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("❌ No data available. Please upload a CSV file or check your data files.")
        return
    
    # Apply filters
    st.sidebar.subheader("🔧 Filters")
    filtered_df = apply_filters(df)
    
    # Navigation
    st.sidebar.subheader("📑 Analytics Modules")
    page = st.sidebar.selectbox(
        "Choose a module:",
        [
            "📊 Overview Dashboard",
            "🤝 Contracting Opportunities", 
            "📦 LOT Size Optimization",
            "🌟 Seasonal Price Optimization",
            "🚨 Anomaly Detection",
            "🌍 Cross-Region Optimization",
            "📈 Reorder Prediction",
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
    elif page == "🤝 Contracting Opportunities":
        show_contracting_opportunities(filtered_df)
    elif page == "📦 LOT Size Optimization":
        show_lot_size_optimization(filtered_df)
    elif page == "🌟 Seasonal Price Optimization":
        show_seasonal_price_optimization(filtered_df)
    elif page == "🚨 Anomaly Detection":
        show_anomaly_detection(filtered_df)
    elif page == "🌍 Cross-Region Optimization":
        show_cross_region_optimization(filtered_df)
    elif page == "📈 Reorder Prediction":
        show_reorder_prediction(filtered_df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("*🚀 Advanced Procurement Analytics Platform - Built with Streamlit & AI*")

if __name__ == "__main__":
    main(), 'DOLLAR', 'DOLLARS'],
            'EUR': ['€', 'EUR', 'EURO', 'EUROS'],
            'GBP': ['£', 'GBP', 'POUND', 'POUNDS', 'STERLING'],
            'SAR': ['SAR', 'SR', 'RIYAL', 'RIYALS', 'ريال'],
            'AED': ['AED', 'DIRHAM', 'DIRHAMS', 'درهم'],
            'QAR': ['QAR', 'QATAR', 'RIYAL'],
            'KWD': ['KWD', 'DINAR', 'DINARS'],
            'BHD': ['BHD', 'BAHRAIN'],
            'OMR': ['OMR', 'RIAL', 'RIALS'],
            'JPY': ['¥', 'JPY', 'YEN'],
            'CNY': ['¥', 'CNY', 'YUAN', 'RMB'],
            'INR': ['₹', 'INR', 'RUPEE', 'RUPEES'],
            'PKR': ['PKR', 'PAKISTAN'],
            'THB': ['THB', 'BAHT'],
            'MYR': ['MYR', 'RINGGIT'],
            'SGD': ['SGD', 'SINGAPORE'],
            'CAD': ['CAD', 'C
        
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
        st.sidebar.info("💡 Generating sample data for demonstration")
        df = generate_sample_data()
        st.session_state['data_source'] = 'Generated Sample Data (Error Recovery)'
        return df

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

# ADVANCED MODULES INTEGRATION

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

def show_contracting_opportunities(df):
    """Contracting Opportunities Module"""
    st.header("🤝 Contracting Opportunities")
    st.markdown("Identify optimal contracting opportunities based on spend analysis, vendor performance, and demand predictability.")
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Configuration parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_spend = st.number_input("Min Annual Spend Threshold", min_value=0, value=50000, step=10000)
    with col2:
        min_frequency = st.number_input("Min Annual Order Frequency", min_value=1, value=6, step=1)
    with col3:
        analysis_period = st.selectbox("Analysis Period", ["All Data", "Last 12 Months", "Last 6 Months"])
    
    # Filter data by period
    if analysis_period == "Last 12 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=365)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    elif analysis_period == "Last 6 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=180)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    else:
        analysis_df = df_clean
    
    if st.button("🔍 Identify Contract Opportunities", type="primary"):
        with st.spinner("Analyzing contract opportunities..."):
            contract_opportunities = []
            
            # Analyze vendor-item combinations
            vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
            
            for (vendor, item), group_data in vendor_item_combinations:
                # Calculate metrics
                total_spend = (group_data['Unit Price'] * group_data['Qty Delivered']).sum()
                order_frequency = len(group_data)
                
                # Calculate time span
                date_range = group_data['Creation Date'].max() - group_data['Creation Date'].min()
                months_span = date_range.days / 30 if date_range.days > 0 else 1
                monthly_frequency = order_frequency / months_span
                
                # Demand predictability
                monthly_demand = group_data.groupby(group_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
                demand_cv = monthly_demand.std() / monthly_demand.mean() if len(monthly_demand) > 1 and monthly_demand.mean() > 0 else 1
                demand_predictability = max(0, 1 - demand_cv)
                
                # Contract suitability score
                spend_score = min(total_spend / min_spend, 1.0) if min_spend > 0 else 1.0
                frequency_score = min(monthly_frequency / (min_frequency / 12), 1.0)
                
                suitability_score = (spend_score * 0.4 + frequency_score * 0.3 + demand_predictability * 0.3)
                
                # Contract recommendation
                if suitability_score >= 0.7 and total_spend >= min_spend:
                    recommendation = "High Priority"
                elif suitability_score >= 0.5 and total_spend >= min_spend * 0.5:
                    recommendation = "Medium Priority"
                elif suitability_score >= 0.3:
                    recommendation = "Low Priority"
                else:
                    recommendation = "Not Suitable"
                
                if recommendation != "Not Suitable":
                    item_desc = group_data['Item Description'].iloc[0] if 'Item Description' in group_data.columns else f"Item {item}"
                    
                    contract_opportunities.append({
                        'Vendor Name': vendor,
                        'Item': item,
                        'Item Description': item_desc[:50] + "..." if len(item_desc) > 50 else item_desc,
                        'Annual Spend': total_spend,
                        'Order Frequency': order_frequency,
                        'Monthly Frequency': monthly_frequency,
                        'Demand Predictability': demand_predictability,
                        'Suitability Score': suitability_score,
                        'Contract Priority': recommendation,
                        'Avg Unit Price': group_data['Unit Price'].mean(),
                    })
            
            if contract_opportunities:
                opportunities_df = pd.DataFrame(contract_opportunities)
                opportunities_df = opportunities_df.sort_values(['Suitability Score', 'Annual Spend'], ascending=[False, False])
                
                # Summary metrics
                total_contract_spend = opportunities_df['Annual Spend'].sum()
                high_priority_count = len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])
                avg_suitability = opportunities_df['Suitability Score'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Opportunities", len(opportunities_df))
                with col2:
                    st.metric("High Priority Items", high_priority_count)
                with col3:
                    st.metric("Total Contract Spend", f"${total_contract_spend:,.0f}")
                with col4:
                    st.metric("Avg Suitability Score", f"{avg_suitability:.2f}")
                
                # Priority distribution
                priority_counts = opportunities_df['Contract Priority'].value_counts()
                
                fig = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title="Contract Priority Distribution",
                    color_discrete_map={
                        'High Priority': '#ff6b6b',
                        'Medium Priority': '#ffd93d',
                        'Low Priority': '#6bcf7f'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Contract Opportunities Details")
                
                st.dataframe(
                    opportunities_df.style.format({
                        'Annual Spend': '{:,.0f}',
                        'Monthly Frequency': '{:.1f}',
                        'Demand Predictability': '{:.2f}',
                        'Suitability Score': '{:.2f}',
                        'Avg Unit Price': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No contract opportunities found with the current criteria.")

def show_lot_size_optimization(df):
    """LOT Size Optimization Module"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management.")
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
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
    
    # Calculate EOQ for all items
    optimization_results = []
    
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        if len(item_data) >= 3:  # Need minimum data points
            annual_demand = item_data['Qty Delivered'].sum()
            avg_unit_cost = item_data['Unit Price'].mean()
            current_avg_order = item_data['Qty Delivered'].mean()
            
            holding_cost = avg_unit_cost * holding_cost_rate
            
            if annual_demand > 0 and holding_cost > 0:
                eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                def total_cost(order_qty):
                    if order_qty <= 0:
                        return float('inf')
                    ordering_cost_total = (annual_demand / order_qty) * ordering_cost
                    holding_cost_total = (order_qty / 2) * holding_cost
                    return ordering_cost_total + holding_cost_total
                
                eoq_cost = total_cost(eoq)
                current_cost = total_cost(current_avg_order)
                potential_savings = current_cost - eoq_cost
                
                optimization_results.append({
                    'Item': item,
                    'Annual Demand': annual_demand,
                    'Current Avg Order': current_avg_order,
                    'Optimal EOQ': eoq,
                    'Current Cost': current_cost,
                    'EOQ Cost': eoq_cost,
                    'Potential Savings': potential_savings,
                    'Savings %': (potential_savings / current_cost * 100) if current_cost > 0 else 0
                })
    
    if optimization_results:
        results_df = pd.DataFrame(optimization_results)
        results_df = results_df.sort_values('Potential Savings', ascending=False)
        
        # Summary metrics
        total_savings = results_df['Potential Savings'].sum()
        avg_savings_pct = results_df['Savings %'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Potential Savings", f"${total_savings:,.0f}")
        with col2:
            st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
        with col3:
            st.metric("Items Analyzed", len(results_df))
        
        # Top opportunities
        st.subheader("🎯 Top Optimization Opportunities")
        
        display_df = results_df.head(15)[['Item', 'Current Avg Order', 'Optimal EOQ', 'Potential Savings', 'Savings %']]
        
        st.dataframe(
            display_df.style.format({
                'Current Avg Order': '{:.0f}',
                'Optimal EOQ': '{:.0f}',
                'Potential Savings': '${:,.0f}',
                'Savings %': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        # Visualization
        fig = px.bar(results_df.head(10), 
                    x='Potential Savings', 
                    y='Item',
                    orientation='h',
                    title="Top 10 Items by Savings Potential")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need more data to perform EOQ optimization.")

def show_seasonal_price_optimization(df):
    """Seasonal Price Optimization Module"""
    st.header("🌟 Seasonal Price Optimization")
    st.markdown("Optimize purchase timing based on seasonal price patterns for maximum cost savings.")
    
    # Basic data validation
    required_columns = ['Creation Date', 'Unit Price', 'Item']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean and prepare data
    df_clean = df.copy()
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date', 'Unit Price', 'Item'])
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Add date components
    df_clean['Year'] = df_clean['Creation Date'].dt.year
    df_clean['Month'] = df_clean['Creation Date'].dt.month
    df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name()
    
    # Item selection
    items = sorted(df_clean['Item'].unique())
    selected_item = st.selectbox("Select Item for Analysis", items)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item]
        
        if len(item_data) < 5:
            st.warning(f"Not enough data points for {selected_item} (need at least 5)")
        else:
            # Monthly price trends
            monthly_prices = item_data.groupby('Month_Name')['Unit Price'].agg(['mean', 'std', 'count']).reset_index()
            
            # Order months correctly
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_prices['Month_Order'] = monthly_prices['Month_Name'].apply(
                lambda x: month_order.index(x) if x in month_order else 12
            )
            monthly_prices = monthly_prices.sort_values('Month_Order')
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_price = item_data['Unit Price'].mean()
                st.metric("Average Price", f"${avg_price:.2f}")
            with col2:
                min_month = monthly_prices.loc[monthly_prices['mean'].idxmin(), 'Month_Name']
                min_price = monthly_prices['mean'].min()
                st.metric("Cheapest Month", min_month, f"${min_price:.2f}")
            with col3:
                max_month = monthly_prices.loc[monthly_prices['mean'].idxmax(), 'Month_Name']
                max_price = monthly_prices['mean'].max()
                st.metric("Most Expensive Month", max_month, f"${max_price:.2f}")
            
            # Price trend chart
            fig = px.line(monthly_prices, x='Month_Name', y='mean',
                         title=f"Monthly Price Trends - Item {selected_item}",
                         labels={'mean': 'Average Unit Price ($)', 'Month_Name': 'Month'})
            fig.update_traces(line=dict(width=3))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(df):
    """Spend Categorization & Anomaly Detection Module"""
    st.header("📊 Spend Categorization & Anomaly Detection")
    st.markdown("AI-powered spend categorization and anomaly detection for complete spend visibility.")
    
    # Basic data validation
    required_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    # Calculate line total if missing
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Anomaly Detection
    contamination_rate = st.slider("Anomaly Detection Sensitivity (%)", 1, 10, 5) / 100
    
    if st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies using AI..."):
            
            # Prepare features for anomaly detection
            features = ['Unit Price', 'Qty Delivered', 'Line Total']
            feature_data = df_clean[features].fillna(df_clean[features].median())
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_features)
            
            # Add anomaly labels
            df_clean['Is_Anomaly'] = anomaly_labels == -1
            
            # Anomaly summary
            total_anomalies = df_clean['Is_Anomaly'].sum()
            anomaly_spend = df_clean[df_clean['Is_Anomaly']]['Line Total'].sum()
            total_spend = df_clean['Line Total'].sum()
            anomaly_percent = (anomaly_spend / total_spend * 100) if total_spend > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Detected", total_anomalies)
            with col2:
                st.metric("Anomaly Spend", f"${anomaly_spend:,.0f}")
            with col3:
                st.metric("Anomaly %", f"{anomaly_percent:.1f}%")
            
            if total_anomalies > 0:
                # Anomaly details
                st.subheader("🚨 Detected Anomalies")
                
                anomaly_data = df_clean[df_clean['Is_Anomaly']].sort_values('Line Total', ascending=False)
                display_cols = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Line Total']
                
                # Show available columns only
                available_cols = [col for col in display_cols if col in anomaly_data.columns]
                
                st.dataframe(
                    anomaly_data[available_cols].head(20).style.format({
                        'Unit Price': '${:.2f}',
                        'Qty Delivered': '{:.1f}',
                        'Line Total': '${:,.0f}'
                    }),
                    use_container_width=True
                )
                
                # Anomaly visualization
                fig = px.scatter(df_clean, 
                                x='Unit Price', 
                                y='Qty Delivered',
                                color='Is_Anomaly',
                                size='Line Total',
                                title="Anomaly Detection: Price vs Quantity",
                                labels={'Is_Anomaly': 'Anomaly'},
                                color_discrete_map={True: 'red', False: 'blue'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No significant anomalies detected with current sensitivity settings.")

def show_cross_region_optimization(df):
    """Cross-Region Vendor Optimization"""
    st.header("🌍 Cross-Region Vendor Optimization")
    
    if 'Item' not in df.columns or 'W/H' not in df.columns:
        st.error("This module requires 'Item' and 'W/H' (Warehouse) columns")
        return
    
    # Select item to analyze across regions/vendors
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    filtered = df[df["Item"] == item]
    
    if len(filtered) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by vendor and warehouse to compare pricing
    result = (
        filtered.groupby(["Vendor Name", "W/H"])["Unit Price"]
        .mean()
        .reset_index()
        .sort_values(by="Unit Price")
    )
    
    # Show results
    st.write("Average Unit Price by Vendor and Warehouse:")
    st.dataframe(result, use_container_width=True)
    
    # Visualization
    if len(result) > 0:
        fig = px.bar(result, x='Unit Price', y='Vendor Name', color='W/H',
                    title=f"Price Comparison for Item {item}",
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)

def show_reorder_prediction(df):
    """Smart Reorder Point Prediction"""
    st.header("📈 Smart Reorder Point Prediction")
    
    if 'Item' not in df.columns or 'Creation Date' not in df.columns or 'Qty Delivered' not in df.columns:
        st.error("This module requires 'Item', 'Creation Date', and 'Qty Delivered' columns")
        return
    
    # User selects an item to analyze
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    
    # Filter the dataset for that item
    item_df = df[df["Item"] == item].copy()
    
    if len(item_df) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by month and sum quantity delivered
    item_df["Month"] = pd.to_datetime(item_df["Creation Date"]).dt.to_period("M")
    demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)
    
    # Calculate reorder point using basic statistical method
    if len(demand_by_month) > 0:
        reorder_point = demand_by_month.mean() + demand_by_month.std()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Monthly Demand", f"{demand_by_month.mean():.1f}")
        with col2:
            st.metric("Demand Variability", f"{demand_by_month.std():.1f}")
        with col3:
            st.metric("Suggested Reorder Point", f"{reorder_point:.1f}")
        
        # Chart
        fig = px.line(x=demand_by_month.index.astype(str), y=demand_by_month.values,
                     title=f"Monthly Demand Pattern - Item {item}",
                     labels={'x': 'Month', 'y': 'Quantity Delivered'})
        fig.add_hline(y=reorder_point, line_dash="dash", line_color="red",
                     annotation_text=f"Reorder Point: {reorder_point:.1f}")
        st.plotly_chart(fig, use_container_width=True)

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
    st.markdown('<div class="main-header">📊 Procurement Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("**Advanced procurement analytics with AI-powered insights and optimization tools**")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Platform Controls")
    
    # File upload
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your procurement data (CSV)",
        type=['csv'],
        help="Upload a CSV file with procurement/PO data"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("❌ No data available. Please upload a CSV file or check your data files.")
        return
    
    # Apply filters
    st.sidebar.subheader("🔧 Filters")
    filtered_df = apply_filters(df)
    
    # Navigation
    st.sidebar.subheader("📑 Analytics Modules")
    page = st.sidebar.selectbox(
        "Choose a module:",
        [
            "📊 Overview Dashboard",
            "🤝 Contracting Opportunities", 
            "📦 LOT Size Optimization",
            "🌟 Seasonal Price Optimization",
            "🚨 Anomaly Detection",
            "🌍 Cross-Region Optimization",
            "📈 Reorder Prediction",
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
    elif page == "🤝 Contracting Opportunities":
        show_contracting_opportunities(filtered_df)
    elif page == "📦 LOT Size Optimization":
        show_lot_size_optimization(filtered_df)
    elif page == "🌟 Seasonal Price Optimization":
        show_seasonal_price_optimization(filtered_df)
    elif page == "🚨 Anomaly Detection":
        show_anomaly_detection(filtered_df)
    elif page == "🌍 Cross-Region Optimization":
        show_cross_region_optimization(filtered_df)
    elif page == "📈 Reorder Prediction":
        show_reorder_prediction(filtered_df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("*🚀 Advanced Procurement Analytics Platform - Built with Streamlit & AI*")

if __name__ == "__main__":
    main(), 'CANADIAN'],
            'AUD': ['AUD', 'A
        
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
        st.sidebar.info("💡 Generating sample data for demonstration")
        df = generate_sample_data()
        st.session_state['data_source'] = 'Generated Sample Data (Error Recovery)'
        return df

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

# ADVANCED MODULES INTEGRATION

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

def show_contracting_opportunities(df):
    """Contracting Opportunities Module"""
    st.header("🤝 Contracting Opportunities")
    st.markdown("Identify optimal contracting opportunities based on spend analysis, vendor performance, and demand predictability.")
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Configuration parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_spend = st.number_input("Min Annual Spend Threshold", min_value=0, value=50000, step=10000)
    with col2:
        min_frequency = st.number_input("Min Annual Order Frequency", min_value=1, value=6, step=1)
    with col3:
        analysis_period = st.selectbox("Analysis Period", ["All Data", "Last 12 Months", "Last 6 Months"])
    
    # Filter data by period
    if analysis_period == "Last 12 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=365)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    elif analysis_period == "Last 6 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=180)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    else:
        analysis_df = df_clean
    
    if st.button("🔍 Identify Contract Opportunities", type="primary"):
        with st.spinner("Analyzing contract opportunities..."):
            contract_opportunities = []
            
            # Analyze vendor-item combinations
            vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
            
            for (vendor, item), group_data in vendor_item_combinations:
                # Calculate metrics
                total_spend = (group_data['Unit Price'] * group_data['Qty Delivered']).sum()
                order_frequency = len(group_data)
                
                # Calculate time span
                date_range = group_data['Creation Date'].max() - group_data['Creation Date'].min()
                months_span = date_range.days / 30 if date_range.days > 0 else 1
                monthly_frequency = order_frequency / months_span
                
                # Demand predictability
                monthly_demand = group_data.groupby(group_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
                demand_cv = monthly_demand.std() / monthly_demand.mean() if len(monthly_demand) > 1 and monthly_demand.mean() > 0 else 1
                demand_predictability = max(0, 1 - demand_cv)
                
                # Contract suitability score
                spend_score = min(total_spend / min_spend, 1.0) if min_spend > 0 else 1.0
                frequency_score = min(monthly_frequency / (min_frequency / 12), 1.0)
                
                suitability_score = (spend_score * 0.4 + frequency_score * 0.3 + demand_predictability * 0.3)
                
                # Contract recommendation
                if suitability_score >= 0.7 and total_spend >= min_spend:
                    recommendation = "High Priority"
                elif suitability_score >= 0.5 and total_spend >= min_spend * 0.5:
                    recommendation = "Medium Priority"
                elif suitability_score >= 0.3:
                    recommendation = "Low Priority"
                else:
                    recommendation = "Not Suitable"
                
                if recommendation != "Not Suitable":
                    item_desc = group_data['Item Description'].iloc[0] if 'Item Description' in group_data.columns else f"Item {item}"
                    
                    contract_opportunities.append({
                        'Vendor Name': vendor,
                        'Item': item,
                        'Item Description': item_desc[:50] + "..." if len(item_desc) > 50 else item_desc,
                        'Annual Spend': total_spend,
                        'Order Frequency': order_frequency,
                        'Monthly Frequency': monthly_frequency,
                        'Demand Predictability': demand_predictability,
                        'Suitability Score': suitability_score,
                        'Contract Priority': recommendation,
                        'Avg Unit Price': group_data['Unit Price'].mean(),
                    })
            
            if contract_opportunities:
                opportunities_df = pd.DataFrame(contract_opportunities)
                opportunities_df = opportunities_df.sort_values(['Suitability Score', 'Annual Spend'], ascending=[False, False])
                
                # Summary metrics
                total_contract_spend = opportunities_df['Annual Spend'].sum()
                high_priority_count = len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])
                avg_suitability = opportunities_df['Suitability Score'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Opportunities", len(opportunities_df))
                with col2:
                    st.metric("High Priority Items", high_priority_count)
                with col3:
                    st.metric("Total Contract Spend", f"${total_contract_spend:,.0f}")
                with col4:
                    st.metric("Avg Suitability Score", f"{avg_suitability:.2f}")
                
                # Priority distribution
                priority_counts = opportunities_df['Contract Priority'].value_counts()
                
                fig = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title="Contract Priority Distribution",
                    color_discrete_map={
                        'High Priority': '#ff6b6b',
                        'Medium Priority': '#ffd93d',
                        'Low Priority': '#6bcf7f'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Contract Opportunities Details")
                
                st.dataframe(
                    opportunities_df.style.format({
                        'Annual Spend': '{:,.0f}',
                        'Monthly Frequency': '{:.1f}',
                        'Demand Predictability': '{:.2f}',
                        'Suitability Score': '{:.2f}',
                        'Avg Unit Price': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No contract opportunities found with the current criteria.")

def show_lot_size_optimization(df):
    """LOT Size Optimization Module"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management.")
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
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
    
    # Calculate EOQ for all items
    optimization_results = []
    
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        if len(item_data) >= 3:  # Need minimum data points
            annual_demand = item_data['Qty Delivered'].sum()
            avg_unit_cost = item_data['Unit Price'].mean()
            current_avg_order = item_data['Qty Delivered'].mean()
            
            holding_cost = avg_unit_cost * holding_cost_rate
            
            if annual_demand > 0 and holding_cost > 0:
                eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                def total_cost(order_qty):
                    if order_qty <= 0:
                        return float('inf')
                    ordering_cost_total = (annual_demand / order_qty) * ordering_cost
                    holding_cost_total = (order_qty / 2) * holding_cost
                    return ordering_cost_total + holding_cost_total
                
                eoq_cost = total_cost(eoq)
                current_cost = total_cost(current_avg_order)
                potential_savings = current_cost - eoq_cost
                
                optimization_results.append({
                    'Item': item,
                    'Annual Demand': annual_demand,
                    'Current Avg Order': current_avg_order,
                    'Optimal EOQ': eoq,
                    'Current Cost': current_cost,
                    'EOQ Cost': eoq_cost,
                    'Potential Savings': potential_savings,
                    'Savings %': (potential_savings / current_cost * 100) if current_cost > 0 else 0
                })
    
    if optimization_results:
        results_df = pd.DataFrame(optimization_results)
        results_df = results_df.sort_values('Potential Savings', ascending=False)
        
        # Summary metrics
        total_savings = results_df['Potential Savings'].sum()
        avg_savings_pct = results_df['Savings %'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Potential Savings", f"${total_savings:,.0f}")
        with col2:
            st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
        with col3:
            st.metric("Items Analyzed", len(results_df))
        
        # Top opportunities
        st.subheader("🎯 Top Optimization Opportunities")
        
        display_df = results_df.head(15)[['Item', 'Current Avg Order', 'Optimal EOQ', 'Potential Savings', 'Savings %']]
        
        st.dataframe(
            display_df.style.format({
                'Current Avg Order': '{:.0f}',
                'Optimal EOQ': '{:.0f}',
                'Potential Savings': '${:,.0f}',
                'Savings %': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        # Visualization
        fig = px.bar(results_df.head(10), 
                    x='Potential Savings', 
                    y='Item',
                    orientation='h',
                    title="Top 10 Items by Savings Potential")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need more data to perform EOQ optimization.")

def show_seasonal_price_optimization(df):
    """Seasonal Price Optimization Module"""
    st.header("🌟 Seasonal Price Optimization")
    st.markdown("Optimize purchase timing based on seasonal price patterns for maximum cost savings.")
    
    # Basic data validation
    required_columns = ['Creation Date', 'Unit Price', 'Item']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean and prepare data
    df_clean = df.copy()
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date', 'Unit Price', 'Item'])
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Add date components
    df_clean['Year'] = df_clean['Creation Date'].dt.year
    df_clean['Month'] = df_clean['Creation Date'].dt.month
    df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name()
    
    # Item selection
    items = sorted(df_clean['Item'].unique())
    selected_item = st.selectbox("Select Item for Analysis", items)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item]
        
        if len(item_data) < 5:
            st.warning(f"Not enough data points for {selected_item} (need at least 5)")
        else:
            # Monthly price trends
            monthly_prices = item_data.groupby('Month_Name')['Unit Price'].agg(['mean', 'std', 'count']).reset_index()
            
            # Order months correctly
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_prices['Month_Order'] = monthly_prices['Month_Name'].apply(
                lambda x: month_order.index(x) if x in month_order else 12
            )
            monthly_prices = monthly_prices.sort_values('Month_Order')
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_price = item_data['Unit Price'].mean()
                st.metric("Average Price", f"${avg_price:.2f}")
            with col2:
                min_month = monthly_prices.loc[monthly_prices['mean'].idxmin(), 'Month_Name']
                min_price = monthly_prices['mean'].min()
                st.metric("Cheapest Month", min_month, f"${min_price:.2f}")
            with col3:
                max_month = monthly_prices.loc[monthly_prices['mean'].idxmax(), 'Month_Name']
                max_price = monthly_prices['mean'].max()
                st.metric("Most Expensive Month", max_month, f"${max_price:.2f}")
            
            # Price trend chart
            fig = px.line(monthly_prices, x='Month_Name', y='mean',
                         title=f"Monthly Price Trends - Item {selected_item}",
                         labels={'mean': 'Average Unit Price ($)', 'Month_Name': 'Month'})
            fig.update_traces(line=dict(width=3))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(df):
    """Spend Categorization & Anomaly Detection Module"""
    st.header("📊 Spend Categorization & Anomaly Detection")
    st.markdown("AI-powered spend categorization and anomaly detection for complete spend visibility.")
    
    # Basic data validation
    required_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    # Calculate line total if missing
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Anomaly Detection
    contamination_rate = st.slider("Anomaly Detection Sensitivity (%)", 1, 10, 5) / 100
    
    if st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies using AI..."):
            
            # Prepare features for anomaly detection
            features = ['Unit Price', 'Qty Delivered', 'Line Total']
            feature_data = df_clean[features].fillna(df_clean[features].median())
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_features)
            
            # Add anomaly labels
            df_clean['Is_Anomaly'] = anomaly_labels == -1
            
            # Anomaly summary
            total_anomalies = df_clean['Is_Anomaly'].sum()
            anomaly_spend = df_clean[df_clean['Is_Anomaly']]['Line Total'].sum()
            total_spend = df_clean['Line Total'].sum()
            anomaly_percent = (anomaly_spend / total_spend * 100) if total_spend > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Detected", total_anomalies)
            with col2:
                st.metric("Anomaly Spend", f"${anomaly_spend:,.0f}")
            with col3:
                st.metric("Anomaly %", f"{anomaly_percent:.1f}%")
            
            if total_anomalies > 0:
                # Anomaly details
                st.subheader("🚨 Detected Anomalies")
                
                anomaly_data = df_clean[df_clean['Is_Anomaly']].sort_values('Line Total', ascending=False)
                display_cols = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Line Total']
                
                # Show available columns only
                available_cols = [col for col in display_cols if col in anomaly_data.columns]
                
                st.dataframe(
                    anomaly_data[available_cols].head(20).style.format({
                        'Unit Price': '${:.2f}',
                        'Qty Delivered': '{:.1f}',
                        'Line Total': '${:,.0f}'
                    }),
                    use_container_width=True
                )
                
                # Anomaly visualization
                fig = px.scatter(df_clean, 
                                x='Unit Price', 
                                y='Qty Delivered',
                                color='Is_Anomaly',
                                size='Line Total',
                                title="Anomaly Detection: Price vs Quantity",
                                labels={'Is_Anomaly': 'Anomaly'},
                                color_discrete_map={True: 'red', False: 'blue'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No significant anomalies detected with current sensitivity settings.")

def show_cross_region_optimization(df):
    """Cross-Region Vendor Optimization"""
    st.header("🌍 Cross-Region Vendor Optimization")
    
    if 'Item' not in df.columns or 'W/H' not in df.columns:
        st.error("This module requires 'Item' and 'W/H' (Warehouse) columns")
        return
    
    # Select item to analyze across regions/vendors
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    filtered = df[df["Item"] == item]
    
    if len(filtered) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by vendor and warehouse to compare pricing
    result = (
        filtered.groupby(["Vendor Name", "W/H"])["Unit Price"]
        .mean()
        .reset_index()
        .sort_values(by="Unit Price")
    )
    
    # Show results
    st.write("Average Unit Price by Vendor and Warehouse:")
    st.dataframe(result, use_container_width=True)
    
    # Visualization
    if len(result) > 0:
        fig = px.bar(result, x='Unit Price', y='Vendor Name', color='W/H',
                    title=f"Price Comparison for Item {item}",
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)

def show_reorder_prediction(df):
    """Smart Reorder Point Prediction"""
    st.header("📈 Smart Reorder Point Prediction")
    
    if 'Item' not in df.columns or 'Creation Date' not in df.columns or 'Qty Delivered' not in df.columns:
        st.error("This module requires 'Item', 'Creation Date', and 'Qty Delivered' columns")
        return
    
    # User selects an item to analyze
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    
    # Filter the dataset for that item
    item_df = df[df["Item"] == item].copy()
    
    if len(item_df) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by month and sum quantity delivered
    item_df["Month"] = pd.to_datetime(item_df["Creation Date"]).dt.to_period("M")
    demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)
    
    # Calculate reorder point using basic statistical method
    if len(demand_by_month) > 0:
        reorder_point = demand_by_month.mean() + demand_by_month.std()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Monthly Demand", f"{demand_by_month.mean():.1f}")
        with col2:
            st.metric("Demand Variability", f"{demand_by_month.std():.1f}")
        with col3:
            st.metric("Suggested Reorder Point", f"{reorder_point:.1f}")
        
        # Chart
        fig = px.line(x=demand_by_month.index.astype(str), y=demand_by_month.values,
                     title=f"Monthly Demand Pattern - Item {item}",
                     labels={'x': 'Month', 'y': 'Quantity Delivered'})
        fig.add_hline(y=reorder_point, line_dash="dash", line_color="red",
                     annotation_text=f"Reorder Point: {reorder_point:.1f}")
        st.plotly_chart(fig, use_container_width=True)

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
    st.markdown('<div class="main-header">📊 Procurement Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("**Advanced procurement analytics with AI-powered insights and optimization tools**")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Platform Controls")
    
    # File upload
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your procurement data (CSV)",
        type=['csv'],
        help="Upload a CSV file with procurement/PO data"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("❌ No data available. Please upload a CSV file or check your data files.")
        return
    
    # Apply filters
    st.sidebar.subheader("🔧 Filters")
    filtered_df = apply_filters(df)
    
    # Navigation
    st.sidebar.subheader("📑 Analytics Modules")
    page = st.sidebar.selectbox(
        "Choose a module:",
        [
            "📊 Overview Dashboard",
            "🤝 Contracting Opportunities", 
            "📦 LOT Size Optimization",
            "🌟 Seasonal Price Optimization",
            "🚨 Anomaly Detection",
            "🌍 Cross-Region Optimization",
            "📈 Reorder Prediction",
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
    elif page == "🤝 Contracting Opportunities":
        show_contracting_opportunities(filtered_df)
    elif page == "📦 LOT Size Optimization":
        show_lot_size_optimization(filtered_df)
    elif page == "🌟 Seasonal Price Optimization":
        show_seasonal_price_optimization(filtered_df)
    elif page == "🚨 Anomaly Detection":
        show_anomaly_detection(filtered_df)
    elif page == "🌍 Cross-Region Optimization":
        show_cross_region_optimization(filtered_df)
    elif page == "📈 Reorder Prediction":
        show_reorder_prediction(filtered_df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("*🚀 Advanced Procurement Analytics Platform - Built with Streamlit & AI*")

if __name__ == "__main__":
    main(), 'AUSTRALIAN'],
            'CHF': ['CHF', 'FRANC', 'FRANCS']
        }
        
        # Static exchange rates to SAR (updated 2024)
        self.fallback_rates = {
            'USD': 3.75, 'EUR': 4.10, 'GBP': 4.75, 'SAR': 1.00, 'AED': 1.02,
            'QAR': 1.03, 'KWD': 12.25, 'BHD': 9.95, 'OMR': 9.75, 'JPY': 0.025,
            'CNY': 0.52, 'INR': 0.045, 'PKR': 0.013, 'THB': 0.106, 'MYR': 0.84,
            'SGD': 2.78, 'CAD': 2.75, 'AUD': 2.45, 'CHF': 4.15
        }
    
    def detect_currency_in_text(self, text):
        """Detect currency from text using symbols and abbreviations"""
        if pd.isna(text):
            return None
        
        text_upper = str(text).upper().strip()
        
        for currency, symbols in self.currency_symbols.items():
            for symbol in symbols:
                if symbol.upper() in text_upper:
                    return currency
        return None
    
    def extract_currency_from_amount(self, amount_text):
        """Extract currency and numeric value from amount text"""
        if pd.isna(amount_text):
            return None, None
        
        text = str(amount_text).strip()
        detected_currency = self.detect_currency_in_text(text)
        
        # Extract numeric value
        numeric_text = re.sub(r'[^\d.,\-+]', '', text)
        
        try:
            if ',' in numeric_text and '.' in numeric_text:
                numeric_text = numeric_text.replace(',', '')
            elif ',' in numeric_text and numeric_text.count(',') == 1:
                if len(numeric_text.split(',')[1]) <= 2:
                    numeric_text = numeric_text.replace(',', '.')
                else:
                    numeric_text = numeric_text.replace(',', '')
            
            numeric_value = float(numeric_text) if numeric_text else 0
        except (ValueError, TypeError):
            numeric_value = 0
        
        return detected_currency, numeric_value
    
    def get_exchange_rates(self):
        """Get exchange rates to SAR"""
        self.exchange_rates = self.fallback_rates.copy()
        self.last_updated = datetime.now()
        return True
    
    def convert_to_sar(self, amount, from_currency):
        """Convert amount from any currency to SAR"""
        if pd.isna(amount) or amount == 0:
            return 0
        
        if pd.isna(from_currency) or from_currency is None:
            return float(amount)
        
        from_currency = str(from_currency).upper().strip()
        
        if from_currency == 'SAR':
            return float(amount)
        
        rate = self.exchange_rates.get(from_currency)
        if rate is None:
            detected_curr = self.detect_currency_in_text(from_currency)
            if detected_curr:
                rate = self.exchange_rates.get(detected_curr)
        
        if rate is None:
            return float(amount)
        
        return float(amount) * rate

@st.cache_data
def load_data(uploaded_file=None):
    """Load and cache the procurement data with enhanced column mapping and currency conversion"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Uploaded file loaded successfully!")
            st.session_state['data_source'] = uploaded_file.name
        else:
            # Try to load available files
            possible_files = [
                'PO_Model_Optimized_Large.csv',
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
                # Generate sample data if no files found
                st.sidebar.info("📊 No data files found. Generating sample data for demonstration.")
                df = generate_sample_data()
                st.session_state['data_source'] = 'Generated Sample Data'
            else:
                st.sidebar.info(f"📁 Auto-loaded: {loaded_file}")
        
        # Currency conversion
        converter = CurrencyConverter()
        converter.get_exchange_rates()
        
        # Detect and convert currencies
        currency_columns = [col for col in df.columns if 'currency' in col.lower() or 'curr' in col.lower()]
        monetary_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['price', 'cost', 'amount', 'value', 'total'])]
        
        conversion_count = 0
        
        # Process monetary columns
        for col in monetary_columns:
            if df[col].dtype == 'object':  # Text column that might contain currency
                converted_amounts = []
                for value in df[col]:
                    currency, amount = converter.extract_currency_from_amount(value)
                    converted_amount = converter.convert_to_sar(amount, currency)
                    converted_amounts.append(converted_amount)
                    if currency and currency != 'SAR':
                        conversion_count += 1
                
                df[col] = converted_amounts
            elif currency_columns:  # Separate currency column exists
                primary_currency_col = currency_columns[0]
                converted_amounts = []
                for idx, row in df.iterrows():
                    amount = row[col]
                    currency = row[primary_currency_col] if primary_currency_col in df.columns else None
                    converted_amount = converter.convert_to_sar(amount, currency)
                    converted_amounts.append(converted_amount)
                    if currency and str(currency).upper() != 'SAR':
                        conversion_count += 1
                
                df[col] = converted_amounts
        
        if conversion_count > 0:
            st.sidebar.success(f"💱 Converted {conversion_count} currency values to SAR")
            st.session_state['currency_conversions'] = conversion_count
        else:
            st.session_state['currency_conversions'] = 0
        
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
        st.sidebar.info("💡 Generating sample data for demonstration")
        df = generate_sample_data()
        st.session_state['data_source'] = 'Generated Sample Data (Error Recovery)'
        return df

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

# ADVANCED MODULES INTEGRATION

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

def show_contracting_opportunities(df):
    """Contracting Opportunities Module"""
    st.header("🤝 Contracting Opportunities")
    st.markdown("Identify optimal contracting opportunities based on spend analysis, vendor performance, and demand predictability.")
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Configuration parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_spend = st.number_input("Min Annual Spend Threshold", min_value=0, value=50000, step=10000)
    with col2:
        min_frequency = st.number_input("Min Annual Order Frequency", min_value=1, value=6, step=1)
    with col3:
        analysis_period = st.selectbox("Analysis Period", ["All Data", "Last 12 Months", "Last 6 Months"])
    
    # Filter data by period
    if analysis_period == "Last 12 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=365)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    elif analysis_period == "Last 6 Months":
        cutoff_date = df_clean['Creation Date'].max() - timedelta(days=180)
        analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
    else:
        analysis_df = df_clean
    
    if st.button("🔍 Identify Contract Opportunities", type="primary"):
        with st.spinner("Analyzing contract opportunities..."):
            contract_opportunities = []
            
            # Analyze vendor-item combinations
            vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
            
            for (vendor, item), group_data in vendor_item_combinations:
                # Calculate metrics
                total_spend = (group_data['Unit Price'] * group_data['Qty Delivered']).sum()
                order_frequency = len(group_data)
                
                # Calculate time span
                date_range = group_data['Creation Date'].max() - group_data['Creation Date'].min()
                months_span = date_range.days / 30 if date_range.days > 0 else 1
                monthly_frequency = order_frequency / months_span
                
                # Demand predictability
                monthly_demand = group_data.groupby(group_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
                demand_cv = monthly_demand.std() / monthly_demand.mean() if len(monthly_demand) > 1 and monthly_demand.mean() > 0 else 1
                demand_predictability = max(0, 1 - demand_cv)
                
                # Contract suitability score
                spend_score = min(total_spend / min_spend, 1.0) if min_spend > 0 else 1.0
                frequency_score = min(monthly_frequency / (min_frequency / 12), 1.0)
                
                suitability_score = (spend_score * 0.4 + frequency_score * 0.3 + demand_predictability * 0.3)
                
                # Contract recommendation
                if suitability_score >= 0.7 and total_spend >= min_spend:
                    recommendation = "High Priority"
                elif suitability_score >= 0.5 and total_spend >= min_spend * 0.5:
                    recommendation = "Medium Priority"
                elif suitability_score >= 0.3:
                    recommendation = "Low Priority"
                else:
                    recommendation = "Not Suitable"
                
                if recommendation != "Not Suitable":
                    item_desc = group_data['Item Description'].iloc[0] if 'Item Description' in group_data.columns else f"Item {item}"
                    
                    contract_opportunities.append({
                        'Vendor Name': vendor,
                        'Item': item,
                        'Item Description': item_desc[:50] + "..." if len(item_desc) > 50 else item_desc,
                        'Annual Spend': total_spend,
                        'Order Frequency': order_frequency,
                        'Monthly Frequency': monthly_frequency,
                        'Demand Predictability': demand_predictability,
                        'Suitability Score': suitability_score,
                        'Contract Priority': recommendation,
                        'Avg Unit Price': group_data['Unit Price'].mean(),
                    })
            
            if contract_opportunities:
                opportunities_df = pd.DataFrame(contract_opportunities)
                opportunities_df = opportunities_df.sort_values(['Suitability Score', 'Annual Spend'], ascending=[False, False])
                
                # Summary metrics
                total_contract_spend = opportunities_df['Annual Spend'].sum()
                high_priority_count = len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])
                avg_suitability = opportunities_df['Suitability Score'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Opportunities", len(opportunities_df))
                with col2:
                    st.metric("High Priority Items", high_priority_count)
                with col3:
                    st.metric("Total Contract Spend", f"${total_contract_spend:,.0f}")
                with col4:
                    st.metric("Avg Suitability Score", f"{avg_suitability:.2f}")
                
                # Priority distribution
                priority_counts = opportunities_df['Contract Priority'].value_counts()
                
                fig = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title="Contract Priority Distribution",
                    color_discrete_map={
                        'High Priority': '#ff6b6b',
                        'Medium Priority': '#ffd93d',
                        'Low Priority': '#6bcf7f'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Contract Opportunities Details")
                
                st.dataframe(
                    opportunities_df.style.format({
                        'Annual Spend': '{:,.0f}',
                        'Monthly Frequency': '{:.1f}',
                        'Demand Predictability': '{:.2f}',
                        'Suitability Score': '{:.2f}',
                        'Avg Unit Price': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No contract opportunities found with the current criteria.")

def show_lot_size_optimization(df):
    """LOT Size Optimization Module"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management.")
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
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
    
    # Calculate EOQ for all items
    optimization_results = []
    
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        if len(item_data) >= 3:  # Need minimum data points
            annual_demand = item_data['Qty Delivered'].sum()
            avg_unit_cost = item_data['Unit Price'].mean()
            current_avg_order = item_data['Qty Delivered'].mean()
            
            holding_cost = avg_unit_cost * holding_cost_rate
            
            if annual_demand > 0 and holding_cost > 0:
                eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                def total_cost(order_qty):
                    if order_qty <= 0:
                        return float('inf')
                    ordering_cost_total = (annual_demand / order_qty) * ordering_cost
                    holding_cost_total = (order_qty / 2) * holding_cost
                    return ordering_cost_total + holding_cost_total
                
                eoq_cost = total_cost(eoq)
                current_cost = total_cost(current_avg_order)
                potential_savings = current_cost - eoq_cost
                
                optimization_results.append({
                    'Item': item,
                    'Annual Demand': annual_demand,
                    'Current Avg Order': current_avg_order,
                    'Optimal EOQ': eoq,
                    'Current Cost': current_cost,
                    'EOQ Cost': eoq_cost,
                    'Potential Savings': potential_savings,
                    'Savings %': (potential_savings / current_cost * 100) if current_cost > 0 else 0
                })
    
    if optimization_results:
        results_df = pd.DataFrame(optimization_results)
        results_df = results_df.sort_values('Potential Savings', ascending=False)
        
        # Summary metrics
        total_savings = results_df['Potential Savings'].sum()
        avg_savings_pct = results_df['Savings %'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Potential Savings", f"${total_savings:,.0f}")
        with col2:
            st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
        with col3:
            st.metric("Items Analyzed", len(results_df))
        
        # Top opportunities
        st.subheader("🎯 Top Optimization Opportunities")
        
        display_df = results_df.head(15)[['Item', 'Current Avg Order', 'Optimal EOQ', 'Potential Savings', 'Savings %']]
        
        st.dataframe(
            display_df.style.format({
                'Current Avg Order': '{:.0f}',
                'Optimal EOQ': '{:.0f}',
                'Potential Savings': '${:,.0f}',
                'Savings %': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        # Visualization
        fig = px.bar(results_df.head(10), 
                    x='Potential Savings', 
                    y='Item',
                    orientation='h',
                    title="Top 10 Items by Savings Potential")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need more data to perform EOQ optimization.")

def show_seasonal_price_optimization(df):
    """Seasonal Price Optimization Module"""
    st.header("🌟 Seasonal Price Optimization")
    st.markdown("Optimize purchase timing based on seasonal price patterns for maximum cost savings.")
    
    # Basic data validation
    required_columns = ['Creation Date', 'Unit Price', 'Item']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean and prepare data
    df_clean = df.copy()
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date', 'Unit Price', 'Item'])
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Add date components
    df_clean['Year'] = df_clean['Creation Date'].dt.year
    df_clean['Month'] = df_clean['Creation Date'].dt.month
    df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name()
    
    # Item selection
    items = sorted(df_clean['Item'].unique())
    selected_item = st.selectbox("Select Item for Analysis", items)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item]
        
        if len(item_data) < 5:
            st.warning(f"Not enough data points for {selected_item} (need at least 5)")
        else:
            # Monthly price trends
            monthly_prices = item_data.groupby('Month_Name')['Unit Price'].agg(['mean', 'std', 'count']).reset_index()
            
            # Order months correctly
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_prices['Month_Order'] = monthly_prices['Month_Name'].apply(
                lambda x: month_order.index(x) if x in month_order else 12
            )
            monthly_prices = monthly_prices.sort_values('Month_Order')
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_price = item_data['Unit Price'].mean()
                st.metric("Average Price", f"${avg_price:.2f}")
            with col2:
                min_month = monthly_prices.loc[monthly_prices['mean'].idxmin(), 'Month_Name']
                min_price = monthly_prices['mean'].min()
                st.metric("Cheapest Month", min_month, f"${min_price:.2f}")
            with col3:
                max_month = monthly_prices.loc[monthly_prices['mean'].idxmax(), 'Month_Name']
                max_price = monthly_prices['mean'].max()
                st.metric("Most Expensive Month", max_month, f"${max_price:.2f}")
            
            # Price trend chart
            fig = px.line(monthly_prices, x='Month_Name', y='mean',
                         title=f"Monthly Price Trends - Item {selected_item}",
                         labels={'mean': 'Average Unit Price ($)', 'Month_Name': 'Month'})
            fig.update_traces(line=dict(width=3))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(df):
    """Spend Categorization & Anomaly Detection Module"""
    st.header("📊 Spend Categorization & Anomaly Detection")
    st.markdown("AI-powered spend categorization and anomaly detection for complete spend visibility.")
    
    # Basic data validation
    required_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    # Calculate line total if missing
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Anomaly Detection
    contamination_rate = st.slider("Anomaly Detection Sensitivity (%)", 1, 10, 5) / 100
    
    if st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies using AI..."):
            
            # Prepare features for anomaly detection
            features = ['Unit Price', 'Qty Delivered', 'Line Total']
            feature_data = df_clean[features].fillna(df_clean[features].median())
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_features)
            
            # Add anomaly labels
            df_clean['Is_Anomaly'] = anomaly_labels == -1
            
            # Anomaly summary
            total_anomalies = df_clean['Is_Anomaly'].sum()
            anomaly_spend = df_clean[df_clean['Is_Anomaly']]['Line Total'].sum()
            total_spend = df_clean['Line Total'].sum()
            anomaly_percent = (anomaly_spend / total_spend * 100) if total_spend > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Detected", total_anomalies)
            with col2:
                st.metric("Anomaly Spend", f"${anomaly_spend:,.0f}")
            with col3:
                st.metric("Anomaly %", f"{anomaly_percent:.1f}%")
            
            if total_anomalies > 0:
                # Anomaly details
                st.subheader("🚨 Detected Anomalies")
                
                anomaly_data = df_clean[df_clean['Is_Anomaly']].sort_values('Line Total', ascending=False)
                display_cols = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Line Total']
                
                # Show available columns only
                available_cols = [col for col in display_cols if col in anomaly_data.columns]
                
                st.dataframe(
                    anomaly_data[available_cols].head(20).style.format({
                        'Unit Price': '${:.2f}',
                        'Qty Delivered': '{:.1f}',
                        'Line Total': '${:,.0f}'
                    }),
                    use_container_width=True
                )
                
                # Anomaly visualization
                fig = px.scatter(df_clean, 
                                x='Unit Price', 
                                y='Qty Delivered',
                                color='Is_Anomaly',
                                size='Line Total',
                                title="Anomaly Detection: Price vs Quantity",
                                labels={'Is_Anomaly': 'Anomaly'},
                                color_discrete_map={True: 'red', False: 'blue'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No significant anomalies detected with current sensitivity settings.")

def show_cross_region_optimization(df):
    """Cross-Region Vendor Optimization"""
    st.header("🌍 Cross-Region Vendor Optimization")
    
    if 'Item' not in df.columns or 'W/H' not in df.columns:
        st.error("This module requires 'Item' and 'W/H' (Warehouse) columns")
        return
    
    # Select item to analyze across regions/vendors
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    filtered = df[df["Item"] == item]
    
    if len(filtered) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by vendor and warehouse to compare pricing
    result = (
        filtered.groupby(["Vendor Name", "W/H"])["Unit Price"]
        .mean()
        .reset_index()
        .sort_values(by="Unit Price")
    )
    
    # Show results
    st.write("Average Unit Price by Vendor and Warehouse:")
    st.dataframe(result, use_container_width=True)
    
    # Visualization
    if len(result) > 0:
        fig = px.bar(result, x='Unit Price', y='Vendor Name', color='W/H',
                    title=f"Price Comparison for Item {item}",
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)

def show_reorder_prediction(df):
    """Smart Reorder Point Prediction"""
    st.header("📈 Smart Reorder Point Prediction")
    
    if 'Item' not in df.columns or 'Creation Date' not in df.columns or 'Qty Delivered' not in df.columns:
        st.error("This module requires 'Item', 'Creation Date', and 'Qty Delivered' columns")
        return
    
    # User selects an item to analyze
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    
    # Filter the dataset for that item
    item_df = df[df["Item"] == item].copy()
    
    if len(item_df) == 0:
        st.warning("No data found for selected item")
        return
    
    # Group by month and sum quantity delivered
    item_df["Month"] = pd.to_datetime(item_df["Creation Date"]).dt.to_period("M")
    demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)
    
    # Calculate reorder point using basic statistical method
    if len(demand_by_month) > 0:
        reorder_point = demand_by_month.mean() + demand_by_month.std()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Monthly Demand", f"{demand_by_month.mean():.1f}")
        with col2:
            st.metric("Demand Variability", f"{demand_by_month.std():.1f}")
        with col3:
            st.metric("Suggested Reorder Point", f"{reorder_point:.1f}")
        
        # Chart
        fig = px.line(x=demand_by_month.index.astype(str), y=demand_by_month.values,
                     title=f"Monthly Demand Pattern - Item {item}",
                     labels={'x': 'Month', 'y': 'Quantity Delivered'})
        fig.add_hline(y=reorder_point, line_dash="dash", line_color="red",
                     annotation_text=f"Reorder Point: {reorder_point:.1f}")
        st.plotly_chart(fig, use_container_width=True)

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
    st.markdown('<div class="main-header">📊 Procurement Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("**Advanced procurement analytics with AI-powered insights and optimization tools**")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Platform Controls")
    
    # File upload
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your procurement data (CSV)",
        type=['csv'],
        help="Upload a CSV file with procurement/PO data"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("❌ No data available. Please upload a CSV file or check your data files.")
        return
    
    # Apply filters
    st.sidebar.subheader("🔧 Filters")
    filtered_df = apply_filters(df)
    
    # Navigation
    st.sidebar.subheader("📑 Analytics Modules")
    page = st.sidebar.selectbox(
        "Choose a module:",
        [
            "📊 Overview Dashboard",
            "🤝 Contracting Opportunities", 
            "📦 LOT Size Optimization",
            "🌟 Seasonal Price Optimization",
            "🚨 Anomaly Detection",
            "🌍 Cross-Region Optimization",
            "📈 Reorder Prediction",
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
    elif page == "🤝 Contracting Opportunities":
        show_contracting_opportunities(filtered_df)
    elif page == "📦 LOT Size Optimization":
        show_lot_size_optimization(filtered_df)
    elif page == "🌟 Seasonal Price Optimization":
        show_seasonal_price_optimization(filtered_df)
    elif page == "🚨 Anomaly Detection":
        show_anomaly_detection(filtered_df)
    elif page == "🌍 Cross-Region Optimization":
        show_cross_region_optimization(filtered_df)
    elif page == "📈 Reorder Prediction":
        show_reorder_prediction(filtered_df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("*🚀 Advanced Procurement Analytics Platform - Built with Streamlit & AI*")

if __name__ == "__main__":
    main()
