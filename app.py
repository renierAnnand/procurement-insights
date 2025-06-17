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
# Optional imports for advanced analytics
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
from math import sqrt
warnings.filterwarnings('ignore')

# Import all your existing modules
try:
    import contracting_opportunities
    CONTRACTING_MODULE = True
except ImportError:
    CONTRACTING_MODULE = False

try:
    import lot_size_optimization
    LOT_MODULE = True
except ImportError:
    LOT_MODULE = False

try:
    import seasonal_price_optimization
    SEASONAL_MODULE = True
except ImportError:
    SEASONAL_MODULE = False

try:
    import spend_categorization_anomaly
    ANOMALY_MODULE = True
except ImportError:
    ANOMALY_MODULE = False

try:
    import cross_region
    CROSS_REGION_MODULE = True
except ImportError:
    CROSS_REGION_MODULE = False

try:
    import reorder_prediction
    REORDER_MODULE = True
except ImportError:
    REORDER_MODULE = False

try:
    import duplicates
    DUPLICATES_MODULE = True
except ImportError:
    DUPLICATES_MODULE = False

# Page configuration
st.set_page_config(
    page_title="Complete Procurement Analytics Platform",
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
    .module-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        font-size: 0.8rem;
    }
    .module-available {
        background-color: #d4edda;
        color: #155724;
    }
    .module-missing {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

class CurrencyConverter:
    """Advanced currency converter for procurement analytics - converts all currencies to Saudi Riyals (SAR)"""
    
    def __init__(self):
        self.base_currency = 'SAR'  # All conversions are to Saudi Riyals
        # Exchange rates: 1 unit of foreign currency = X Saudi Riyals
        self.exchange_rates = {
            'USD': 3.75, 'EUR': 4.10, 'GBP': 4.75, 'SAR': 1.00, 'AED': 1.02,
            'QAR': 1.03, 'KWD': 12.25, 'BHD': 9.95, 'OMR': 9.75, 'JPY': 0.025,
            'CNY': 0.52, 'INR': 0.045, 'PKR': 0.013, 'THB': 0.106, 'MYR': 0.84,
            'SGD': 2.78, 'CAD': 2.75, 'AUD': 2.45, 'CHF': 4.15, 'NOK': 0.34,
            'SEK': 0.35, 'DKK': 0.55, 'PLN': 0.93, 'CZK': 0.16, 'HUF': 0.01,
            'RUB': 0.04, 'TRY': 0.11, 'ZAR': 0.20, 'EGP': 0.08, 'JOD': 5.30,
            'LBP': 0.00025, 'KRW': 0.0028, 'TWD': 0.12, 'HKD': 0.48, 'PHP': 0.067,
            'IDR': 0.00025, 'VND': 0.00015, 'BRL': 0.62, 'MXN': 0.18, 'CLP': 0.0038,
            'ARS': 0.0037, 'COP': 0.00085, 'PEN': 1.00
        }
        
        self.currency_patterns = {
            'USD': ['USD', 'DOLLAR', 'DOLLARS', '$'],
            'EUR': ['EUR', 'EURO', 'EUROS'],
            'GBP': ['GBP', 'POUND', 'POUNDS', 'STERLING'],
            'SAR': ['SAR', 'RIYAL', 'RIYALS', 'SR'],
            'AED': ['AED', 'DIRHAM', 'DIRHAMS'],
            'QAR': ['QAR', 'QATAR'],
            'KWD': ['KWD', 'DINAR', 'DINARS'],
            'INR': ['INR', 'RUPEE', 'RUPEES'],
            'CNY': ['CNY', 'YUAN', 'RMB'],
            'JPY': ['JPY', 'YEN'],
            'THB': ['THB', 'BAHT'],
            'MYR': ['MYR', 'RINGGIT'],
            'SGD': ['SGD', 'SINGAPORE'],
            'CAD': ['CAD', 'CANADIAN'],
            'AUD': ['AUD', 'AUSTRALIAN'],
            'CHF': ['CHF', 'FRANC', 'FRANCS']
        }
    
    def detect_currency(self, text):
        """Detect currency from text"""
        if pd.isna(text):
            return None
        
        text_upper = str(text).upper()
        
        for currency, patterns in self.currency_patterns.items():
            for pattern in patterns:
                if pattern in text_upper:
                    return currency
        
        return None
    
    def extract_amount(self, text):
        """Extract numeric amount from text"""
        if pd.isna(text):
            return 0
        
        # Remove non-numeric characters except dots and commas
        numeric_text = re.sub(r'[^\d.,\-]', '', str(text))
        
        if not numeric_text:
            return 0
        
        try:
            # Handle different number formats
            if ',' in numeric_text and '.' in numeric_text:
                # Format like 1,234.56
                numeric_text = numeric_text.replace(',', '')
            elif ',' in numeric_text:
                # Could be thousands separator or decimal
                parts = numeric_text.split(',')
                if len(parts[-1]) <= 2:
                    # Decimal comma (European format)
                    numeric_text = numeric_text.replace(',', '.')
                else:
                    # Thousands separator
                    numeric_text = numeric_text.replace(',', '')
            
            return float(numeric_text)
        except:
            return 0
    
    def convert_to_sar(self, amount, currency):
        """Convert amount from any currency to Saudi Riyals (SAR)"""
        if pd.isna(amount) or amount == 0:
            return 0
        
        if pd.isna(currency):
            return float(amount)  # Assume already in SAR if no currency specified
        
        currency = str(currency).upper().strip()
        
        if currency == 'SAR':
            return float(amount)  # Already in SAR
        
        rate = self.exchange_rates.get(currency, 1.0)
        converted_amount = float(amount) * rate
        return converted_amount

# Built-in data loading and cleaning functions (replacing utils.py)
def load_and_clean_data(csv_file):
    """Load and clean procurement data with currency conversion"""
    
    # Load the CSV file
    if isinstance(csv_file, str):
        # File path
        df = pd.read_csv(csv_file)
    else:
        # Uploaded file object
        df = pd.read_csv(csv_file)
    
    # Initialize currency converter
    converter = CurrencyConverter()
    
    # Track conversions
    conversion_count = 0
    currencies_found = set()
    
    # Enhanced column mapping for better compatibility
    column_mappings = {
        'Creation Date': ['Creation Date', 'creation_date', 'create_date', 'po_date', 'order_date', 'date'],
        'Vendor Name': ['Vendor Name', 'vendor_name', 'supplier_name', 'supplier', 'vendor'],
        'Item': ['Item', 'item', 'item_code', 'product_code', 'part_number'],
        'Item Description': ['Item Description', 'item_description', 'description', 'product_desc', 'item_desc'],
        'Unit Price': ['Unit Price', 'unit_price', 'price', 'cost', 'unit_cost'],
        'Qty Delivered': ['Qty Delivered', 'qty_delivered', 'quantity_delivered', 'delivered_qty', 'quantity', 'qty'],
        'Qty Ordered': ['Qty Ordered', 'qty_ordered', 'quantity_ordered', 'ordered_qty', 'order_qty'],
        'Qty Rejected': ['Qty Rejected', 'qty_rejected', 'rejected_qty', 'rejection_qty'],
        'Line Total': ['Line Total', 'line_total', 'total', 'amount', 'line_amount'],
        'Product Family': ['Product Family', 'product_family', 'category', 'product_category'],
        'DEP': ['DEP', 'dep', 'department', 'dept'],
        'W/H': ['W/H', 'warehouse', 'wh', 'location'],
        'Buyer': ['Buyer', 'buyer', 'purchaser', 'buyer_name'],
        'PO Status': ['PO Status', 'status', 'po_status', 'order_status'],
        'Approved Date': ['Approved Date', 'approved_date', 'approval_date'],
        'PO Receipt Date': ['PO Receipt Date', 'receipt_date', 'received_date'],
        'Requested Delivery Date': ['Requested Delivery Date', 'requested_date', 'req_delivery_date'],
        'Promised Delivery Date': ['Promised Delivery Date', 'promised_date', 'delivery_date']
    }
    
    # Apply column mappings
    for standard_name, possible_names in column_mappings.items():
        for possible_name in possible_names:
            if possible_name in df.columns and standard_name not in df.columns:
                df[standard_name] = df[possible_name]
                break
    
    # Handle mixed item types (alphanumeric codes vs numeric IDs)
    if 'Item' in df.columns:
        # Create both numeric and text versions for compatibility
        df['Item_Original'] = df['Item'].astype(str)  # Keep original as text
        
        # Try to create numeric Item IDs for modules that need them
        unique_items = df['Item'].astype(str).unique()
        item_mapping = {item: idx + 1 for idx, item in enumerate(unique_items)}
        df['Item_Numeric'] = df['Item'].astype(str).map(item_mapping)
        
        # Use numeric version as default Item for compatibility
        df['Item'] = df['Item_Numeric']
        
        # Store mapping for reference
        if 'item_mapping' not in st.session_state:
            st.session_state['item_mapping'] = item_mapping
            st.session_state['reverse_item_mapping'] = {v: k for k, v in item_mapping.items()}
    
    # Find currency column
    currency_column = None
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['currency', 'curr', 'ccy']):
            currency_column = col
            break
    
    # Process currency conversion for price columns - convert everything to SAR
    price_columns = ['Unit Price', 'Line Total', 'Total', 'Amount', 'Cost', 'Value']
    
    for price_col in price_columns:
        if price_col in df.columns:
            converted_values = []
            
            for idx, row in df.iterrows():
                price_value = row[price_col]
                
                # Get currency
                if currency_column and currency_column in df.columns:
                    currency = row[currency_column]
                else:
                    # Try to detect currency from price text
                    currency = converter.detect_currency(str(price_value))
                
                # Extract numeric amount
                if isinstance(price_value, str):
                    amount = converter.extract_amount(price_value)
                else:
                    amount = price_value if not pd.isna(price_value) else 0
                
                # Convert to SAR (Saudi Riyals)
                converted_amount = converter.convert_to_sar(amount, currency)
                converted_values.append(converted_amount)
                
                # Track conversion stats
                if currency and currency != 'SAR':
                    conversion_count += 1
                    currencies_found.add(currency)
            
            # Update the column with converted values (now all in SAR)
            df[price_col] = converted_values
    
    # Calculate Line Total if missing
    if 'Line Total' not in df.columns:
        if 'Unit Price' in df.columns and 'Qty Delivered' in df.columns:
            df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
        elif 'Unit Price' in df.columns and 'Qty Ordered' in df.columns:
            df['Line Total'] = df['Unit Price'] * df['Qty Ordered']
    
    # Add missing columns with default values for better module compatibility
    default_columns = {
        'Qty Rejected': 0,
        'Product Family': 'General',
        'DEP': 'General',
        'W/H': 'Main',
        'Buyer': 'System',
        'PO Status': 'Completed'
    }
    
    for col, default_val in default_columns.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['Unit Price', 'Qty Delivered', 'Qty Ordered', 'Line Total', 'Qty Rejected']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert date columns
    date_columns = ['Creation Date', 'Approved Date', 'PO Receipt Date', 'Requested Delivery Date', 'Promised Delivery Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Fill missing values in text columns
    text_columns = ['Vendor Name', 'Item Description', 'Product Family', 'DEP', 'W/H', 'Buyer', 'PO Status']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Store conversion info
    st.session_state['currency_conversions'] = conversion_count
    st.session_state['currencies_found'] = list(currencies_found)
    
    return df

def forecast_demand(df, periods=30):
    """Built-in demand forecasting function"""
    
    # Check required columns
    if 'Creation Date' not in df.columns:
        st.error("Creation Date column required for forecasting")
        return None, None
    
    # Find quantity column
    qty_col = None
    for col in ['Qty Delivered', 'Qty Ordered', 'Quantity']:
        if col in df.columns:
            qty_col = col
            break
    
    if qty_col is None:
        st.error("No quantity column found for forecasting")
        return None, None
    
    # Prepare time series data
    df_ts = df.copy()
    df_ts['Date'] = pd.to_datetime(df_ts['Creation Date'])
    
    # Group by date and sum quantities
    daily_demand = df_ts.groupby('Date')[qty_col].sum().sort_index()
    
    # Fill missing dates
    date_range = pd.date_range(start=daily_demand.index.min(), 
                              end=daily_demand.index.max(), 
                              freq='D')
    daily_demand = daily_demand.reindex(date_range, fill_value=0)
    
    # Simple forecasting using moving average and trend
    window = min(7, len(daily_demand) // 4)  # Adaptive window size
    
    if len(daily_demand) < 7:
        # Not enough data for sophisticated forecasting
        avg_demand = daily_demand.mean()
        forecast_dates = pd.date_range(start=daily_demand.index.max() + timedelta(days=1), 
                                     periods=periods, freq='D')
        forecast_values = [avg_demand] * periods
    else:
        # Moving average
        ma = daily_demand.rolling(window=window, min_periods=1).mean()
        
        # Simple trend calculation
        recent_data = daily_demand.tail(window)
        if len(recent_data) > 1:
            x = np.arange(len(recent_data))
            y = recent_data.values
            trend = np.polyfit(x, y, 1)[0]  # Linear trend
        else:
            trend = 0
        
        # Generate forecasts
        last_ma = ma.iloc[-1]
        forecast_dates = pd.date_range(start=daily_demand.index.max() + timedelta(days=1), 
                                     periods=periods, freq='D')
        
        # Apply trend and add some noise
        forecast_values = []
        for i in range(periods):
            base_forecast = last_ma + (trend * (i + 1))
            # Add slight randomness but keep positive
            noise = np.random.normal(0, abs(base_forecast) * 0.1)
            forecast_val = max(0, base_forecast + noise)
            forecast_values.append(forecast_val)
    
    # Create forecast series
    forecast_series = pd.Series(forecast_values, index=forecast_dates)
    
    return daily_demand, forecast_series

def safe_module_display(module, module_name, df):
    """Safely display module with error handling"""
    try:
        module.display(df)
    except Exception as e:
        st.error(f"❌ Error in {module_name} module:")
        
        # Show detailed error for debugging
        error_details = str(e)
        st.code(error_details)
        
        # Provide specific solutions based on module
        if module_name == "Contracting Opportunities":
            st.info("💡 **Contracting Opportunities Error Solutions:**")
            st.write("• This might be a data compatibility issue")
            st.write("• Try using the built-in Enhanced Contract Analysis below")
            
            # Provide enhanced contract analysis as fallback
            st.markdown("---")
            st.subheader("🤝 Enhanced Contract Analysis (Fallback)")
            
            enhanced_contract_analysis(df)
            
        else:
            st.info("💡 **Possible Solutions:**")
            st.write("• Check if your data has the required columns")
            st.write("• Try using a different dataset")
            st.write("• Contact support if the error persists")
        
        # Show data structure for debugging
        if st.checkbox(f"🔍 Show data structure for debugging ({module_name})"):
            st.write("**Available Columns:**")
            st.write(list(df.columns))
            st.write("**Data Sample:**")
            st.dataframe(df.head(3))

def enhanced_contract_analysis(df):
    """Enhanced contract analysis as fallback for contracting opportunities"""
    
    if df is None or df.empty:
        st.warning("No data available for contract analysis")
        return
    
    st.info("🔧 **Built-in Contract Analysis** - Enhanced fallback analysis")
    st.info("🏛️ **All monetary values shown in Saudi Riyals (SAR)** after automatic currency conversion")
    
    # Show item mapping info if available
    if 'item_mapping' in st.session_state:
        with st.expander("🔍 Item Code Mapping"):
            st.write("**Item codes have been converted to numeric IDs for compatibility:**")
            mapping_df = pd.DataFrame([
                {'Original Code': k, 'Numeric ID': v} 
                for k, v in list(st.session_state['item_mapping'].items())[:10]
            ])
            st.dataframe(mapping_df)
            if len(st.session_state['item_mapping']) > 10:
                st.write(f"... and {len(st.session_state['item_mapping']) - 10} more items")
    
    # Basic contract opportunity analysis
    tabs = st.tabs(["📊 Spend Analysis", "🎯 Top Opportunities", "💰 Savings Potential", "📋 Recommendations"])
    
    with tabs[0]:
        st.subheader("📊 Spend Analysis by Vendor")
        
        if 'Vendor Name' in df.columns and 'Line Total' in df.columns:
            # Vendor spend analysis
            vendor_spend = df.groupby('Vendor Name').agg({
                'Line Total': ['sum', 'count', 'mean'],
                'Item': 'nunique'
            }).round(2)
            
            vendor_spend.columns = ['Total Spend', 'Order Count', 'Avg Order Value', 'Unique Items']
            vendor_spend = vendor_spend.sort_values('Total Spend', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 10 vendors by spend
                top_vendors = vendor_spend.head(10)
                
                fig = px.bar(
                    x=top_vendors['Total Spend'],
                    y=top_vendors.index,
                    orientation='h',
                    title="Top 10 Vendors by Total Spend (SAR)"
                )
                fig.update_layout(height=400)
                fig.update_xaxes(title="Total Spend (SAR)")
                fig.update_yaxes(title="Vendor Name")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Vendor concentration analysis
                total_spend = vendor_spend['Total Spend'].sum()
                top_5_concentration = (vendor_spend.head(5)['Total Spend'].sum() / total_spend * 100)
                top_10_concentration = (vendor_spend.head(10)['Total Spend'].sum() / total_spend * 100)
                
                st.metric("Top 5 Vendors Concentration", f"{top_5_concentration:.1f}%")
                st.metric("Top 10 Vendors Concentration", f"{top_10_concentration:.1f}%")
                
                if top_5_concentration > 60:
                    st.warning("⚠️ High vendor concentration risk")
                elif top_5_concentration > 40:
                    st.info("ℹ️ Moderate vendor concentration")
                else:
                    st.success("✅ Good vendor diversification")
            
            # Detailed vendor table
            st.subheader("📋 Detailed Vendor Analysis")
            st.dataframe(
                vendor_spend.head(20).style.format({
                    'Total Spend': '{:,.0f}',
                    'Avg Order Value': '{:,.0f}'
                }),
                use_container_width=True
            )
    
    with tabs[1]:
        st.subheader("🎯 Top Contract Opportunities")
        
        # Contract opportunity scoring
        if 'Vendor Name' in df.columns and 'Line Total' in df.columns:
            
            # Configuration
            col1, col2 = st.columns(2)
            with col1:
                min_annual_spend = st.number_input("Minimum Annual Spend Threshold (SAR)", 
                                                 min_value=0, value=50000, step=10000)
            with col2:
                min_orders = st.number_input("Minimum Annual Orders", 
                                           min_value=1, value=6, step=1)
            
            # Calculate contract opportunities
            opportunities = []
            
            for vendor in df['Vendor Name'].unique():
                vendor_data = df[df['Vendor Name'] == vendor]
                
                total_spend = vendor_data['Line Total'].sum()
                order_count = len(vendor_data)
                unique_items = vendor_data['Item'].nunique() if 'Item' in df.columns else 1
                avg_order_value = vendor_data['Line Total'].mean()
                
                # Get sample of actual item codes for display
                if 'Item_Original' in df.columns:
                    sample_items = vendor_data['Item_Original'].unique()[:3]
                    items_display = ', '.join([str(item) for item in sample_items])
                    if len(vendor_data['Item_Original'].unique()) > 3:
                        items_display += f" (+{len(vendor_data['Item_Original'].unique()) - 3} more)"
                else:
                    items_display = f"{unique_items} items"
                
                # Calculate opportunity score
                spend_score = min(total_spend / min_annual_spend, 1.0) if min_annual_spend > 0 else 1.0
                frequency_score = min(order_count / min_orders, 1.0) if min_orders > 0 else 1.0
                consistency_score = 1 - (vendor_data['Line Total'].std() / vendor_data['Line Total'].mean()) if vendor_data['Line Total'].mean() > 0 else 0
                consistency_score = max(0, min(1, consistency_score))
                
                opportunity_score = (spend_score * 0.4 + frequency_score * 0.3 + consistency_score * 0.3)
                
                if total_spend >= min_annual_spend * 0.5 and order_count >= min_orders * 0.5:
                    opportunities.append({
                        'Vendor': vendor,
                        'Total Spend': total_spend,
                        'Order Count': order_count,
                        'Items': items_display,
                        'Avg Order Value': avg_order_value,
                        'Opportunity Score': opportunity_score,
                        'Priority': 'High' if opportunity_score >= 0.7 else 'Medium' if opportunity_score >= 0.4 else 'Low'
                    })
            
            if opportunities:
                opp_df = pd.DataFrame(opportunities)
                opp_df = opp_df.sort_values('Opportunity Score', ascending=False)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Opportunities", len(opp_df))
                with col2:
                    high_priority = len(opp_df[opp_df['Priority'] == 'High'])
                    st.metric("High Priority", high_priority)
                with col3:
                    total_contract_value = opp_df['Total Spend'].sum()
                    st.metric("Total Contract Value", f"{total_contract_value:,.0f}")
                
                # Opportunities table
                st.dataframe(
                    opp_df.style.format({
                        'Total Spend': '{:,.0f}',
                        'Avg Order Value': '{:,.0f}',
                        'Opportunity Score': '{:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Priority distribution
                priority_counts = opp_df['Priority'].value_counts()
                fig = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title="Contract Priority Distribution",
                    color_discrete_map={
                        'High': '#ff6b6b',
                        'Medium': '#ffd93d',
                        'Low': '#6bcf7f'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("No contract opportunities found with current criteria. Try lowering the thresholds.")
    
    with tabs[2]:
        st.subheader("💰 Potential Savings Analysis")
        
        # Savings calculation
        col1, col2 = st.columns(2)
        with col1:
            volume_discount = st.slider("Expected Volume Discount (%)", 0, 20, 5) / 100
        with col2:
            admin_savings_per_order = st.number_input("Admin Savings per Order (SAR)", 0, 500, 100)
        
        if 'opportunities' in locals() and opportunities:
            savings_analysis = []
            
            for opp in opportunities:
                current_annual_cost = opp['Total Spend']
                current_orders = opp['Order Count']
                
                # Price savings from volume discount
                price_savings = current_annual_cost * volume_discount
                
                # Admin savings (assume contract reduces order frequency by 50%)
                reduced_orders = max(1, current_orders * 0.5)
                admin_savings = (current_orders - reduced_orders) * admin_savings_per_order
                
                total_savings = price_savings + admin_savings
                savings_percent = (total_savings / current_annual_cost * 100) if current_annual_cost > 0 else 0
                
                savings_analysis.append({
                    'Vendor': opp['Vendor'],
                    'Current Annual Cost': current_annual_cost,
                    'Price Savings': price_savings,
                    'Admin Savings': admin_savings,
                    'Total Savings': total_savings,
                    'Savings %': savings_percent
                })
            
            savings_df = pd.DataFrame(savings_analysis)
            savings_df = savings_df.sort_values('Total Savings', ascending=False)
            
            # Summary
            total_potential_savings = savings_df['Total Savings'].sum()
            total_current_cost = savings_df['Current Annual Cost'].sum()
            overall_savings_percent = (total_potential_savings / total_current_cost * 100) if total_current_cost > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Potential Savings", f"{total_potential_savings:,.0f} SAR")
            with col2:
                st.metric("Overall Savings %", f"{overall_savings_percent:.1f}%")
            with col3:
                st.metric("Avg Savings per Vendor", f"{total_potential_savings/len(savings_df):,.0f} SAR")
            
            # Detailed savings table
            st.dataframe(
                savings_df.style.format({
                    'Current Annual Cost': '{:,.0f}',
                    'Price Savings': '{:,.0f}',
                    'Admin Savings': '{:,.0f}',
                    'Total Savings': '{:,.0f}',
                    'Savings %': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            # Top savings opportunities chart
            fig = px.bar(
                savings_df.head(10),
                x='Total Savings',
                y='Vendor',
                orientation='h',
                title="Top 10 Savings Opportunities (SAR)"
            )
            fig.update_layout(height=500)
            fig.update_xaxes(title="Total Savings (SAR)")
            fig.update_yaxes(title="Vendor Name")
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run the opportunities analysis in the previous tab first.")
    
    with tabs[3]:
        st.subheader("📋 Contract Recommendations")
        
        st.markdown("""
        #### 🎯 **Contract Strategy Recommendations**
        
        **Phase 1: Quick Wins (Month 1-2)**
        - Focus on top 3-5 vendors by spend
        - Target vendors with high order frequency
        - Negotiate volume discounts for predictable items
        
        **Phase 2: Strategic Contracts (Month 3-6)**
        - Implement blanket purchase orders
        - Establish framework agreements
        - Set up automated ordering processes
        
        **Phase 3: Advanced Optimization (Month 6+)**
        - Performance-based contracts
        - Supplier development programs
        - Innovation partnerships
        
        #### 📋 **Contract Types to Consider**
        
        | Contract Type | Best For | Duration | Expected Savings |
        |---------------|----------|----------|------------------|
        | Volume Commitment | High spend, predictable demand | 12-36 months | 5-15% |
        | Blanket PO | Multiple items, same supplier | 12-24 months | 3-8% |
        | Framework Agreement | Service providers | 24-48 months | 8-20% |
        | Fixed Price | Stable market conditions | 6-18 months | 2-6% |
        
        #### ⚠️ **Risk Considerations**
        
        - **Vendor Concentration**: Monitor dependency on key suppliers
        - **Market Volatility**: Include price adjustment clauses
        - **Performance Risk**: Establish clear KPIs and penalties
        - **Contract Management**: Ensure adequate resources for oversight
        """)
        
        # Export recommendations
        if st.button("📥 Export Contract Strategy Report"):
            if 'opportunities' in locals() and opportunities:
                export_data = {
                    'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                    'total_opportunities': len(opportunities),
                    'high_priority_count': len([o for o in opportunities if o['Priority'] == 'High']),
                    'total_spend_analyzed': sum(o['Total Spend'] for o in opportunities),
                    'recommendations': [
                        'Implement volume discounts with top vendors',
                        'Establish blanket purchase orders',
                        'Negotiate performance-based contracts',
                        'Set up automated ordering processes'
                    ]
                }
                
                st.success("✅ Analysis complete! Use the data above to develop your contract strategy.")
            else:
                st.info("Complete the opportunity analysis first to generate the export.")

@st.cache_data
def generate_sample_data(num_records=1000):
    """Generate comprehensive sample data for testing all modules"""
    np.random.seed(42)
    
    vendors = [
        "Global Tech Solutions", "Premium Office Supplies", "Industrial Materials Corp",
        "Elite Manufacturing", "Smart Logistics Ltd", "Quality Parts Inc",
        "Advanced Systems", "Corporate Supplies", "Modern Equipment"
    ]
    
    product_families = ['IT & Technology', 'Office Supplies', 'Raw Materials', 'Electronics', 'Facilities', 'Marketing']
    departments = ['IT', 'Operations', 'Facilities', 'R&D', 'Marketing', 'Finance']
    warehouses = ['WH-001', 'WH-002', 'WH-003', 'WH-CENTRAL', 'WH-NORTH']
    buyers = ['John Smith', 'Sarah Johnson', 'Mike Chen', 'Lisa Wang', 'Ahmed Ali']
    statuses = ['Approved', 'Delivered', 'Pending', 'Completed']
    
    currencies = ['USD', 'EUR', 'GBP', 'SAR', 'AED', 'QAR', 'INR', 'CNY']
    currency_weights = [0.30, 0.15, 0.10, 0.25, 0.08, 0.05, 0.04, 0.03]
    
    data = []
    
    for i in range(num_records):
        currency = np.random.choice(currencies, p=currency_weights)
        
        # Different price ranges by currency - but will be converted to SAR
        if currency == 'USD':
            price = np.random.uniform(10, 500)  # $10-500 → SAR 37.5-1875
        elif currency == 'EUR':
            price = np.random.uniform(8, 450)   # €8-450 → SAR 33-1845
        elif currency == 'SAR':
            price = np.random.uniform(40, 1800) # Already in SAR
        elif currency == 'AED':
            price = np.random.uniform(35, 1700) # AED 35-1700 → SAR 36-1734
        elif currency == 'INR':
            price = np.random.uniform(800, 40000) # INR 800-40K → SAR 36-1800
        else:
            price = np.random.uniform(20, 1000)  # Other currencies → various SAR amounts
        
        creation_date = datetime.now() - timedelta(days=np.random.randint(0, 730))
        qty_ordered = np.random.randint(1, 100)
        qty_delivered = qty_ordered - np.random.randint(0, 5)  # Usually deliver most of what's ordered
        qty_rejected = max(0, np.random.randint(-2, 3))  # Usually no rejections
        
        record = {
            'Creation Date': creation_date,
            'Vendor Name': np.random.choice(vendors),
            'Item': np.random.randint(1, 50),
            'Item Description': f"Sample Product {np.random.randint(1, 100)} - {np.random.choice(['Widget', 'Component', 'Tool', 'Supply'])}",
            'Unit Price': price,
            'Currency': currency if np.random.random() < 0.7 else None,
            'Qty Delivered': qty_delivered,
            'Qty Ordered': qty_ordered,
            'Qty Rejected': qty_rejected,
            'Product Family': np.random.choice(product_families),
            'DEP': np.random.choice(departments),
            'W/H': np.random.choice(warehouses),
            'Buyer': np.random.choice(buyers),
            'PO Status': np.random.choice(statuses),
            'Approved Date': creation_date + timedelta(days=np.random.randint(1, 5)),
            'PO Receipt Date': creation_date + timedelta(days=np.random.randint(7, 14)),
            'Requested Delivery Date': creation_date + timedelta(days=np.random.randint(14, 30)),
            'Promised Delivery Date': creation_date + timedelta(days=np.random.randint(10, 25))
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

def apply_filters(df):
    """Apply filters to the dataframe"""
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
    
    return filtered_df

def display_key_metrics(df):
    """Display key procurement metrics"""
    if df is None or df.empty:
        st.warning("No data available for metrics")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_orders = len(df)
        st.metric("📋 Total Orders", f"{total_orders:,}")
    
    with col2:
        if 'Line Total' in df.columns:
            total_value = df['Line Total'].sum()
            st.metric("💰 Total Value", f"{total_value:,.0f} SAR")
        else:
            st.metric("💰 Total Value", "N/A")
    
    with col3:
        unique_vendors = df['Vendor Name'].nunique() if 'Vendor Name' in df.columns else 0
        st.metric("🏢 Unique Vendors", f"{unique_vendors:,}")
    
    with col4:
        if 'Unit Price' in df.columns:
            avg_unit_price = df['Unit Price'].mean()
            st.metric("💵 Avg Unit Price", f"{avg_unit_price:.2f} SAR")
        else:
            st.metric("💵 Avg Unit Price", "N/A")
    
    with col5:
        if 'Qty Delivered' in df.columns:
            total_qty = df['Qty Delivered'].sum()
            st.metric("📦 Total Quantity", f"{total_qty:,.0f}")
        else:
            st.metric("📦 Total Quantity", "N/A")

def show_overview_dashboard(df):
    """Display overview dashboard with key visualizations"""
    st.header("📊 Overview Dashboard")
    
    if df is None or df.empty:
        st.warning("No data available for dashboard")
        return
    
    # Currency conversion summary if available
    if 'currency_conversions' in st.session_state and st.session_state['currency_conversions'] > 0:
        st.info(f"💱 **Multi-Currency Data Converted to SAR:** {st.session_state['currency_conversions']} values converted from various currencies to Saudi Riyals (SAR)")
        
        if 'currencies_found' in st.session_state and st.session_state['currencies_found']:
            st.write(f"**Original Currencies Detected:** {', '.join(st.session_state['currencies_found'])} → **All converted to SAR**")
    
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

def show_demand_forecasting(df):
    """Enhanced demand forecasting module"""
    st.header("🔮 Smart Demand Forecasting")
    st.markdown("**AI-powered demand prediction with multi-currency support**")
    
    if df is None or df.empty:
        st.warning("No data available for forecasting")
        return
    
    # Currency conversion info
    if 'currency_conversions' in st.session_state and st.session_state['currency_conversions'] > 0:
        st.success(f"💱 **Currency Conversion Applied:** {st.session_state['currency_conversions']} values converted to SAR for accurate forecasting in Saudi Riyals")
    
    # Forecasting parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_days = st.number_input("Forecast Days", min_value=7, max_value=90, value=30)
    with col2:
        if 'Item' in df.columns:
            items = sorted(df['Item'].unique())
            selected_item = st.selectbox("Select Item (Optional)", ['All Items'] + [str(item) for item in items])
        else:
            selected_item = 'All Items'
    with col3:
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
    
    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Generating demand forecast..."):
            
            # Filter data if specific item selected
            if selected_item != 'All Items':
                forecast_df = df[df['Item'] == int(selected_item)]
                title_suffix = f" - Item {selected_item}"
            else:
                forecast_df = df
                title_suffix = " - All Items"
            
            if len(forecast_df) == 0:
                st.error("No data found for selected item")
                return
            
            # Generate forecast
            try:
                historical_ts, forecast_ts = forecast_demand(forecast_df, periods=forecast_days)
                
                if historical_ts is not None and forecast_ts is not None:
                    
                    # Display forecast metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_historical = historical_ts.mean()
                        st.metric("📊 Avg Historical Demand", f"{avg_historical:.1f}")
                    with col2:
                        avg_forecast = forecast_ts.mean()
                        st.metric("🔮 Avg Forecast Demand", f"{avg_forecast:.1f}")
                    with col3:
                        total_forecast = forecast_ts.sum()
                        st.metric("📈 Total Forecast", f"{total_forecast:.0f}")
                    with col4:
                        if avg_historical > 0:
                            change_pct = ((avg_forecast - avg_historical) / avg_historical) * 100
                            st.metric("📊 Change vs Historical", f"{change_pct:+.1f}%")
                    
                    # Enhanced plotting with Plotly
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=historical_ts.index,
                        y=historical_ts.values,
                        mode='lines',
                        name='Historical Demand',
                        line=dict(color='#1f77b4', width=3)
                    ))
                    
                    # Add forecast data
                    fig.add_trace(go.Scatter(
                        x=forecast_ts.index,
                        y=forecast_ts.values,
                        mode='lines',
                        name='Forecasted Demand',
                        line=dict(color='#ff7f0e', width=3, dash='dash')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"📈 {forecast_days}-Day Demand Forecast{title_suffix}",
                        xaxis_title="Date",
                        yaxis_title="Quantity",
                        hovermode='x unified',
                        showlegend=True,
                        height=500,
                        plot_bgcolor='rgba(248,249,250,0.8)',
                        font=dict(size=12)
                    )
                    
                    # Add grid
                    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast data table
                    st.subheader("📅 Detailed Forecast Data")
                    
                    # Create forecast table
                    forecast_df_display = pd.DataFrame({
                        'Date': forecast_ts.index,
                        'Forecasted Quantity': forecast_ts.values
                    })
                    forecast_df_display['Forecasted Quantity'] = forecast_df_display['Forecasted Quantity'].round(1)
                    
                    # Display with formatting
                    st.dataframe(
                        forecast_df_display.style.format({'Forecasted Quantity': '{:.1f}'}),
                        use_container_width=True
                    )
                    
                    # Download forecast data
                    csv_forecast = forecast_df_display.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast Data",
                        data=csv_forecast,
                        file_name=f"demand_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.error("Unable to generate forecast with available data")
                    
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                st.info("💡 Please ensure your data has the required date and quantity columns")

def show_data_explorer(df):
    """Data explorer with search and filtering"""
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

def show_module_status():
    """Display status of imported modules"""
    st.sidebar.subheader("📦 Module Status")
    
    modules_status = [
        ("Contracting Opportunities", CONTRACTING_MODULE),
        ("LOT Size Optimization", LOT_MODULE),
        ("Seasonal Price Optimization", SEASONAL_MODULE),
        ("Anomaly Detection", ANOMALY_MODULE and SKLEARN_AVAILABLE),
        ("Cross-Region Analysis", CROSS_REGION_MODULE),
        ("Reorder Prediction", REORDER_MODULE),
        ("Duplicate Detection", DUPLICATES_MODULE)
    ]
    
    for module_name, status in modules_status:
        if status:
            st.sidebar.markdown(f'<div class="module-status module-available">✅ {module_name}</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f'<div class="module-status module-missing">❌ {module_name}</div>', unsafe_allow_html=True)
    
    # Show dependency status
    st.sidebar.subheader("🔧 Dependencies")
    st.sidebar.markdown(f'<div class="module-status {"module-available" if SKLEARN_AVAILABLE else "module-missing"}">{"✅" if SKLEARN_AVAILABLE else "❌"} Scikit-learn</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    # App title and description
    st.markdown('<div class="main-header">📊 Complete Procurement Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("**Advanced procurement analytics with AI-powered insights, multi-currency support, and demand forecasting**")
    st.markdown("🏛️ **All monetary values displayed in Saudi Riyals (SAR)** | Multi-currency data automatically converted")
    
    # Quick status check
    total_modules = 7
    available_modules_count = sum([CONTRACTING_MODULE, LOT_MODULE, SEASONAL_MODULE, 
                                  ANOMALY_MODULE and SKLEARN_AVAILABLE, CROSS_REGION_MODULE, 
                                  REORDER_MODULE, DUPLICATES_MODULE])
    
    if available_modules_count == total_modules:
        st.success(f"🎉 All {total_modules} analytics modules are ready!")
    elif available_modules_count > 0:
        st.info(f"ℹ️ {available_modules_count}/{total_modules} analytics modules available. Check sidebar for details.")
    else:
        st.warning("⚠️ No advanced analytics modules found. Core features (Dashboard, Forecasting, Explorer) still available.")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Platform Controls")
    
    # Module status
    show_module_status()
    
    # Currency conversion settings
    st.sidebar.subheader("💱 Currency Settings")
    enable_currency = st.sidebar.checkbox("Enable Multi-Currency Conversion to SAR", value=True)
    
    if enable_currency:
        st.sidebar.info("🏛️ All currencies automatically converted to Saudi Riyals (SAR)")
        show_rates = st.sidebar.checkbox("Show Exchange Rates to SAR")
        if show_rates:
            converter = CurrencyConverter()
            st.sidebar.write("**Exchange Rates to SAR:**")
            for curr, rate in list(converter.exchange_rates.items())[:8]:
                st.sidebar.text(f"1 {curr} = {rate} SAR")
    
    # File upload
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload procurement data (CSV)",
        type=['csv'],
        help="Upload a CSV file with procurement/PO data"
    )
    
    # Load data
    if uploaded_file is not None:
        try:
            df = load_and_clean_data(uploaded_file)
            st.sidebar.success("✅ File loaded and processed!")
            st.session_state['data_source'] = uploaded_file.name
            
            # Show processing summary
            with st.sidebar.expander("📊 Processing Summary"):
                st.write(f"✅ **Rows processed:** {len(df):,}")
                st.write(f"✅ **Columns mapped:** {len(df.columns)}")
                if 'currency_conversions' in st.session_state:
                    conv = st.session_state['currency_conversions']
                    if conv > 0:
                        st.write(f"💱 **Currency conversions to SAR:** {conv}")
                        if 'currencies_found' in st.session_state:
                            currencies = st.session_state['currencies_found']
                            st.write(f"💰 **Original currencies:** {', '.join(currencies)}")
                            st.write(f"🏛️ **All values now in:** Saudi Riyals (SAR)")
                    else:
                        st.write("🏛️ **Currency:** Saudi Riyals (SAR)")
                    
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            st.sidebar.info("🔄 Using sample data for demonstration")
            df = load_and_clean_data(generate_sample_data())
            st.session_state['data_source'] = 'Generated Sample Data (Error Recovery)'
    else:
        # Generate sample data for demonstration
        st.sidebar.info("📊 Generating sample multi-currency data for demonstration")
        df = load_and_clean_data(generate_sample_data())
        st.session_state['data_source'] = 'Generated Sample Data'
    
    if df is None or df.empty:
        st.error("❌ No data available. Please upload a CSV file.")
        return
    
    # Apply filters
    st.sidebar.subheader("🔧 Data Filters")
    filtered_df = apply_filters(df)
    
    # Navigation
    st.sidebar.subheader("📑 Analytics Modules")
    
    # Build module list based on available imports
    available_modules = ["📊 Overview Dashboard", "🔮 Demand Forecasting", "🔍 Data Explorer"]
    
    if CONTRACTING_MODULE:
        available_modules.append("🤝 Contracting Opportunities")
    if LOT_MODULE:
        available_modules.append("📦 LOT Size Optimization")
    if SEASONAL_MODULE:
        available_modules.append("🌟 Seasonal Price Optimization")
    if ANOMALY_MODULE and SKLEARN_AVAILABLE:
        available_modules.append("🚨 Anomaly Detection")
    if CROSS_REGION_MODULE:
        available_modules.append("🌍 Cross-Region Analysis")
    if REORDER_MODULE:
        available_modules.append("📈 Reorder Prediction")
    if DUPLICATES_MODULE:
        available_modules.append("🔍 Duplicate Detection")
    
    page = st.sidebar.selectbox("Choose a module:", available_modules)
    
    # Data info sidebar
    st.sidebar.subheader("ℹ️ Data Information")
    st.sidebar.info(f"**Source:** {st.session_state.get('data_source', 'Unknown')}")
    st.sidebar.info(f"**Rows:** {len(filtered_df):,} / {len(df):,}")
    st.sidebar.info(f"**Columns:** {len(df.columns)}")
    
    if 'currency_conversions' in st.session_state:
        conversions = st.session_state['currency_conversions']
        if conversions > 0:
            st.sidebar.success(f"💱 {conversions} values converted to SAR")
        else:
            st.sidebar.info("🏛️ All values in Saudi Riyals (SAR)")
    
    # Data quality check
    if st.sidebar.checkbox("📋 Show Data Quality"):
        required_cols = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.sidebar.warning(f"⚠️ Missing: {', '.join(missing_cols)}")
        else:
            st.sidebar.success("✅ All core columns present")
    
    # Display selected page
    if page == "📊 Overview Dashboard":
        show_overview_dashboard(filtered_df)
    elif page == "🔮 Demand Forecasting":
        show_demand_forecasting(filtered_df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(filtered_df)
    elif page == "🤝 Contracting Opportunities" and CONTRACTING_MODULE:
        safe_module_display(contracting_opportunities, "Contracting Opportunities", filtered_df)
    elif page == "📦 LOT Size Optimization" and LOT_MODULE:
        safe_module_display(lot_size_optimization, "LOT Size Optimization", filtered_df)
    elif page == "🌟 Seasonal Price Optimization" and SEASONAL_MODULE:
        safe_module_display(seasonal_price_optimization, "Seasonal Price Optimization", filtered_df)
    elif page == "🚨 Anomaly Detection" and ANOMALY_MODULE and SKLEARN_AVAILABLE:
        safe_module_display(spend_categorization_anomaly, "Anomaly Detection", filtered_df)
    elif page == "🌍 Cross-Region Analysis" and CROSS_REGION_MODULE:
        safe_module_display(cross_region, "Cross-Region Analysis", filtered_df)
    elif page == "📈 Reorder Prediction" and REORDER_MODULE:
        safe_module_display(reorder_prediction, "Reorder Prediction", filtered_df)
    elif page == "🔍 Duplicate Detection" and DUPLICATES_MODULE:
        safe_module_display(duplicates, "Duplicate Detection", filtered_df)
    else:
        st.error("Selected module is not available. Please check module imports.")
    
    # Footer with helpful information
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("*🚀 Complete Procurement Analytics Platform - Built with Advanced AI & Multi-Currency Support*")
        st.markdown("*🏛️ All monetary values displayed in Saudi Riyals (SAR)*")
    
    with col2:
        if st.button("💡 Installation Help"):
            st.info("""
            **For local installation, create requirements.txt:**
            ```
            streamlit
            pandas
            plotly
            numpy
            scikit-learn
            fuzzywuzzy
            python-levenshtein
            ```
            
            **Then run:**
            ```bash
            pip install -r requirements.txt
            streamlit run app.py
            ```
            """)


if __name__ == "__main__":
    main()
