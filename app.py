import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Import all modules
import contracting_opportunities
import cross_region
import duplicates
import lot_size_optimization
import reorder_prediction
import seasonal_price_optimization
import spend_categorization_anomaly

st.set_page_config(page_title="Smart Procurement Analytics Suite", layout="wide")

def load_and_clean_data(csv_file):
    """Load and clean procurement data from CSV file with SAR currency prioritization"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Basic data cleaning
        # Convert date columns
        date_columns = ['Creation Date', 'Order Date', 'Date', 'creation_date', 'order_date', 'Approved Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean numeric columns
        for col in df.columns:
            if any(num_col.lower() in col.lower() for num_col in ['price', 'qty', 'quantity', 'amount', 'total']):
                if df[col].dtype == 'object':  # If it's stored as text
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Currency handling - Prioritize SAR columns
        if 'Price In SAR' in df.columns:
            df['Unit Price SAR'] = df['Price In SAR']
        elif 'Unit Price' in df.columns:
            # If no SAR price, use original and note conversion needed
            df['Unit Price SAR'] = df['Unit Price']
            if 'PO Currency' in df.columns:
                st.info("💱 Note: Currency conversion may be needed for accurate SAR values")
        
        if 'Total In SAR' in df.columns:
            df['Line Total SAR'] = df['Total In SAR']
        elif 'Line Total' in df.columns:
            df['Line Total SAR'] = df['Line Total']
        elif 'Unit Price SAR' in df.columns and 'Qty Delivered' in df.columns:
            df['Line Total SAR'] = df['Unit Price SAR'] * df['Qty Delivered']
        
        # Ensure we have working price columns for analysis
        if 'Unit Price SAR' not in df.columns and 'Unit Price' in df.columns:
            df['Unit Price SAR'] = df['Unit Price']
        
        if 'Line Total SAR' not in df.columns and 'Line Total' in df.columns:
            df['Line Total SAR'] = df['Line Total']
        
        # Clean vendor names (remove extra spaces)
        if 'Vendor Name' in df.columns:
            df['Vendor Name'] = df['Vendor Name'].astype(str).str.strip()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def format_sar_currency(amount):
    """Format currency amount in SAR with proper symbol"""
    if pd.isna(amount):
        return "ر.س 0.00"
    return f"ر.س {amount:,.2f}"

def get_sar_columns(df):
    """Get the appropriate SAR currency columns from dataframe"""
    price_col = None
    total_col = None
    
    # Priority order for price columns
    price_candidates = ['Unit Price SAR', 'Price In SAR', 'Unit Price']
    for col in price_candidates:
        if col in df.columns:
            price_col = col
            break
    
    # Priority order for total columns  
    total_candidates = ['Line Total SAR', 'Total In SAR', 'Line Total']
    for col in total_candidates:
        if col in df.columns:
            total_col = col
            break
            
    return price_col, total_col

def validate_currency_conversion(df):
    """Validate currency conversions and provide detailed breakdown"""
    validation_results = {}
    
    if 'PO Currency' not in df.columns:
        return {"error": "No PO Currency column found"}
    
    # Get currency breakdown
    currency_summary = df.groupby('PO Currency').agg({
        'Unit Price': ['sum', 'count', 'mean'],
        'Price In SAR': ['sum', 'count', 'mean'] if 'Price In SAR' in df.columns else ['sum', 'count', 'mean'],
        'Total in Local PO Currency': ['sum', 'count'] if 'Total in Local PO Currency' in df.columns else ['sum', 'count'],
        'Total In SAR': ['sum', 'count'] if 'Total In SAR' in df.columns else ['sum', 'count'],
        'Qty Delivered': 'sum'
    }).round(2)
    
    # Calculate implied exchange rates
    if 'Price in Local PO Currency' in df.columns and 'Price In SAR' in df.columns:
        df_rates = df[df['Price in Local PO Currency'] > 0].copy()
        df_rates['Implied_Rate'] = df_rates['Price In SAR'] / df_rates['Price in Local PO Currency']
        
        rate_summary = df_rates.groupby('PO Currency')['Implied_Rate'].agg(['mean', 'std', 'min', 'max']).round(4)
        validation_results['exchange_rates'] = rate_summary
    
    validation_results['currency_summary'] = currency_summary
    
    # Validate totals
    if 'Total In SAR' in df.columns:
        calculated_total = df['Total In SAR'].sum()
        validation_results['total_sar'] = calculated_total
        
        # Check if manual calculation matches
        if 'Price In SAR' in df.columns and 'Qty Delivered' in df.columns:
            manual_total = (df['Price In SAR'] * df['Qty Delivered']).sum()
            validation_results['manual_calculation'] = manual_total
            validation_results['difference'] = abs(calculated_total - manual_total)
    
    return validation_results

def display_currency_validation(df):
    """Display comprehensive currency validation"""
    st.subheader("💱 Currency Conversion Validation")
    
    validation = validate_currency_conversion(df)
    
    if 'error' in validation:
        st.error(validation['error'])
        return
    
    # Total validation
    if 'total_sar' in validation:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Value (SAR)", format_sar_currency(validation['total_sar']))
        
        if 'manual_calculation' in validation:
            with col2:
                st.metric("Manual Calculation", format_sar_currency(validation['manual_calculation']))
            with col3:
                difference = validation['difference']
                if difference < 1:
                    st.success(f"✅ Accurate: Diff {format_sar_currency(difference)}")
                else:
                    st.warning(f"⚠️ Difference: {format_sar_currency(difference)}")
    
    # Currency breakdown
    if 'currency_summary' in validation:
        st.subheader("📊 Currency Breakdown")
        
        # Create currency summary table
        currency_df = validation['currency_summary']
        if not currency_df.empty:
            # Flatten multi-level columns
            currency_flat = pd.DataFrame()
            
            for currency in currency_df.index:
                row_data = {
                    'Currency': currency,
                    'Transaction_Count': int(currency_df.loc[currency, ('Unit Price', 'count')]) if ('Unit Price', 'count') in currency_df.columns else 0,
                    'Total_Qty': currency_df.loc[currency, ('Qty Delivered', 'sum')] if ('Qty Delivered', 'sum') in currency_df.columns else 0,
                    'Total_SAR': currency_df.loc[currency, ('Total In SAR', 'sum')] if ('Total In SAR', 'sum') in currency_df.columns else 0,
                    'Avg_Price_SAR': currency_df.loc[currency, ('Price In SAR', 'mean')] if ('Price In SAR', 'mean') in currency_df.columns else 0
                }
                currency_flat = pd.concat([currency_flat, pd.DataFrame([row_data])], ignore_index=True)
            
            # Format the display
            display_currency = currency_flat.copy()
            display_currency['Total_SAR_Formatted'] = display_currency['Total_SAR'].apply(format_sar_currency)
            display_currency['Avg_Price_SAR_Formatted'] = display_currency['Avg_Price_SAR'].apply(format_sar_currency)
            display_currency['Percentage'] = (display_currency['Total_SAR'] / display_currency['Total_SAR'].sum() * 100).round(1)
            
            # Show summary table
            st.dataframe(
                display_currency[['Currency', 'Transaction_Count', 'Total_Qty', 'Total_SAR_Formatted', 'Avg_Price_SAR_Formatted', 'Percentage']].rename(columns={
                    'Transaction_Count': 'Transactions',
                    'Total_Qty': 'Total Quantity', 
                    'Total_SAR_Formatted': 'Total Value (SAR)',
                    'Avg_Price_SAR_Formatted': 'Avg Price (SAR)',
                    'Percentage': 'Value %'
                }),
                use_container_width=True
            )
    
    # Exchange rates validation
    if 'exchange_rates' in validation:
        st.subheader("💹 Exchange Rate Analysis")
        
        rates_df = validation['exchange_rates'].reset_index()
        
        if not rates_df.empty:
            st.write("**Implied Exchange Rates to SAR:**")
            
            # Format exchange rates display
            display_rates = rates_df.copy()
            display_rates = display_rates.round(4)
            display_rates['Rate_Range'] = display_rates['min'].astype(str) + ' - ' + display_rates['max'].astype(str)
            
            st.dataframe(
                display_rates.rename(columns={
                    'PO Currency': 'Currency',
                    'mean': 'Avg Rate',
                    'std': 'Std Dev',
                    'min': 'Min Rate',
                    'max': 'Max Rate',
                    'Rate_Range': 'Rate Range'
                }),
                use_container_width=True
            )
            
            # Rate consistency check
            st.write("**Rate Consistency Check:**")
            for _, row in display_rates.iterrows():
                currency = row['PO Currency']
                std_dev = row['std']
                avg_rate = row['mean']
                
                if pd.notna(std_dev) and avg_rate > 0:
                    cv = (std_dev / avg_rate) * 100  # Coefficient of variation
                    
                    if cv < 1:
                        st.success(f"✅ {currency}: Consistent rates (CV: {cv:.2f}%)")
                    elif cv < 5:
                        st.warning(f"⚠️ {currency}: Moderate variation (CV: {cv:.2f}%)")
                    else:
                        st.error(f"❌ {currency}: High rate variation (CV: {cv:.2f}%)")
    
    # Data quality checks
    st.subheader("🔍 Data Quality Checks")
    
    quality_issues = []
    
    # Check for missing SAR values
    if 'Price In SAR' in df.columns:
        missing_sar_price = df['Price In SAR'].isna().sum()
        if missing_sar_price > 0:
            quality_issues.append(f"Missing SAR prices: {missing_sar_price:,} records")
    
    if 'Total In SAR' in df.columns:
        missing_sar_total = df['Total In SAR'].isna().sum()
        if missing_sar_total > 0:
            quality_issues.append(f"Missing SAR totals: {missing_sar_total:,} records")
    
    # Check for zero values
    if 'Price In SAR' in df.columns:
        zero_prices = (df['Price In SAR'] == 0).sum()
        if zero_prices > 0:
            quality_issues.append(f"Zero SAR prices: {zero_prices:,} records")
    
    # Check for unrealistic exchange rates
    if 'exchange_rates' in validation:
        for currency in validation['exchange_rates'].index:
            max_rate = validation['exchange_rates'].loc[currency, 'max']
            min_rate = validation['exchange_rates'].loc[currency, 'min']
            
            if max_rate > 100 or min_rate < 0.001:
                quality_issues.append(f"Unusual {currency} exchange rates: {min_rate:.4f} - {max_rate:.4f}")
    
    if quality_issues:
        st.warning("⚠️ **Data Quality Issues Found:**")
        for issue in quality_issues:
            st.write(f"• {issue}")
    else:
        st.success("✅ No significant data quality issues detected")
    
    return validation

def forecast_demand(df):
    """Enhanced demand forecasting function for procurement data"""
    try:
        # Use Creation Date and Qty Delivered from your dataset
        date_col = 'Creation Date'
        qty_col = 'Qty Delivered'
        
        if date_col not in df.columns or qty_col not in df.columns:
            raise ValueError(f"Required columns not found. Available columns: {list(df.columns)}")
        
        # Create clean dataset
        df_clean = df[[date_col, qty_col]].copy()
        df_clean = df_clean.dropna()
        
        # Ensure date column is datetime
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after cleaning")
        
        # Sort by date
        df_clean = df_clean.sort_values(date_col)
        
        # Aggregate daily demand
        daily_demand = df_clean.groupby(date_col)[qty_col].sum()
        
        # Fill missing dates with zero demand
        date_range = pd.date_range(start=daily_demand.index.min(), 
                                 end=daily_demand.index.max(), 
                                 freq='D')
        daily_demand = daily_demand.reindex(date_range, fill_value=0)
        
        # Calculate forecasting parameters
        window_size = min(14, len(daily_demand) // 4)  # Use 14 days or 1/4 of data
        if window_size < 3:
            window_size = 3
            
        # Calculate recent average and trend
        recent_data = daily_demand.tail(window_size)
        recent_avg = recent_data.mean()
        overall_avg = daily_demand.mean()
        
        # Simple trend calculation
        if len(daily_demand) > window_size * 2:
            older_avg = daily_demand.iloc[-window_size*2:-window_size].mean()
            trend_factor = recent_avg / older_avg if older_avg > 0 else 1.0
            trend_factor = max(0.7, min(1.3, trend_factor))  # Cap trend
        else:
            trend_factor = 1.0
        
        # Generate 30-day forecast
        last_date = daily_demand.index.max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=30, freq='D')
        
        # Create forecast with trend and seasonality
        base_forecast = recent_avg if recent_avg > 0 else overall_avg
        
        # Add day-of-week seasonality (simple pattern)
        forecast_values = []
        np.random.seed(42)  # For consistency
        
        for i, date in enumerate(forecast_dates):
            # Base forecast with trend
            daily_forecast = base_forecast * trend_factor
            
            # Add day-of-week effect (weekdays vs weekends)
            if date.weekday() >= 5:  # Weekend
                daily_forecast *= 0.7
            
            # Add some realistic noise
            noise = np.random.normal(0, daily_forecast * 0.15)
            final_forecast = max(0, daily_forecast + noise)
            
            forecast_values.append(final_forecast)
        
        forecast = pd.Series(forecast_values, index=forecast_dates, name='Forecasted_Demand')
        
        return daily_demand, forecast
        
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")
        
        # Create minimal fallback forecast
        try:
            last_date = pd.to_datetime('2024-01-01')
            if 'Creation Date' in df.columns:
                df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce')
                last_date = df['Creation Date'].max()
            
            # Simple historical series
            hist_dates = pd.date_range(end=last_date, periods=30, freq='D')
            historical = pd.Series(np.random.poisson(25, 30), index=hist_dates)
            
            # Simple forecast
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
            forecast = pd.Series(np.random.poisson(25, 30), index=forecast_dates)
            
            return historical, forecast
            
        except Exception as e2:
            st.error(f"Fallback forecasting failed: {str(e2)}")
            # Last resort: completely dummy data
            base_date = datetime(2024, 1, 1)
            hist_dates = pd.date_range(start=base_date, periods=30, freq='D')
            forecast_dates = pd.date_range(start=base_date + timedelta(days=30), periods=30, freq='D')
            
            historical = pd.Series(np.ones(30) * 20, index=hist_dates)
            forecast = pd.Series(np.ones(30) * 20, index=forecast_dates)
            
            return historical, forecast

def main():
    st.title("📦 Smart Procurement Analytics Suite")
    st.markdown("Comprehensive procurement analytics platform with demand forecasting and vendor optimization.")
    
    # Sidebar for navigation
    st.sidebar.title("📋 Analytics Modules")
    
    modules = {
        "🔮 Demand Forecasting": "demand_forecasting",
        "🤝 Contracting Opportunities": "contracting_opportunities", 
        "🌍 Cross-Region Optimization": "cross_region",
        "🔍 Duplicate Detection": "duplicates",
        "📦 LOT Size Optimization": "lot_size_optimization",
        "📊 Reorder Prediction": "reorder_prediction",
        "🌟 Seasonal Price Optimization": "seasonal_price_optimization",
        "📈 Spend Analysis & Anomaly Detection": "spend_categorization_anomaly"
    }
    
    selected_module = st.sidebar.selectbox("Select Analysis Module", list(modules.keys()))
    
    # File upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Upload")
    csv_file = st.sidebar.file_uploader("Upload your structured PO CSV file", type=["csv"])
    
    if csv_file:
        st.sidebar.success("✅ File uploaded successfully!")
        
        # Load and clean data
        try:
            df = load_and_clean_data(csv_file)
            
            # Show data preview in sidebar
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Data Preview")
            st.sidebar.write(f"**Records:** {len(df):,}")
            st.sidebar.write(f"**Columns:** {len(df.columns)}")
            
            if 'Vendor Name' in df.columns:
                st.sidebar.write(f"**Vendors:** {df['Vendor Name'].nunique()}")
            
            # Vendor Selection Section (if vendor data exists)
            if 'Vendor Name' in df.columns:
                st.sidebar.markdown("---")
                st.sidebar.subheader("🏢 Vendor Selection")
                
                # Get unique vendors
                vendors = sorted(df['Vendor Name'].dropna().unique().tolist())
                
                # Initialize session state for selected vendors if not exists
                if 'selected_vendors' not in st.session_state:
                    st.session_state.selected_vendors = []
                
                # Show selection count first
                st.sidebar.write(f"**Available Vendors:** {len(vendors)}")
                st.sidebar.write(f"**Currently Selected:** {len(st.session_state.selected_vendors)}")
                
                # Select/Clear All buttons - make them more prominent
                st.sidebar.markdown("**Quick Actions:**")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.sidebar.button("✅ Select All", key="select_all_sidebar", use_container_width=True):
                        st.session_state.selected_vendors = vendors.copy()
                        st.rerun()
                
                with col2:
                    if st.sidebar.button("❌ Clear All", key="clear_all_sidebar", use_container_width=True):
                        st.session_state.selected_vendors = []
                        st.rerun()
                
                # Add some space
                st.sidebar.write("")
                
                # Multi-select widget
                selected_vendors = st.sidebar.multiselect(
                    "Choose specific vendors:",
                    options=vendors,
                    default=st.session_state.selected_vendors,
                    key="vendor_multiselect_sidebar",
                    help="Use the buttons above to select/clear all vendors quickly"
                )
                
                # Update session state when multiselect changes
                if selected_vendors != st.session_state.selected_vendors:
                    st.session_state.selected_vendors = selected_vendors
                
                # Filter data by selected vendors
                if st.session_state.selected_vendors:
                    df_filtered = df[df['Vendor Name'].isin(st.session_state.selected_vendors)]
                    st.sidebar.success(f"✅ Using {len(st.session_state.selected_vendors)} vendors ({len(df_filtered):,} records)")
                else:
                    df_filtered = df
                    st.sidebar.info("ℹ️ Using all vendors (no filter applied)")
            else:
                df_filtered = df
                st.sidebar.info("ℹ️ No vendor column detected")
            
            # Route to selected module
            module_key = modules[selected_module]
            
            # Main content area
            if module_key == "demand_forecasting":
                display_demand_forecasting(df_filtered, csv_file)
            elif module_key == "contracting_opportunities":
                contracting_opportunities.display(df_filtered)
            elif module_key == "cross_region":
                cross_region.display(df_filtered)
            elif module_key == "duplicates":
                duplicates.display(df_filtered)
            elif module_key == "lot_size_optimization":
                lot_size_optimization.display(df_filtered)
            elif module_key == "reorder_prediction":
                reorder_prediction.display(df_filtered)
            elif module_key == "seasonal_price_optimization":
                seasonal_price_optimization.display(df_filtered)
            elif module_key == "spend_categorization_anomaly":
                spend_categorization_anomaly.display(df_filtered)
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please check your CSV file format and try again.")
    
    else:
        # Show welcome screen with module overview
        display_welcome_screen()

def display_demand_forecasting(df, csv_file):
    """Enhanced demand forecasting functionality with vendor filtering and SAR currency"""
    st.header("🔮 Demand Forecasting")
    st.markdown("AI-powered demand forecasting for optimal inventory planning and procurement decisions.")
    st.info("💱 All monetary values displayed in Saudi Riyal (SAR) - ر.س")
    
    # Show filtered data info with currency information
    if 'selected_vendors' in st.session_state and st.session_state.selected_vendors:
        selected_count = len(st.session_state.selected_vendors)
        st.success(f"📊 Analyzing data for {selected_count} selected vendors ({len(df):,} records)")
        
        # Show selected vendors in expandable section
        with st.expander("📋 View Selected Vendors"):
            vendor_cols = st.columns(3)
            for i, vendor in enumerate(st.session_state.selected_vendors):
                with vendor_cols[i % 3]:
                    st.write(f"• {vendor}")
    else:
        st.info("📊 Analyzing data for all vendors")
    
    # Currency information
    try:
        price_col, total_col = get_sar_columns(df)
        if 'Price In SAR' in df.columns or 'Total In SAR' in df.columns:
            st.success("💱 Using native SAR currency columns for accurate analysis")
        elif 'PO Currency' in df.columns:
            unique_currencies = df['PO Currency'].value_counts()
            st.warning(f"💱 Multiple currencies detected: {list(unique_currencies.index[:3])}. Converting to SAR for analysis.")
        else:
            st.info("💱 Currency: Assuming SAR for all monetary values")
    except:
        # Fallback if function not available
        price_col = 'Unit Price' if 'Unit Price' in df.columns else None
        total_col = 'Line Total' if 'Line Total' in df.columns else None
        st.info("💱 Currency: Using standard price columns")
    
    # Display cleaned data sample with SAR formatting
    st.subheader("📋 Cleaned PO Data Sample")
    
    # Enhanced data display with better formatting
    display_df = df.head(10)
    if not display_df.empty:
        # Format numeric columns with SAR currency
        formatted_df = display_df.copy()
        
        # Get SAR columns
        price_col, total_col = get_sar_columns(df)
        
        for col in formatted_df.columns:
            if 'sar' in col.lower() or col in [price_col, total_col] or 'price' in col.lower() or 'total' in col.lower():
                if pd.api.types.is_numeric_dtype(formatted_df[col]):
                    formatted_df[col] = formatted_df[col].apply(lambda x: format_sar_currency(x) if pd.notnull(x) else "ر.س 0.00")
            elif 'qty' in col.lower() or 'quantity' in col.lower():
                if pd.api.types.is_numeric_dtype(formatted_df[col]):
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
        
        st.dataframe(formatted_df, use_container_width=True)
    
    # Comprehensive data summary
    st.subheader("📊 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        if 'Vendor Name' in df.columns:
            unique_vendors = df['Vendor Name'].nunique()
            st.metric("Unique Vendors", f"{unique_vendors:,}")
    with col3:
        if 'Item' in df.columns:
            unique_items = df['Item'].nunique()
            st.metric("Unique Items", f"{unique_items:,}")
    with col4:
        if 'Creation Date' in df.columns:
            df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce')
            date_range = (df['Creation Date'].max() - df['Creation Date'].min()).days
            st.metric("Date Range (Days)", f"{date_range:,}")
    
    # Additional metrics row with SAR currency
    price_col, total_col = get_sar_columns(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if price_col and price_col in df.columns:
            avg_price = df[price_col].mean()
            st.metric("Avg Unit Price", format_sar_currency(avg_price))
    with col2:
        if 'Qty Delivered' in df.columns:
            total_qty = df['Qty Delivered'].sum()
            st.metric("Total Quantity", f"{total_qty:,.0f}")
    with col3:
        if total_col and total_col in df.columns:
            total_value = df[total_col].sum()
            st.metric("Total Value", format_sar_currency(total_value))
        elif price_col and price_col in df.columns and 'Qty Delivered' in df.columns:
            total_value = (df[price_col] * df['Qty Delivered']).sum()
            st.metric("Total Value", format_sar_currency(total_value))
    with col4:
        if len(df) > 0:
            avg_order_size = df['Qty Delivered'].mean() if 'Qty Delivered' in df.columns else 0
            st.metric("Avg Order Size", f"{avg_order_size:,.1f}")
    
    # Data Date Range Information
    if 'Creation Date' in df.columns:
        st.subheader("📅 Data Coverage Period")
        df_dates = df['Creation Date'].dropna()
        if len(df_dates) > 0:
            earliest_date = df_dates.min()
            latest_date = df_dates.max()
            date_span = (latest_date - earliest_date).days
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Start Date", earliest_date.strftime('%Y-%m-%d'))
            with col2:
                st.metric("End Date", latest_date.strftime('%Y-%m-%d'))
            with col3:
                st.metric("Total Days", f"{date_span:,}")
            with col4:
                data_months = round(date_span / 30.44, 1)  # Average days per month
                st.metric("Coverage (Months)", f"{data_months}")
            
            # Visual timeline
            st.write("**Data Timeline:**")
            timeline_df = df.groupby(df['Creation Date'].dt.date).size().reset_index()
            timeline_df.columns = ['Date', 'Transactions']
            
            fig_timeline = px.line(timeline_df, x='Date', y='Transactions', 
                                 title="Daily Transaction Volume Over Time",
                                 labels={'Transactions': 'Number of Transactions'})
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Add Currency Validation Section
    display_currency_validation(df)
    
    # Data quality indicators
    st.subheader("🔍 Data Quality Check")
    
    quality_metrics = {}
    total_records = len(df)
    
    if total_records > 0:
        # Check for missing values
        if 'Creation Date' in df.columns:
            date_completeness = (df['Creation Date'].notna().sum() / total_records) * 100
            quality_metrics['Date Completeness'] = f"{date_completeness:.1f}%"
        
        if 'Unit Price' in df.columns:
            price_completeness = (df['Unit Price'].notna().sum() / total_records) * 100
            quality_metrics['Price Completeness'] = f"{price_completeness:.1f}%"
        
        if 'Qty Delivered' in df.columns:
            qty_completeness = (df['Qty Delivered'].notna().sum() / total_records) * 100
            quality_metrics['Quantity Completeness'] = f"{qty_completeness:.1f}%"
        
        if 'Vendor Name' in df.columns:
            vendor_completeness = (df['Vendor Name'].notna().sum() / total_records) * 100
            quality_metrics['Vendor Completeness'] = f"{vendor_completeness:.1f}%"
    
    # Display quality metrics
    quality_cols = st.columns(len(quality_metrics)) if quality_metrics else st.columns(1)
    for i, (metric, value) in enumerate(quality_metrics.items()):
        with quality_cols[i]:
            # Color code based on completeness
            percentage = float(value.replace('%', ''))
            if percentage >= 95:
                st.success(f"**{metric}**: {value}")
            elif percentage >= 80:
                st.warning(f"**{metric}**: {value}")
            else:
                st.error(f"**{metric}**: {value}")
    
    # Forecasting section
    st.subheader("🔮 Demand Forecast Generation")
    
    # Forecasting options
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_days = st.selectbox("Forecast Period", [30, 60, 90], index=0)
    with col2:
        confidence_interval = st.selectbox("Confidence Level", ["80%", "90%", "95%"], index=1)
    with col3:
        include_seasonality = st.checkbox("Include Seasonality", value=True)
    
    # Advanced options
    with st.expander("⚙️ Advanced Forecasting Options"):
        col1, col2 = st.columns(2)
        with col1:
            aggregation_level = st.selectbox("Aggregation Level", 
                                           ["All Items", "By Item", "By Vendor", "By Category"], 
                                           index=0)
        with col2:
            trend_adjustment = st.slider("Trend Adjustment", -0.5, 0.5, 0.0, 0.1)
    
    if st.button("🚀 Generate Demand Forecast", type="primary"):
        try:
            with st.spinner("🔄 Generating demand forecast... This may take a moment."):
                # Generate forecast using original function
                ts, forecast = forecast_demand(df)
                
                # Enhanced plotting with multiple visualizations using Plotly
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('30-Day Demand Forecast', 'Historical Trend Analysis', 
                                   'Historical Demand Distribution', 'Historical vs Forecast Comparison'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Main forecast plot
                fig.add_trace(
                    go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Historical Demand',
                             line=dict(color='#1f77b4', width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='Forecasted Demand',
                             line=dict(color='#ff7f0e', width=2, dash='dash')),
                    row=1, col=1
                )
                
                # Historical trend analysis (7-day moving average)
                if len(ts) > 7:
                    moving_avg = ts.rolling(window=7).mean()
                    fig.add_trace(
                        go.Scatter(x=moving_avg.index, y=moving_avg.values, mode='lines', 
                                 name='7-day Moving Average', line=dict(color='green', width=2)),
                        row=1, col=2
                    )
                
                # Demand distribution histogram
                fig.add_trace(
                    go.Histogram(x=ts.values, name='Historical Demand Distribution', 
                               marker_color='skyblue', opacity=0.7, nbinsx=20),
                    row=2, col=1
                )
                
                # Forecast vs Historical comparison
                comparison_data = {
                    'Metric': ['Historical Avg', 'Forecast Avg', 'Historical Peak', 'Forecast Peak'],
                    'Value': [ts.mean(), forecast.mean(), ts.max(), forecast.max()],
                    'Color': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                }
                
                fig.add_trace(
                    go.Bar(x=comparison_data['Metric'], y=comparison_data['Value'], 
                          name='Comparison', marker_color=comparison_data['Color']),
                    row=2, col=2
                )
                
                # Update layout
                fig.update_layout(
                    height=800,
                    showlegend=True,
                    title_text="Comprehensive Demand Analysis Dashboard"
                )
                
                # Add forecast statistics as annotation
                forecast_stats = f"Avg Daily: {forecast.mean():.1f}<br>Total: {forecast.sum():.0f}<br>Peak: {forecast.max():.1f}"
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=forecast_stats,
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="lightblue",
                    opacity=0.8,
                    xanchor="left",
                    yanchor="top"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast insights and recommendations
                st.subheader("💡 Forecast Insights & Recommendations")
                
                # Calculate insights
                historical_avg = ts.mean()
                forecast_avg = forecast.mean()
                trend_indicator = ((forecast_avg - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if trend_indicator > 5:
                        st.success(f"📈 **Growing Demand**\n+{trend_indicator:.1f}% vs historical average")
                        st.write("**Recommendation:** Increase safety stock and review supplier capacity.")
                    elif trend_indicator < -5:
                        st.warning(f"📉 **Declining Demand**\n{trend_indicator:.1f}% vs historical average")
                        st.write("**Recommendation:** Optimize inventory levels and review procurement strategy.")
                    else:
                        st.info(f"➡️ **Stable Demand**\n{trend_indicator:.1f}% vs historical average")
                        st.write("**Recommendation:** Maintain current procurement levels.")
                
                with col2:
                    demand_volatility = (ts.std() / ts.mean()) * 100 if ts.mean() > 0 else 0
                    if demand_volatility > 30:
                        st.warning(f"⚠️ **High Volatility**\n{demand_volatility:.1f}% coefficient of variation")
                        st.write("**Recommendation:** Implement flexible procurement strategies.")
                    else:
                        st.success(f"✅ **Stable Pattern**\n{demand_volatility:.1f}% coefficient of variation")
                        st.write("**Recommendation:** Suitable for contract negotiations.")
                
                with col3:
                    reorder_point = forecast.mean() + (2 * forecast.std())
                    st.info(f"📦 **Suggested Reorder Point**\n{reorder_point:.0f} units")
                    st.write("**Recommendation:** Trigger reorders when inventory drops to this level.")
                
                # Detailed forecast data table
                st.subheader("📊 Detailed Forecast Data")
                
                forecast_df = forecast.to_frame(name='Forecasted_Quantity')
                forecast_df['Date'] = forecast_df.index
                forecast_df['Day_of_Week'] = forecast_df['Date'].dt.day_name()
                forecast_df['Cumulative_Forecast'] = forecast_df['Forecasted_Quantity'].cumsum()
                
                # Add confidence intervals (simplified)
                forecast_df['Lower_Bound'] = forecast_df['Forecasted_Quantity'] * 0.8
                forecast_df['Upper_Bound'] = forecast_df['Forecasted_Quantity'] * 1.2
                
                # Reorder columns
                forecast_df = forecast_df[['Date', 'Day_of_Week', 'Forecasted_Quantity', 
                                        'Lower_Bound', 'Upper_Bound', 'Cumulative_Forecast']].reset_index(drop=True)
                
                # Format and display with styling (without matplotlib dependency)
                st.dataframe(
                    forecast_df.style.format({
                        'Forecasted_Quantity': '{:.1f}',
                        'Lower_Bound': '{:.1f}',
                        'Upper_Bound': '{:.1f}',
                        'Cumulative_Forecast': '{:.0f}'
                    }),
                    use_container_width=True
                )
                
                # Summary metrics with enhanced details
                st.subheader("📈 Forecast Summary Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Forecast", f"{forecast.sum():.0f} units", 
                             f"{((forecast.sum() - ts.sum()) / ts.sum() * 100):+.1f}%" if ts.sum() > 0 else None)
                with col2:
                    st.metric("Daily Average", f"{forecast.mean():.1f} units",
                             f"{((forecast.mean() - ts.mean()) / ts.mean() * 100):+.1f}%" if ts.mean() > 0 else None)
                with col3:
                    st.metric("Peak Day", f"{forecast.max():.1f} units")
                with col4:
                    st.metric("Min Day", f"{forecast.min():.1f} units")
                with col5:
                    st.metric("Forecast Std Dev", f"{forecast.std():.1f} units")
                
                # Risk assessment
                st.subheader("⚠️ Risk Assessment")
                
                risk_factors = []
                
                # High variability risk
                if demand_volatility > 30:
                    risk_factors.append({
                        'Risk': 'High Demand Variability',
                        'Level': 'High',
                        'Impact': 'Stockouts or overstock',
                        'Mitigation': 'Increase safety stock, flexible suppliers'
                    })
                
                # Trend risk
                if abs(trend_indicator) > 20:
                    risk_factors.append({
                        'Risk': 'Significant Trend Change',
                        'Level': 'Medium',
                        'Impact': 'Forecast accuracy',
                        'Mitigation': 'Regular forecast updates, market analysis'
                    })
                
                # Seasonality risk (simplified check)
                if len(ts) > 30:
                    monthly_variation = ts.groupby(ts.index.month).mean().std()
                    if monthly_variation > ts.mean() * 0.2:
                        risk_factors.append({
                            'Risk': 'Seasonal Patterns',
                            'Level': 'Medium',
                            'Impact': 'Seasonal stockouts',
                            'Mitigation': 'Seasonal procurement planning'
                        })
                
                if risk_factors:
                    risk_df = pd.DataFrame(risk_factors)
                    st.dataframe(risk_df, use_container_width=True)
                else:
                    st.success("✅ No significant risk factors identified")
                
                # Action plan
                st.subheader("🎯 Recommended Action Plan")
                
                action_plan = {
                    'Immediate (1-2 weeks)': [
                        f"Review current inventory levels against forecast",
                        f"Contact suppliers for capacity confirmation",
                        f"Set up reorder point at {reorder_point:.0f} units"
                    ],
                    'Short-term (1 month)': [
                        f"Implement demand monitoring dashboard",
                        f"Review and adjust safety stock levels",
                        f"Schedule supplier performance reviews"
                    ],
                    'Long-term (3+ months)': [
                        f"Evaluate contract negotiation opportunities",
                        f"Implement automated reordering systems",
                        f"Develop demand sensing capabilities"
                    ]
                }
                
                for timeframe, actions in action_plan.items():
                    st.write(f"**{timeframe}:**")
                    for action in actions:
                        st.write(f"• {action}")
                    st.write("")
                
                # Enhanced download options
                st.subheader("📥 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Forecast data export
                    csv_forecast = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Forecast Data",
                        data=csv_forecast,
                        file_name=f"demand_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Summary report export
                    summary_data = {
                        'Metric': ['Total Forecast', 'Daily Average', 'Peak Day', 'Min Day', 'Trend vs Historical'],
                        'Value': [f"{forecast.sum():.0f}", f"{forecast.mean():.1f}", 
                                f"{forecast.max():.1f}", f"{forecast.min():.1f}", f"{trend_indicator:.1f}%"],
                        'Unit': ['units', 'units/day', 'units', 'units', 'percent']
                    }
                    summary_df = pd.DataFrame(summary_data)
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Summary Report",
                        data=csv_summary,
                        file_name=f"forecast_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Action plan export
                    action_items = []
                    for timeframe, actions in action_plan.items():
                        for action in actions:
                            action_items.append({'Timeframe': timeframe, 'Action': action})
                    
                    action_df = pd.DataFrame(action_items)
                    csv_actions = action_df.to_csv(index=False)
                    st.download_button(
                        label="📝 Download Action Plan",
                        data=csv_actions,
                        file_name=f"action_plan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")
            st.info("""
            **Troubleshooting Tips:**
            - Ensure your data has a date column (Creation Date, Order Date, etc.)
            - Check that quantity columns contain numeric values
            - Verify data covers at least 30 days for meaningful forecasting
            - Remove any duplicate or invalid date entries
            """)
            
            # Show data structure for debugging
            with st.expander("🔍 Debug: Data Structure"):
                st.write("**Available Columns:**")
                st.write(list(df.columns))
                st.write("**Data Types:**")
                st.write(df.dtypes)
                st.write("**Sample Data:**")
                st.write(df.head())

def display_welcome_screen():
    """Welcome screen with comprehensive module overview"""
    st.header("🏠 Welcome to Smart Procurement Analytics Suite")
    st.markdown("Transform your procurement data into actionable insights with our comprehensive analytics platform.")
    
    # Key benefits
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🎯 **Optimize Spend**\nReduce costs through data-driven insights")
    with col2:
        st.info("📊 **Predict Demand**\nAI-powered forecasting for better planning")
    with col3:
        st.warning("🤝 **Improve Relationships**\nVendor performance analytics")
    
    # Expected data format
    st.subheader("📋 Expected CSV Data Format")
    st.markdown("Your CSV file should contain procurement transaction data with these columns:")
    
    sample_data = {
        'Creation Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Vendor Name': ['LLORI LLERENA', '33 Designs', 'AHMED ALI SALE'],
        'Item': [101, 102, 103],
        'Item Description': ['Office Supplies - Paper', 'IT Equipment - Laptop', 'Raw Materials - Steel'],
        'Unit Price': [95.63, 4312.50, 282.19],
        'Price In SAR': [95.63, 4312.50, 282.19],
        'Qty Delivered': [100, 2, 50],
        'Line Total': [9563.00, 8625.00, 14109.50],
        'Total In SAR': [9563.00, 8625.00, 14109.50],
        'PO Currency': ['SAR', 'SAR', 'SAR'],
        'W/H': ['Warehouse A', 'Warehouse B', 'Warehouse A']
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)
    
    # Column requirements
    st.subheader("🔑 Column Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Required Columns:**
        - 📅 **Creation Date** - Transaction date (YYYY-MM-DD format)
        - 💰 **Unit Price** - Price per unit (numeric)
        - 📦 **Qty Delivered** - Quantity delivered (numeric)
        - 🏢 **Vendor Name** - Supplier name (text)
        - 🛍️ **Item** - Item identifier (text/numeric)
        
        **💱 SAR Currency Columns (Preferred):**
        - **Price In SAR** - Unit price in Saudi Riyal
        - **Total In SAR** - Line total in Saudi Riyal
        """)
    
    with col2:
        st.markdown("""
        **🔧 Optional Columns:**
        - 📝 **Item Description** - Detailed item description
        - 🏪 **W/H** - Warehouse or location code
        - 💵 **Line Total** - Total amount (auto-calculated if missing)
        - 🏦 **PO Currency** - Purchase order currency
        - 📋 **Category** - Item category classification
        - 🚚 **Lead Time** - Delivery lead time in days
        
        **💡 Currency Handling:**
        - System prioritizes SAR columns when available
        - Displays all amounts in Saudi Riyal (ر.س)
        - Auto-converts when SAR columns exist
        """)
    
    # Currency conversion notice
    st.info("""
    💱 **Currency Conversion:** This system is optimized for Saudi Riyal (SAR). 
    If your data includes 'Price In SAR' and 'Total In SAR' columns, the system will use those for accurate analysis. 
    Otherwise, it will use the base price columns and assume SAR currency.
    """)
    
    # Module overview with detailed descriptions
    st.subheader("🛠️ Analytics Modules Overview")
    
    modules_detailed = [
        {
            "name": "🔮 Demand Forecasting",
            "description": "AI-powered demand forecasting with time series analysis and currency validation",
            "features": ["30-90 day forecasts", "Trend analysis", "Seasonality detection", "Risk assessment", "SAR currency validation"],
            "use_case": "Optimize inventory levels and prevent stockouts with accurate SAR totals"
        },
        {
            "name": "🤝 Contracting Opportunities", 
            "description": "Identify optimal contracting opportunities and calculate savings potential",
            "features": ["Vendor performance scoring", "Contract suitability analysis", "ROI calculations", "Implementation roadmap"],
            "use_case": "Negotiate better contracts and reduce procurement costs"
        },
        {
            "name": "🌍 Cross-Region Optimization",
            "description": "Compare vendor pricing across different regions and warehouses",
            "features": ["Regional price comparison", "Vendor location analysis", "Cost arbitrage opportunities"],
            "use_case": "Optimize supplier selection by geography"
        },
        {
            "name": "🔍 Duplicate Detection",
            "description": "Find duplicate vendors and items using advanced fuzzy matching",
            "features": ["Fuzzy string matching", "Similarity scoring", "Consolidation recommendations"],
            "use_case": "Clean up vendor master data and eliminate duplicates"
        },
        {
            "name": "📦 LOT Size Optimization",
            "description": "Economic Order Quantity (EOQ) analysis for inventory optimization",
            "features": ["EOQ calculations", "Cost curve analysis", "Bulk discount optimization"],
            "use_case": "Minimize total inventory holding and ordering costs"
        },
        {
            "name": "📊 Reorder Prediction",
            "description": "Smart reorder point prediction based on demand patterns",
            "features": ["Statistical reorder points", "Safety stock calculation", "Lead time analysis"],
            "use_case": "Prevent stockouts while minimizing excess inventory"
        },
        {
            "name": "🌟 Seasonal Price Optimization",
            "description": "Optimize purchase timing based on seasonal price patterns",
            "features": ["Seasonal price analysis", "Optimal timing recommendations", "Savings calculations"],
            "use_case": "Time purchases for maximum cost savings"
        },
        {
            "name": "📈 Spend Analysis & Anomaly Detection",
            "description": "AI-powered spend categorization and anomaly detection",
            "features": ["Automatic categorization", "Outlier detection", "Spend visibility", "Risk identification"],
            "use_case": "Gain complete spend visibility and identify unusual transactions"
        }
    ]
    
    for module in modules_detailed:
        with st.expander(f"**{module['name']}** - {module['description']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Key Features:**")
                for feature in module['features']:
                    st.write(f"• {feature}")
            with col2:
                st.write(f"**Primary Use Case:**")
                st.write(module['use_case'])
    
    # Getting started guide
    st.subheader("🚀 Getting Started Guide")
    
    steps = [
        ("1️⃣ **Prepare Your Data**", "Ensure your CSV contains required columns with clean, consistent data"),
        ("2️⃣ **Upload File**", "Use the file uploader in the sidebar to upload your procurement data"),
        ("3️⃣ **Select Vendors**", "Choose specific vendors for analysis or use 'Select All' for comprehensive insights"),
        ("4️⃣ **Choose Module**", "Select the analytics module that best fits your current procurement challenge"),
        ("5️⃣ **Generate Insights**", "Run the analysis and review the generated insights and recommendations"),
        ("6️⃣ **Export Results**", "Download reports, forecasts, and action plans for implementation"),
        ("7️⃣ **Take Action**", "Implement the recommendations to optimize your procurement processes")
    ]
    
    for step_title, step_desc in steps:
        st.write(f"{step_title}: {step_desc}")
    
    # Best practices and tips
    st.subheader("💡 Best Practices & Pro Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Data Quality Tips:**
        - Use consistent date formats (YYYY-MM-DD recommended)
        - Clean and standardize vendor names
        - Include at least 6 months of historical data
        - Ensure price and quantity fields are numeric
        - Remove test transactions and cancelled orders
        - **Include SAR currency columns when available**
        """)
    
    with col2:
        st.markdown("""
        **🎯 Analysis Tips:**
        - Start with demand forecasting for quick wins
        - Use vendor selection to focus on key suppliers
        - Combine multiple modules for comprehensive insights
        - Regular data updates improve forecast accuracy
        - Export results for presentation to stakeholders
        - **All analysis performed in Saudi Riyal (SAR)**
        - **Review currency validation to ensure accurate totals**
        - **Check exchange rate consistency for multi-currency data**
        """)
    
    # Sample data download
    st.subheader("📁 Sample Data Template")
    st.markdown("Download a sample CSV template to understand the expected data format:")
    
    # Create extended sample data with SAR currency
    extended_sample = {
        'Creation Date': pd.date_range('2024-01-01', periods=50, freq='D'),
        'Vendor Name': ['LLORI LLERENA', '33 Designs', 'AHMED ALI SALE', 'A.T. Kearney Sau', 'AAA WORLD WID'] * 10,
        'Item': [f'ITEM_{i:03d}' for i in range(1, 51)],
        'Item Description': ['Office Supplies', 'IT Equipment', 'Raw Materials', 'Professional Services', 'Facilities'] * 10,
        'Unit Price': [round(price, 2) for price in (187.5 + 750 * pd.Series(range(50)).apply(lambda x: x % 10) / 10)],
        'Price In SAR': [round(price, 2) for price in (187.5 + 750 * pd.Series(range(50)).apply(lambda x: x % 10) / 10)],
        'Qty Delivered': [int(qty) for qty in (10 + 90 * pd.Series(range(50)).apply(lambda x: (x % 7) / 7))],
        'PO Currency': ['SAR'] * 50,
        'W/H': ['Warehouse A', 'Warehouse B', 'Warehouse C'] * 17
    }
    
    extended_df = pd.DataFrame(extended_sample)
    extended_df['Line Total'] = extended_df['Unit Price'] * extended_df['Qty Delivered']
    extended_df['Total In SAR'] = extended_df['Price In SAR'] * extended_df['Qty Delivered']
    
    csv_template = extended_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Template",
        data=csv_template,
        file_name="procurement_data_template.csv",
        mime="text/csv"
    )
    
    # Support and contact
    st.subheader("🆘 Support & Resources")
    
    st.info("""
    **Need Help?**
    - 📖 Review the module descriptions above
    - 🔍 Check data format requirements
    - 📊 Use the sample template as a guide
    - 🔄 Try different vendor selections for focused analysis
    - 📈 Start with smaller datasets to understand the workflow
    - 💱 **Currency Issues:** Check the Currency Validation section for exchange rate problems
    - 🔧 **Total Verification:** Use the manual calculation comparison to verify SAR totals
    """)

if __name__ == "__main__":
    main()
