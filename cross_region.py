import streamlit as st
import pandas as pd

def get_currency_rates():
    """Static currency exchange rates to USD"""
    return {
        'USD': 1.0, 'EUR': 0.85, 'GBP': 0.73, 'JPY': 110.0, 
        'CAD': 1.25, 'AUD': 1.35, 'CHF': 0.92, 'CNY': 6.45,
        'INR': 74.5, 'BRL': 5.2, 'MXN': 20.1, 'SGD': 1.35,
        'SAR': 3.75, 'AED': 3.67
    }

def detect_currency(amount_str):
    """Detect currency from amount string and return currency code and numeric value"""
    if isinstance(amount_str, (int, float)):
        return 'USD', float(amount_str)
    
    amount_str = str(amount_str).upper()
    
    # Simple currency detection
    if '$' in amount_str or 'USD' in amount_str:
        currency = 'USD'
    elif '€' in amount_str or 'EUR' in amount_str:
        currency = 'EUR'
    elif '£' in amount_str or 'GBP' in amount_str:
        currency = 'GBP'
    elif 'SAR' in amount_str:
        currency = 'SAR'
    elif 'AED' in amount_str:
        currency = 'AED'
    else:
        currency = 'USD'  # Default
    
    # Extract numeric value
    numeric_str = ''.join(c for c in amount_str if c.isdigit() or c in '.,')
    try:
        amount = float(numeric_str.replace(',', ''))
    except:
        amount = 0.0
    
    return currency, amount

def convert_to_usd(amount, from_currency):
    """Convert amount to USD"""
    rates = get_currency_rates()
    if from_currency == 'USD':
        return amount
    return amount / rates.get(from_currency, 1.0)

def get_region_from_warehouse(warehouse_code):
    """Extract region from warehouse code"""
    if not warehouse_code:
        return 'Unknown Region'
    
    warehouse_code = str(warehouse_code).upper()
    
    if 'EAST' in warehouse_code:
        return 'East Region'
    elif 'WEST' in warehouse_code:
        return 'West Region'
    elif 'NORTH' in warehouse_code:
        return 'North Region'
    elif 'SOUTH' in warehouse_code:
        return 'South Region'
    elif 'CENTRAL' in warehouse_code:
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization")
    
    # Process currency data
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Handle currency conversion
    if 'Currency' not in df.columns:
        # Auto-detect currency from Unit Price
        currency_data = []
        original_amounts = []
        usd_amounts = []
        
        for _, row in df.iterrows():
            currency, amount = detect_currency(row['Unit Price'])
            currency_data.append(currency)
            original_amounts.append(amount)
            usd_amounts.append(convert_to_usd(amount, currency))
        
        df['Currency'] = currency_data
        df['Original_Amount'] = original_amounts
        df['Unit_Price_USD'] = usd_amounts
    else:
        # Use existing currency column
        df['Unit_Price_USD'] = df.apply(
            lambda row: convert_to_usd(row['Unit Price'], row.get('Currency', 'USD')), 
            axis=1
        )
    
    # Business Unit mapping
    business_unit_mapping = {
        'Manufacturing': ['East Region', 'Central Region'],
        'Retail': ['West Region', 'North Region'], 
        'Distribution': ['South Region', 'Central Region'],
        'Corporate': ['East Region', 'West Region']
    }
    
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Display regional spending with currency correction
    st.subheader("📊 Regional Spending Analysis (USD)")
    regional_spending = df.groupby('Region')['Unit_Price_USD'].sum().round(2)
    
    cols = st.columns(len(regional_spending))
    for i, (region, spend) in enumerate(regional_spending.items()):
        with cols[i]:
            st.metric(region, f"${spend:,.2f}")
    
    # Currency summary
    st.subheader("💱 Currency Breakdown")
    currency_summary = df.groupby('Currency').agg({
        'Unit_Price_USD': ['count', 'sum'],
        'Original_Amount': 'sum'
    }).round(2)
    
    currency_summary.columns = ['Transactions', 'Total_USD', 'Total_Original']
    st.dataframe(currency_summary)
    
    # Original vendor analysis with currency awareness
    st.subheader("🔍 Vendor Analysis")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_region = st.selectbox("Select Region", ['All'] + list(df['Region'].unique()))
    with col2:
        selected_bu = st.selectbox("Select Business Unit", ['All'] + list(df['Business Unit'].unique()))
    
    # Filter data
    filtered_df = df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    
    if len(filtered_df) > 0:
        # Item selection
        item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
        item_filtered = filtered_df[filtered_df["Item"] == item]
        
        # Vendor comparison using USD prices
        result = (
            item_filtered.groupby(["Vendor Name", "W/H", "Region", "Business Unit", "Currency"])
            .agg({
                'Unit_Price_USD': 'mean',
                'Original_Amount': 'mean'
            })
            .reset_index()
            .sort_values(by="Unit_Price_USD")
        )
        
        result.columns = ["Vendor", "Warehouse", "Region", "Business Unit", "Currency", "Avg_Price_USD", "Avg_Original_Price"]
        result['Avg_Price_USD'] = result['Avg_Price_USD'].round(2)
        result['Avg_Original_Price'] = result['Avg_Original_Price'].round(2)
        
        st.write(f"Average Unit Price for **{item}** (sorted by USD price):")
        st.dataframe(result, use_container_width=True)
        
        # Best deals
        if len(result) > 1:
            best_deal = result.iloc[0]
            worst_deal = result.iloc[-1]
            savings = worst_deal['Avg_Price_USD'] - best_deal['Avg_Price_USD']
            
            st.success(f"💡 **Best Deal**: {best_deal['Vendor']} in {best_deal['Region']} - ${best_deal['Avg_Price_USD']:.2f}")
            st.info(f"💰 **Potential Savings**: ${savings:.2f} per unit ({((savings/worst_deal['Avg_Price_USD'])*100):.1f}% reduction)")
    
    # Business Unit summary
    st.subheader("🏢 Business Unit Summary")
    bu_summary = df.groupby('Business Unit').agg({
        'Unit_Price_USD': ['count', 'sum', 'mean'],
        'Currency': lambda x: ', '.join(x.unique())
    }).round(2)
    
    bu_summary.columns = ['Transactions', 'Total_Spend_USD', 'Avg_Price_USD', 'Currencies_Used']
    st.dataframe(bu_summary, use_container_width=True)
