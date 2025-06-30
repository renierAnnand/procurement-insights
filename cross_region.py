import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

def get_currency_rates(base_currency='USD'):
    """Get current currency exchange rates"""
    # Default rates - in production, use real-time API
    default_rates = {
        'USD': 1.0,
        'EUR': 0.85,
        'GBP': 0.73,
        'JPY': 110.0,
        'CAD': 1.25,
        'AUD': 1.35,
        'CHF': 0.92,
        'CNY': 6.45,
        'INR': 74.5,
        'BRL': 5.2,
        'MXN': 20.1,
        'SGD': 1.35,
        'SAR': 3.75,  # Saudi Riyal
        'AED': 3.67   # UAE Dirham
    }
    
    # Try to get real-time rates (optional)
    try:
        # You can integrate with a real currency API here
        # For now, using default rates
        pass
    except:
        pass
    
    return default_rates

def detect_currency(amount_str):
    """Detect currency from amount string"""
    if isinstance(amount_str, (int, float)):
        return 'USD', amount_str
    
    amount_str = str(amount_str).upper()
    
    # Currency symbols and codes
    currency_patterns = {
        'USD': ['

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Currency Configuration
    st.sidebar.subheader("Currency Settings")
    base_currency = st.sidebar.selectbox(
        "Base Currency for Analysis",
        ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR', 'BRL', 'MXN', 'SGD', 'SAR', 'AED'],
        index=0
    )
    
    # Get currency rates
    currency_rates = get_currency_rates(base_currency)
    
    # Display current exchange rates
    if st.sidebar.checkbox("Show Exchange Rates"):
        st.sidebar.write("**Current Exchange Rates**")
        rates_df = pd.DataFrame(list(currency_rates.items()), columns=['Currency', f'Rate to {base_currency}'])
        st.sidebar.dataframe(rates_df, hide_index=True)
    
    # Process currency data
    st.sidebar.subheader("Currency Processing")
    
    # Handle currency columns
    currency_columns = []
    amount_columns = []
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['price', 'cost', 'amount', 'spend', 'value']):
            amount_columns.append(col)
    
    if not amount_columns:
        st.error("No amount/price columns detected. Please ensure your data has columns with 'price', 'cost', 'amount', 'spend', or 'value' in the name.")
        return
    
    # Auto-detect and convert currencies
    if 'Currency' not in df.columns:
        # Detect currency from Unit Price or first amount column
        main_amount_col = amount_columns[0]
        st.sidebar.info(f"Auto-detecting currency from '{main_amount_col}' column...")
        
        currency_data = []
        converted_amounts = []
        
        for idx, row in df.iterrows():
            detected_currency, clean_amount = detect_currency(row[main_amount_col])
            currency_data.append(detected_currency)
            
            # Convert to base currency
            converted_amount = convert_currency(clean_amount, detected_currency, base_currency, currency_rates)
            converted_amounts.append(converted_amount)
        
        df['Original_Currency'] = currency_data
        df['Unit_Price_Original'] = [detect_currency(row[main_amount_col])[1] for _, row in df.iterrows()]
        df[f'Unit_Price_{base_currency}'] = converted_amounts
        
        # Update the main price column to use converted values
        df['Unit Price'] = df[f'Unit_Price_{base_currency}']
    else:
        # Use existing currency column
        df['Original_Currency'] = df['Currency']
        main_amount_col = 'Unit Price'
        
        # Convert existing amounts
        converted_amounts = []
        for idx, row in df.iterrows():
            original_currency = row.get('Currency', 'USD')
            original_amount = float(str(row[main_amount_col]).replace(',', '').replace('
    
    
    # Item selection
    if len(filtered_df) > 0:
        item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
        item_filtered = filtered_df[filtered_df["Item"] == item]
    else:
        st.warning("No data available for the selected filters.")
        return
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison", "Currency Analysis"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        st.caption(f"Prices shown in {base_currency} after currency conversion")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H", "Original_Currency"])
            .agg({
                "Unit Price": ['mean', 'count'],
                f"Unit_Price_{base_currency}": 'mean'
            })
            .reset_index()
        )
        
        # Flatten column names
        vendor_analysis.columns = [
            "Business Unit", "Region", "Vendor Name", "Warehouse", "Original Currency",
            f"Avg Price ({base_currency})", "Orders Count", f"Converted Avg ({base_currency})"
        ]
        
        # Remove duplicate converted column and sort
        vendor_analysis = vendor_analysis.drop(f"Converted Avg ({base_currency})", axis=1)
        vendor_analysis = vendor_analysis.sort_values(f"Avg Price ({base_currency})")
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[
                vendor_analysis.groupby("Business Unit")[f"Avg Price ({base_currency})"].idxmin()
            ]
            st.dataframe(
                best_vendors[["Business Unit", "Region", "Vendor Name", f"Avg Price ({base_currency})", "Original Currency"]], 
                use_container_width=True
            )
    
    with tab2:
        st.subheader("Business Unit Overview")
        st.caption(f"All financial metrics in {base_currency}")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")
            .agg({
                "Unit Price": ['mean', 'min', 'max', 'count'],
                "Original_Currency": lambda x: ', '.join(x.unique())
            })
            .reset_index()
        )
        
        # Flatten columns
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders", "Currencies Used"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Currency usage by business unit
        st.subheader("Currency Usage by Business Unit")
        currency_usage = (
            item_filtered.groupby(["Business Unit", "Original_Currency"])
            .size()
            .reset_index(name='Transaction Count')
        )
        
        if not currency_usage.empty:
            currency_pivot = currency_usage.pivot(
                index="Business Unit", 
                columns="Original_Currency", 
                values="Transaction Count"
            ).fillna(0).astype(int)
            st.dataframe(currency_pivot, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    
    with tab3:
        st.subheader("Region Comparison")
        st.caption(f"All amounts converted to {base_currency} for accurate comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")
            .agg({
                "Unit Price": ['mean', 'min', 'max', 'count'],
                "Original_Currency": lambda x: ', '.join(x.unique())
            })
            .reset_index()
        )
        
        # Flatten columns
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders", "Currencies Used"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
            .round(2)
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
            
        # Currency distribution by region
        st.subheader("Currency Distribution by Region")
        region_currency = (
            item_filtered.groupby(["Region", "Original_Currency"])
            .size()
            .reset_index(name='Transaction Count')
        )
        
        if not region_currency.empty:
            region_currency_pivot = region_currency.pivot(
                index="Region", 
                columns="Original_Currency", 
                values="Transaction Count"
            ).fillna(0).astype(int)
            st.dataframe(region_currency_pivot, use_container_width=True)
    
    with tab4:
        st.subheader("Currency Analysis")
        st.caption("Multi-currency transaction analysis and conversion details")
        
        # Currency conversion summary
        st.subheader("Currency Conversion Summary")
        conversion_summary = []
        
        for currency in item_filtered['Original_Currency'].unique():
            currency_data = item_filtered[item_filtered['Original_Currency'] == currency]
            total_original = currency_data['Unit_Price_Original'].sum()
            total_converted = currency_data[f'Unit_Price_{base_currency}'].sum()
            avg_rate = total_converted / total_original if total_original > 0 else 0
            
            conversion_summary.append({
                'Currency': currency,
                'Transactions': len(currency_data),
                f'Total Original ({currency})': f"{total_original:,.2f}",
                f'Total Converted ({base_currency})': f"{total_converted:,.2f}",
                f'Avg Rate ({currency} to {base_currency})': f"{avg_rate:.4f}"
            })
        
        conversion_df = pd.DataFrame(conversion_summary)
        st.dataframe(conversion_df, use_container_width=True)
        
        # Exchange rate impact analysis
        st.subheader("Exchange Rate Impact Analysis")
        
        # Calculate potential savings from currency optimization
        if len(item_filtered['Original_Currency'].unique()) > 1:
            best_currency_data = item_filtered.groupby('Original_Currency')['Unit Price'].mean().sort_values()
            if len(best_currency_data) > 1:
                best_currency = best_currency_data.index[0]
                worst_currency = best_currency_data.index[-1]
                
                potential_savings = best_currency_data.iloc[-1] - best_currency_data.iloc[0]
                
                st.info(f"""
                **Currency Optimization Opportunity:**
                - Best performing currency: **{best_currency}** (Avg: {best_currency_data.iloc[0]:.2f} {base_currency})
                - Worst performing currency: **{worst_currency}** (Avg: {best_currency_data.iloc[-1]:.2f} {base_currency})
                - Potential savings per unit: **{potential_savings:.2f} {base_currency}**
                """)
        
        # Current exchange rates table
        st.subheader("Current Exchange Rates")
        rates_display = pd.DataFrame([
            {'From Currency': curr, 'To Currency': base_currency, 'Exchange Rate': rate}
            for curr, rate in currency_rates.items()
            if curr in item_filtered['Original_Currency'].unique()
        ])
        st.dataframe(rates_display, use_container_width=True)
    
    
    # Summary insights
    st.subheader("🔑 Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        total_currencies = item_filtered['Original_Currency'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        total_spend = item_filtered['Unit Price'].sum()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Currencies", total_currencies)
        with col5:
            st.metric(f"Total Spend ({base_currency})", f"{total_spend:,.0f}")
        
        # Multi-currency insights
        st.subheader("💱 Multi-Currency Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Currency distribution
            currency_dist = item_filtered['Original_Currency'].value_counts()
            st.write("**Transaction Volume by Currency:**")
            for currency, count in currency_dist.items():
                percentage = (count / len(item_filtered)) * 100
                st.write(f"- {currency}: {count} transactions ({percentage:.1f}%)")
        
        with col2:
            # Price analysis by currency
            price_by_currency = item_filtered.groupby('Original_Currency')['Unit Price'].agg(['mean', 'count']).round(2)
            st.write(f"**Average Prices by Currency (in {base_currency}):**")
            for currency, data in price_by_currency.iterrows():
                st.write(f"- {currency}: {data['mean']:.2f} ({data['count']} transactions)")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis[f"Avg Price ({base_currency})"].max() - vendor_analysis[f"Avg Price ({base_currency})"].min()
            st.success(f"💡 **Optimization Opportunity**: Price difference of {price_diff:.2f} {base_currency} between highest and lowest vendor for {item}")
        
        # Currency optimization recommendation
        if total_currencies > 1:
            best_currency_avg = item_filtered.groupby('Original_Currency')['Unit Price'].mean().min()
            worst_currency_avg = item_filtered.groupby('Original_Currency')['Unit Price'].mean().max()
            currency_savings = worst_currency_avg - best_currency_avg
            
            if currency_savings > 0:
                st.info(f"💱 **Currency Optimization**: Standardizing to the best-performing currency could save up to {currency_savings:.2f} {base_currency} per unit")
    
    # Export functionality
    st.subheader("📤 Export Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Detailed Analysis"):
            # Combine all analysis data
            try:
                with pd.ExcelWriter('vendor_optimization_analysis.xlsx', engine='openpyxl') as writer:
                    # Main data with conversions
                    item_filtered.to_excel(writer, sheet_name='Raw Data', index=False)
                    
                    # Analysis sheets
                    vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
                    bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
                    region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
                    
                    if not cross_region.empty:
                        cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
                    
                    # Currency analysis
                    conversion_df.to_excel(writer, sheet_name='Currency Analysis', index=False)
                    
                    # Exchange rates
                    rates_display.to_excel(writer, sheet_name='Exchange Rates', index=False)
                
                st.success("✅ Detailed analysis exported to 'vendor_optimization_analysis.xlsx'")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col2:
        if st.button("Export Currency Summary"):
            try:
                # Create summary report
                summary_data = {
                    'Regional Spending': regional_spending,
                    'Currency Summary': currency_summary,
                    'Conversion Rates': pd.DataFrame(list(currency_rates.items()), columns=['Currency', f'Rate to {base_currency}'])
                }
                
                with pd.ExcelWriter('currency_summary.xlsx', engine='openpyxl') as writer:
                    for sheet_name, data in summary_data.items():
                        data.to_excel(writer, sheet_name=sheet_name)
                
                st.success("✅ Currency summary exported to 'currency_summary.xlsx'")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    # Data quality warnings
    st.subheader("⚠️ Data Quality Checks")
    
    warnings = []
    
    # Check for missing currency conversions
    if item_filtered['Unit Price'].isna().any():
        warnings.append("Some currency conversions failed - check for invalid amount formats")
    
    # Check for extreme exchange rate differences
    price_variance = item_filtered.groupby('Original_Currency')['Unit Price'].std()
    high_variance_currencies = price_variance[price_variance > price_variance.mean() * 2]
    if len(high_variance_currencies) > 0:
        warnings.append(f"High price variance detected in: {', '.join(high_variance_currencies.index)}")
    
    # Check for single-source currencies
    single_transaction_currencies = item_filtered['Original_Currency'].value_counts()
    single_currencies = single_transaction_currencies[single_transaction_currencies == 1]
    if len(single_currencies) > 0:
        warnings.append(f"Single transaction currencies detected: {', '.join(single_currencies.index)} - verify exchange rates")
    
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
    else:
        st.success("✅ All data quality checks passed")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df), 'USD', 'DOLLAR'],
        'EUR': ['€', 'EUR', 'EURO'],
        'GBP': ['£', 'GBP', 'POUND'],
        'JPY': ['¥', 'JPY', 'YEN'],
        'CAD': ['CAD', 'C

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df)],
        'AUD': ['AUD', 'A

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df)],
        'CHF': ['CHF', 'SFR'],
        'CNY': ['CNY', '¥', 'YUAN'],
        'INR': ['INR', '₹', 'RUPEE'],
        'BRL': ['BRL', 'R

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df), 'REAL'],
        'MXN': ['MXN', 'PESO'],
        'SGD': ['SGD', 'S

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df)],
        'SAR': ['SAR', 'RIYAL'],
        'AED': ['AED', 'DIRHAM']
    }
    
    for currency, patterns in currency_patterns.items():
        for pattern in patterns:
            if pattern in amount_str:
                # Extract numeric value
                numeric_str = ''.join(c for c in amount_str if c.isdigit() or c in '.,')
                try:
                    amount = float(numeric_str.replace(',', ''))
                    return currency, amount
                except:
                    continue
    
    # Default to USD if no currency detected
    try:
        amount = float(str(amount_str).replace(',', '').replace('

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df), ''))
        return 'USD', amount
    except:
        return 'USD', 0.0

def convert_currency(amount, from_currency, to_currency, rates):
    """Convert amount from one currency to another"""
    if from_currency == to_currency:
        return amount
    
    # Convert to USD first, then to target currency
    usd_amount = amount / rates.get(from_currency, 1.0)
    converted_amount = usd_amount * rates.get(to_currency, 1.0)
    
    return converted_amount

def create_business_unit_mapping():
    """Create or load business unit to region mapping"""
    # Default mapping - can be customized or loaded from file
    default_mapping = {
        'Manufacturing': ['East Region', 'Central Region'],
        'Retail': ['West Region', 'North Region'],
        'Distribution': ['South Region', 'Central Region'],
        'Corporate': ['East Region', 'West Region']
    }
    return default_mapping

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df), ''))
            converted_amount = convert_currency(original_amount, original_currency, base_currency, currency_rates)
            converted_amounts.append(converted_amount)
        
        df[f'Unit_Price_{base_currency}'] = converted_amounts
        df['Unit Price'] = df[f'Unit_Price_{base_currency}']
    
    # Show currency summary
    currency_summary = df.groupby('Original_Currency').agg({
        'Unit Price': ['count', 'sum', 'mean'],
        f'Unit_Price_{base_currency}': ['sum', 'mean']
    }).round(2)
    
    st.sidebar.write("**Currency Summary**")
    st.sidebar.dataframe(currency_summary)
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    with col3:
        # Currency filter
        currencies = ['All'] + list(df['Original_Currency'].unique())
        selected_currency = st.selectbox("Filter by Original Currency", currencies)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_currency != 'All':
        filtered_df = filtered_df[filtered_df['Original_Currency'] == selected_currency]
    
    # Regional Spending Analysis (corrected for multi-currency)
    st.subheader("📊 Regional Spending Analysis")
    st.caption(f"All amounts converted to {base_currency} for accurate comparison")
    
    regional_spending = filtered_df.groupby('Region').agg({
        f'Unit_Price_{base_currency}': 'sum',
        'Vendor Name': 'nunique',
        'Item': 'nunique',
        'Unit Price': 'mean'
    }).round(2)
    
    regional_spending.columns = ['Total Spend', 'Unique Vendors', 'Unique Items', 'Avg Unit Price']
    
    # Display spending metrics
    cols = st.columns(len(regional_spending))
    for i, (region, data) in enumerate(regional_spending.iterrows()):
        with cols[i]:
            st.metric(
                label=region,
                value=f"{currency_rates.get(base_currency, 1)} {data['Total Spend']:,.2f}",
                delta=f"{data['Unique Vendors']} vendors"
            )
    
    # Regional comparison table
    st.subheader("🔍 Regional Comparison")
    comparison_df = regional_spending.copy()
    comparison_df['Spend %'] = (comparison_df['Total Spend'] / comparison_df['Total Spend'].sum() * 100).round(1)
    comparison_df = comparison_df.sort_values('Total Spend', ascending=False)
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df), 'USD', 'DOLLAR'],
        'EUR': ['€', 'EUR', 'EURO'],
        'GBP': ['£', 'GBP', 'POUND'],
        'JPY': ['¥', 'JPY', 'YEN'],
        'CAD': ['CAD', 'C

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df)],
        'AUD': ['AUD', 'A

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df)],
        'CHF': ['CHF', 'SFR'],
        'CNY': ['CNY', '¥', 'YUAN'],
        'INR': ['INR', '₹', 'RUPEE'],
        'BRL': ['BRL', 'R

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df), 'REAL'],
        'MXN': ['MXN', 'PESO'],
        'SGD': ['SGD', 'S

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df)],
        'SAR': ['SAR', 'RIYAL'],
        'AED': ['AED', 'DIRHAM']
    }
    
    for currency, patterns in currency_patterns.items():
        for pattern in patterns:
            if pattern in amount_str:
                # Extract numeric value
                numeric_str = ''.join(c for c in amount_str if c.isdigit() or c in '.,')
                try:
                    amount = float(numeric_str.replace(',', ''))
                    return currency, amount
                except:
                    continue
    
    # Default to USD if no currency detected
    try:
        amount = float(str(amount_str).replace(',', '').replace('

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df), ''))
        return 'USD', amount
    except:
        return 'USD', 0.0

def convert_currency(amount, from_currency, to_currency, rates):
    """Convert amount from one currency to another"""
    if from_currency == to_currency:
        return amount
    
    # Convert to USD first, then to target currency
    usd_amount = amount / rates.get(from_currency, 1.0)
    converted_amount = usd_amount * rates.get(to_currency, 1.0)
    
    return converted_amount

def create_business_unit_mapping():
    """Create or load business unit to region mapping"""
    # Default mapping - can be customized or loaded from file
    default_mapping = {
        'Manufacturing': ['East Region', 'Central Region'],
        'Retail': ['West Region', 'North Region'],
        'Distribution': ['South Region', 'Central Region'],
        'Corporate': ['East Region', 'West Region']
    }
    return default_mapping

def get_region_from_warehouse(warehouse_code, region_mapping=None):
    """Extract region from warehouse code or use custom mapping"""
    if region_mapping:
        for region, warehouses in region_mapping.items():
            if warehouse_code in warehouses:
                return region
    
    # Fallback: try to extract region from warehouse code
    if 'EAST' in warehouse_code.upper():
        return 'East Region'
    elif 'WEST' in warehouse_code.upper():
        return 'West Region'
    elif 'NORTH' in warehouse_code.upper():
        return 'North Region'
    elif 'SOUTH' in warehouse_code.upper():
        return 'South Region'
    elif 'CENTRAL' in warehouse_code.upper():
        return 'Central Region'
    else:
        return 'Unknown Region'

def display(df):
    st.header("Cross-Region Vendor Optimization with Business Units")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Business Unit mapping configuration
    st.sidebar.subheader("Business Unit - Region Mapping")
    business_unit_mapping = create_business_unit_mapping()
    
    # Allow users to modify the mapping
    if st.sidebar.checkbox("Edit Business Unit Mapping"):
        edited_mapping = {}
        for bu, regions in business_unit_mapping.items():
            st.sidebar.write(f"**{bu}**")
            selected_regions = st.sidebar.multiselect(
                f"Regions for {bu}",
                options=['East Region', 'West Region', 'North Region', 'South Region', 'Central Region'],
                default=regions,
                key=f"bu_{bu}"
            )
            if selected_regions:
                edited_mapping[bu] = selected_regions
        business_unit_mapping = edited_mapping
    
    # Add Region column if not present
    if 'Region' not in df.columns:
        df['Region'] = df['W/H'].apply(get_region_from_warehouse)
    
    # Add Business Unit column based on mapping
    def assign_business_unit(region):
        for bu, regions in business_unit_mapping.items():
            if region in regions:
                return bu
        return "Unassigned"
    
    df['Business Unit'] = df['Region'].apply(assign_business_unit)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Business Unit filter
        business_units = ['All'] + list(df['Business Unit'].unique())
        selected_bu = st.selectbox("Select Business Unit", business_units)
    
    with col2:
        # Region filter
        regions = ['All'] + list(df['Region'].unique())
        selected_region = st.selectbox("Select Region", regions)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_bu != 'All':
        filtered_df = filtered_df[filtered_df['Business Unit'] == selected_bu]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Item selection
    item = st.selectbox("Select Item", filtered_df["Item"].dropna().unique())
    item_filtered = filtered_df[filtered_df["Item"] == item]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Vendor Analysis", "Business Unit Overview", "Region Comparison"])
    
    with tab1:
        st.subheader("Vendor Analysis by Business Unit & Region")
        
        # Group by vendor, business unit, and region
        vendor_analysis = (
            item_filtered.groupby(["Business Unit", "Region", "Vendor Name", "W/H"])["Unit Price"]
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values(by="mean")
        )
        vendor_analysis.columns = ["Business Unit", "Region", "Vendor Name", "Warehouse", "Avg Unit Price", "Orders Count"]
        
        st.dataframe(vendor_analysis, use_container_width=True)
        
        # Best vendor per business unit
        if not vendor_analysis.empty:
            st.subheader("Best Vendor by Business Unit (Lowest Price)")
            best_vendors = vendor_analysis.loc[vendor_analysis.groupby("Business Unit")["Avg Unit Price"].idxmin()]
            st.dataframe(best_vendors[["Business Unit", "Region", "Vendor Name", "Avg Unit Price"]], use_container_width=True)
    
    with tab2:
        st.subheader("Business Unit Overview")
        
        # Business unit summary
        bu_summary = (
            item_filtered.groupby("Business Unit")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        bu_summary.columns = ["Business Unit", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        bu_summary['Price Variance'] = bu_summary['Max Price'] - bu_summary['Min Price']
        
        st.dataframe(bu_summary, use_container_width=True)
        
        # Business unit to region mapping display
        st.subheader("Current Business Unit - Region Mapping")
        mapping_data = []
        for bu, regions in business_unit_mapping.items():
            for region in regions:
                mapping_data.append({"Business Unit": bu, "Region": region})
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    with tab3:
        st.subheader("Region Comparison")
        
        # Regional analysis
        region_analysis = (
            item_filtered.groupby("Region")["Unit Price"]
            .agg(['mean', 'min', 'max', 'count'])
            .reset_index()
        )
        region_analysis.columns = ["Region", "Avg Price", "Min Price", "Max Price", "Total Orders"]
        region_analysis['Price Variance'] = region_analysis['Max Price'] - region_analysis['Min Price']
        region_analysis = region_analysis.sort_values(by="Avg Price")
        
        st.dataframe(region_analysis, use_container_width=True)
        
        # Cross-region vendor comparison
        st.subheader("Cross-Region Vendor Comparison")
        cross_region = (
            item_filtered.groupby(["Vendor Name", "Region"])["Unit Price"]
            .mean()
            .reset_index()
            .pivot(index="Vendor Name", columns="Region", values="Unit Price")
            .fillna("-")
        )
        
        if not cross_region.empty:
            st.dataframe(cross_region, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not item_filtered.empty:
        total_vendors = item_filtered['Vendor Name'].nunique()
        total_regions = item_filtered['Region'].nunique()
        total_business_units = item_filtered['Business Unit'].nunique()
        avg_price_overall = item_filtered['Unit Price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", total_vendors)
        with col2:
            st.metric("Active Regions", total_regions)
        with col3:
            st.metric("Business Units", total_business_units)
        with col4:
            st.metric("Overall Avg Price", f"${avg_price_overall:.2f}")
        
        # Optimization opportunities
        if len(vendor_analysis) > 1:
            price_diff = vendor_analysis['Avg Unit Price'].max() - vendor_analysis['Avg Unit Price'].min()
            st.info(f"💡 **Optimization Opportunity**: Price difference of ${price_diff:.2f} between highest and lowest vendor for {item}")
    
    # Export functionality
    if st.button("Export Analysis"):
        # Combine all analysis data
        with pd.ExcelWriter('vendor_optimization_analysis.xlsx') as writer:
            vendor_analysis.to_excel(writer, sheet_name='Vendor Analysis', index=False)
            bu_summary.to_excel(writer, sheet_name='Business Unit Summary', index=False)
            region_analysis.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not cross_region.empty:
                cross_region.to_excel(writer, sheet_name='Cross Region Comparison')
        
        st.success("Analysis exported to 'vendor_optimization_analysis.xlsx'")

# Example usage function for testing
def create_sample_data():
    """Create sample data for testing"""
    import random
    
    items = ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    warehouses = ['EAST-WH1', 'WEST-WH2', 'NORTH-WH3', 'SOUTH-WH4', 'CENTRAL-WH5']
    
    data = []
    for _ in range(100):
        data.append({
            'Item': random.choice(items),
            'Vendor Name': random.choice(vendors),
            'W/H': random.choice(warehouses),
            'Unit Price': round(random.uniform(50, 500), 2)
        })
    
    return pd.DataFrame(data)

# Uncomment the lines below to test with sample data
# if __name__ == "__main__":
#     sample_df = create_sample_data()
#     display(sample_df)
