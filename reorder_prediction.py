import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def display(df):
    st.header("Smart Reorder Point Prediction")
    
    # Region and Currency Selection
    st.subheader("📍 Region & Currency Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique regions from the dataframe
        regions = df["Region"].dropna().unique() if "Region" in df.columns else ["AGC", "UAE", "KSA"]
        selected_region = st.selectbox("Select Region", regions)
    
    with col2:
        # Currency mapping (you can expand this based on your data)
        currency_map = {
            "AGC": "AED",
            "UAE": "AED", 
            "KSA": "SAR",
            "USA": "USD",
            "EUR": "EUR"
        }
        display_currency = currency_map.get(selected_region, "AED")
        st.metric("Currency", display_currency)
    
    # Filter data by selected region
    if "Region" in df.columns:
        region_df = df[df["Region"] == selected_region].copy()
    else:
        region_df = df.copy()  # Use all data if no region column
    
    if region_df.empty:
        st.warning(f"No data available for region: {selected_region}")
        return
    
    # Vendor Selection with Select All/Deselect All
    if "Vendor" in region_df.columns:
        st.subheader("🏢 Vendor Selection")
        
        vendors = region_df["Vendor"].dropna().unique().tolist()
        
        # Select All / Deselect All buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("Select All"):
                st.session_state.selected_vendors = vendors
        
        with col2:
            if st.button("Deselect All"):
                st.session_state.selected_vendors = []
        
        # Initialize session state for vendors
        if 'selected_vendors' not in st.session_state:
            st.session_state.selected_vendors = vendors
        
        selected_vendors = st.multiselect(
            "Select Vendors",
            vendors,
            default=st.session_state.selected_vendors,
            key="vendor_multiselect"
        )
        
        # Update session state
        st.session_state.selected_vendors = selected_vendors
        
        # Filter by selected vendors
        if selected_vendors:
            region_df = region_df[region_df["Vendor"].isin(selected_vendors)]
        else:
            st.warning("Please select at least one vendor.")
            return
    
    # Item Selection
    st.subheader("📦 Item Analysis")
    
    available_items = region_df["Item"].dropna().unique()
    if len(available_items) == 0:
        st.warning("No items available for the selected filters.")
        return
    
    selected_item = st.selectbox("Select Item", available_items)
    
    # Filter the dataset for selected item
    item_df = region_df[region_df["Item"] == selected_item].copy()
    
    if item_df.empty:
        st.warning(f"No data available for item: {selected_item}")
        return
    
    # Reorder Point Parameters
    st.subheader("⚙️ Reorder Point Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lead_time_days = st.number_input(
            "Lead Time (days)", 
            min_value=1, 
            max_value=365, 
            value=14, 
            help="Average time between placing an order and receiving it"
        )
    
    with col2:
        safety_stock_days = st.number_input(
            "Safety Stock (days)", 
            min_value=0, 
            max_value=90, 
            value=7, 
            help="Additional buffer stock to account for demand variability"
        )
    
    # Data Processing and Analysis
    st.subheader("📊 Analysis Results")
    
    try:
        # Convert Creation Date to datetime
        item_df["Creation Date"] = pd.to_datetime(item_df["Creation Date"])
        
        # Calculate daily demand
        daily_demand = item_df.groupby(item_df["Creation Date"].dt.date)["Qty Delivered"].sum()
        
        # Calculate average daily demand
        avg_daily_demand = daily_demand.mean()
        
        # Calculate demand standard deviation for safety stock calculation
        demand_std = daily_demand.std()
        
        # Enhanced Reorder Point Calculation
        # Formula: Average Daily Demand × Lead Time + Safety Stock
        lead_time_demand = avg_daily_demand * lead_time_days
        safety_stock = avg_daily_demand * safety_stock_days
        reorder_point = lead_time_demand + safety_stock
        
        # Alternative safety stock using statistical method
        statistical_safety_stock = demand_std * np.sqrt(lead_time_days) * 1.65  # 95% service level
        statistical_reorder_point = lead_time_demand + statistical_safety_stock
        
        # Display Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Average Daily Demand", 
                f"{avg_daily_demand:.2f}",
                help="Average quantity demanded per day"
            )
        
        with col2:
            st.metric(
                "Lead Time Demand", 
                f"{lead_time_demand:.2f}",
                help="Expected demand during lead time"
            )
        
        with col3:
            st.metric(
                "Safety Stock", 
                f"{safety_stock:.2f}",
                help="Buffer stock for demand variability"
            )
        
        with col4:
            st.metric(
                "**Reorder Point**", 
                f"{reorder_point:.2f}",
                help="Recommended reorder point"
            )
        
        # Additional Statistics
        st.subheader("📈 Demand Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Minimum Daily Demand", f"{daily_demand.min():.2f}")
            st.metric("Maximum Daily Demand", f"{daily_demand.max():.2f}")
        
        with col2:
            st.metric("Demand Standard Deviation", f"{demand_std:.2f}")
            st.metric("Statistical Reorder Point", f"{statistical_reorder_point:.2f}")
        
        # Visualizations
        st.subheader("📊 Demand Visualization")
        
        # Daily demand chart
        st.write("**Daily Demand Over Time**")
        daily_demand_df = daily_demand.reset_index()
        daily_demand_df.columns = ['Date', 'Daily Demand']
        st.line_chart(daily_demand_df.set_index('Date'))
        
        # Monthly aggregation for trend analysis
        item_df["Month"] = item_df["Creation Date"].dt.to_period("M")
        monthly_demand = item_df.groupby("Month")["Qty Delivered"].sum()
        
        st.write("**Monthly Demand Trend**")
        monthly_df = monthly_demand.reset_index()
        monthly_df['Month'] = monthly_df['Month'].astype(str)
        monthly_df.columns = ['Month', 'Monthly Demand']
        st.bar_chart(monthly_df.set_index('Month'))
        
        # Summary Table
        st.subheader("📋 Summary")
        
        summary_data = {
            "Metric": [
                "Selected Region",
                "Currency",
                "Selected Item",
                "Analysis Period",
                "Total Days with Data",
                "Average Daily Demand",
                "Lead Time (days)",
                "Safety Stock (days)",
                "Recommended Reorder Point",
                "Statistical Reorder Point"
            ],
            "Value": [
                selected_region,
                display_currency,
                selected_item,
                f"{daily_demand.index.min()} to {daily_demand.index.max()}",
                len(daily_demand),
                f"{avg_daily_demand:.2f}",
                lead_time_days,
                safety_stock_days,
                f"{reorder_point:.2f}",
                f"{statistical_reorder_point:.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if reorder_point > statistical_reorder_point * 1.2:
            st.info("🔍 Your current safety stock setting is conservative. Consider the statistical reorder point for cost optimization.")
        elif reorder_point < statistical_reorder_point * 0.8:
            st.warning("⚠️ Your safety stock might be too low. Consider increasing it to avoid stockouts.")
        else:
            st.success("✅ Your reorder point settings appear balanced between cost and service level.")
            
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.info("Please check your data format and ensure 'Creation Date' and 'Qty Delivered' columns are present.")

# Example usage (uncomment to test)
# if __name__ == "__main__":
#     # Sample data for testing
#     sample_data = {
#         'Item': ['Item A'] * 100,
#         'Creation Date': pd.date_range('2024-01-01', periods=100, freq='D'),
#         'Qty Delivered': np.random.randint(10, 50, 100),
#         'Region': ['AGC'] * 100,
#         'Vendor': ['Vendor 1'] * 50 + ['Vendor 2'] * 50
#     }
#     df = pd.DataFrame(sample_data)
#     display(df)
