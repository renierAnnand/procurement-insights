import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

def display(df):
    st.header("📦 Smart Reorder Point Prediction")
    st.markdown("**Advanced procurement analytics with intelligent reorder point calculations**")
    
    # Region & Currency Selection (matching contracting module)
    st.subheader("🌍 Region & Currency Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get actual business regions from data
        if "Region" in df.columns:
            regions = sorted(df["Region"].dropna().unique())
        else:
            # Try other common region column names
            region_columns = ["Business Region", "Region", "Country", "Location", "Site"]
            regions = []
            for col in region_columns:
                if col in df.columns:
                    regions = sorted(df[col].dropna().unique())
                    break
            if not regions:
                regions = ["All Regions"]
        
        selected_region = st.selectbox("Select Business Region", regions)
    
    with col2:
        # Currency options
        currency_options = ["AED", "SAR", "USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CNY", "INR", "SGD", "HKD", "CHF", "NOK", "SEK", "DKK", "PLN", "CZK", "BRL", "MXN", "ZAR", "NGN", "KES", "EGP", "QAR", "KWD", "BHD", "OMR"]
        
        # Auto-detect currency based on region
        region_currency_map = {
            "AGC": "AED", "UAE": "AED", "KSA": "SAR", "Saudi Arabia": "SAR",
            "Qatar": "QAR", "Kuwait": "KWD", "Bahrain": "BHD", "Oman": "OMR",
            "Egypt": "EGP", "South Africa": "ZAR", "Nigeria": "NGN",
            "USA": "USD", "Canada": "CAD", "Mexico": "MXN", "Brazil": "BRL",
            "Germany": "EUR", "France": "EUR", "UK": "GBP", "Switzerland": "CHF",
            "China": "CNY", "Japan": "JPY", "India": "INR", "Singapore": "SGD", "Australia": "AUD"
        }
        
        detected_currency = "USD"  # Default
        for region_key, currency in region_currency_map.items():
            if region_key.lower() in selected_region.lower():
                detected_currency = currency
                break
        
        default_index = currency_options.index(detected_currency) if detected_currency in currency_options else 0
        selected_currency = st.selectbox("Select Currency", currency_options, index=default_index)
    
    # Filter data by selected region
    if "Region" in df.columns and selected_region != "All Regions":
        region_df = df[df["Region"] == selected_region].copy()
    else:
        region_df = df.copy()
    
    if region_df.empty:
        st.warning(f"No data available for region: {selected_region}")
        return
    
    # Display region summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Records:** {len(region_df)}")
    
    with col2:
        unique_items = region_df["Item"].nunique() if "Item" in region_df.columns else 0
        st.info(f"**Unique Items:** {unique_items}")
    
    with col3:
        st.info(f"**Total Orders:** {len(region_df)}")
    
    # Vendor Selection (if vendor filtering is needed)
    if "Vendor" in region_df.columns:
        st.subheader("🏢 Vendor Selection")
        
        vendors = sorted(region_df["Vendor"].dropna().unique())
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("Select All"):
                st.session_state.selected_vendors = vendors
        
        with col2:
            if st.button("Deselect All"):
                st.session_state.selected_vendors = []
        
        # Initialize session state
        if 'selected_vendors' not in st.session_state:
            st.session_state.selected_vendors = vendors
        
        selected_vendors = st.multiselect(
            "Select Vendors",
            vendors,
            default=st.session_state.selected_vendors,
            key="vendor_multiselect"
        )
        
        st.session_state.selected_vendors = selected_vendors
        
        if selected_vendors:
            region_df = region_df[region_df["Vendor"].isin(selected_vendors)]
        else:
            st.warning("Please select at least one vendor.")
            return
    
    # Analysis Mode Selection
    st.subheader("📊 Analysis Mode")
    analysis_mode = st.radio(
        "Choose Analysis Type:",
        ["🎯 Single Item Analysis", "📋 Bulk Analysis"],
        horizontal=True
    )
    
    # Get available items
    available_items = sorted(region_df["Item"].dropna().unique()) if "Item" in region_df.columns else []
    
    if not available_items:
        st.warning("No items available for analysis.")
        return
    
    if analysis_mode == "🎯 Single Item Analysis":
        # Single Item Analysis
        st.subheader("📦 Single Item Analysis")
        
        # Search for items
        search_term = st.text_input(
            "🔍 Search for Item:", 
            placeholder="Type to search items...",
            help="Search by item name, code, or description"
        )
        
        # Filter items based on search
        if search_term:
            filtered_items = [item for item in available_items if search_term.lower() in str(item).lower()]
            if not filtered_items:
                st.warning(f"No items found matching '{search_term}'")
                return
        else:
            filtered_items = available_items
        
        selected_item = st.selectbox("Select Item", filtered_items)
        
        # Filter data for selected item
        item_df = region_df[region_df["Item"] == selected_item].copy()
        
        if item_df.empty:
            st.error(f"No data found for item: {selected_item}")
            return
        
        # Check required columns
        required_columns = ["Creation Date", "Qty Delivered"]
        missing_columns = [col for col in required_columns if col not in item_df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return
        
        # Clean data
        item_df = item_df.dropna(subset=required_columns)
        if item_df.empty:
            st.error("No valid data after cleaning missing values.")
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
        
        # Calculate reorder point
        try:
            # Convert dates and calculate daily demand
            item_df["Creation Date"] = pd.to_datetime(item_df["Creation Date"], errors='coerce')
            item_df = item_df.dropna(subset=["Creation Date"])
            
            if item_df.empty:
                st.error("No valid dates found.")
                return
            
            # Group by date and sum quantities
            daily_demand = item_df.groupby(item_df["Creation Date"].dt.date)["Qty Delivered"].sum()
            
            if daily_demand.empty:
                st.error("No demand data could be calculated.")
                return
            
            # Calculate metrics
            avg_daily_demand = daily_demand.mean()
            demand_std = daily_demand.std()
            
            # Reorder Point Formula: Average Daily Demand × Lead Time + Safety Stock
            lead_time_demand = avg_daily_demand * lead_time_days
            safety_stock = avg_daily_demand * safety_stock_days
            reorder_point = lead_time_demand + safety_stock
            
            # Statistical alternative (95% service level)
            if demand_std > 0:
                statistical_safety_stock = demand_std * np.sqrt(lead_time_days) * 1.65
                statistical_reorder_point = lead_time_demand + statistical_safety_stock
            else:
                statistical_safety_stock = safety_stock
                statistical_reorder_point = reorder_point
            
            # Display Key Performance Indicators
            st.subheader("📊 Key Performance Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Average Daily Demand", 
                    f"{avg_daily_demand:.2f}",
                    help="Average quantity demanded per day"
                )
            
            with col2:
                st.metric(
                    "Average Lead Time", 
                    f"{lead_time_days} days",
                    help="Expected lead time for procurement"
                )
            
            with col3:
                st.metric(
                    "Safety Stock", 
                    f"{safety_stock:.2f}",
                    help="Buffer stock for demand variability"
                )
            
            with col4:
                st.metric(
                    "Final Suggested Reorder Point", 
                    f"{reorder_point:.2f}",
                    help="Recommended reorder point using formula: Avg Daily Demand × Lead Time + Safety Stock"
                )
            
            # Visual Analysis
            st.subheader("📈 Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Daily Demand Trend Analysis**")
                daily_demand_df = daily_demand.reset_index()
                daily_demand_df.columns = ['Date', 'Daily Demand']
                daily_demand_df['7-Day Moving Average'] = daily_demand_df['Daily Demand'].rolling(window=7, center=True).mean()
                
                chart_data = daily_demand_df.set_index('Date')[['Daily Demand', '7-Day Moving Average']]
                st.line_chart(chart_data)
            
            with col2:
                st.write("**🎯 Reorder Point Scenarios Comparison**")
                
                optimistic_rp = reorder_point * 0.6
                likely_rp = reorder_point
                conservative_rp = reorder_point * 1.4
                
                scenarios_df = pd.DataFrame({
                    'Scenario Type': ['Optimistic', 'Likely', 'Conservative'],
                    'Reorder Point Quantity': [optimistic_rp, likely_rp, conservative_rp]
                })
                
                st.bar_chart(scenarios_df.set_index('Scenario Type'))
                
                st.write("**Scenario Values:**")
                st.write(f"• Optimistic: {optimistic_rp:.1f}")
                st.write(f"• Likely: {likely_rp:.1f}")
                st.write(f"• Conservative: {conservative_rp:.1f}")
            
            # Additional Statistics
            st.subheader("📋 Additional Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Minimum Daily Demand", f"{daily_demand.min():.2f}")
                st.metric("Maximum Daily Demand", f"{daily_demand.max():.2f}")
            
            with col2:
                st.metric("Demand Standard Deviation", f"{demand_std:.2f}")
                st.metric("Statistical Reorder Point", f"{statistical_reorder_point:.2f}")
            
            with col3:
                st.metric("Total Analysis Days", len(daily_demand))
                variability = (demand_std/avg_daily_demand)*100 if avg_daily_demand > 0 else 0
                st.metric("Demand Variability", f"{variability:.1f}%")
            
            # Summary Table
            st.subheader("📋 Analysis Summary")
            
            summary_data = {
                "Metric": [
                    "Selected Region",
                    "Currency", 
                    "Selected Item",
                    "Analysis Period",
                    "Average Daily Demand",
                    "Lead Time (days)",
                    "Safety Stock (days)",
                    "**Final Reorder Point**",
                    "Statistical Alternative",
                    "Formula Used"
                ],
                "Value": [
                    selected_region,
                    selected_currency,
                    selected_item,
                    f"{daily_demand.index.min()} to {daily_demand.index.max()}",
                    f"{avg_daily_demand:.2f}",
                    lead_time_days,
                    safety_stock_days,
                    f"**{reorder_point:.2f}**",
                    f"{statistical_reorder_point:.2f}",
                    "Avg Daily Demand × Lead Time + Safety Stock"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            
            # Smart Recommendations
            st.subheader("💡 Smart Recommendations")
            
            if reorder_point > statistical_reorder_point * 1.2:
                st.info("🔍 Your safety stock setting is conservative. Consider the statistical alternative for cost optimization.")
            elif reorder_point < statistical_reorder_point * 0.8:
                st.warning("⚠️ Your safety stock might be too low. Consider increasing it to avoid stockouts.")
            else:
                st.success("✅ Your reorder point settings appear well-balanced between cost and service level.")
            
            # Export Analysis
            st.subheader("📥 Export Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Summary export
                export_data = {
                    'Item': [selected_item],
                    'Region': [selected_region],
                    'Currency': [selected_currency],
                    'Average_Daily_Demand': [avg_daily_demand],
                    'Lead_Time_Days': [lead_time_days],
                    'Safety_Stock_Days': [safety_stock_days],
                    'Safety_Stock_Quantity': [safety_stock],
                    'Reorder_Point': [reorder_point],
                    'Statistical_Reorder_Point': [statistical_reorder_point],
                    'Analysis_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                }
                
                export_df = pd.DataFrame(export_data)
                csv_data = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📄 Download Summary (CSV)",
                    data=csv_data,
                    file_name=f"reorder_point_analysis_{selected_item}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Detailed daily data export
                detailed_export = daily_demand_df.copy()
                detailed_export['Item'] = selected_item
                detailed_export['Region'] = selected_region
                detailed_export['Currency'] = selected_currency
                detailed_export['Reorder_Point'] = reorder_point
                
                detailed_csv = detailed_export.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download Detailed Data (CSV)",
                    data=detailed_csv,
                    file_name=f"detailed_demand_data_{selected_item}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.info("Please check your data format and ensure required columns are present.")
    
    else:
        # Bulk Analysis
        st.subheader("📋 Bulk Analysis")
        
        # Search and filter for bulk selection
        search_term_bulk = st.text_input(
            "🔍 Search Items for Bulk Analysis:", 
            placeholder="Type to filter items for bulk analysis...",
            key="bulk_search"
        )
        
        # Filter items based on search
        if search_term_bulk:
            filtered_bulk_items = [item for item in available_items if search_term_bulk.lower() in str(item).lower()]
        else:
            filtered_bulk_items = available_items
        
        if not filtered_bulk_items:
            st.warning(f"No items found matching '{search_term_bulk}'")
            return
        
        # Bulk selection controls
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("Select All Items"):
                st.session_state.selected_bulk_items = filtered_bulk_items[:20]  # Limit to 20 for performance
        
        with col2:
            if st.button("Clear Selection"):
                st.session_state.selected_bulk_items = []
        
        # Initialize session state for bulk items
        if 'selected_bulk_items' not in st.session_state:
            st.session_state.selected_bulk_items = filtered_bulk_items[:10]  # Default to first 10
        
        selected_bulk_items = st.multiselect(
            "Select Items for Bulk Analysis:",
            filtered_bulk_items,
            default=st.session_state.selected_bulk_items,
            key="bulk_items_multiselect",
            help="Select multiple items to analyze simultaneously (max 50 recommended)"
        )
        
        st.session_state.selected_bulk_items = selected_bulk_items
        
        if not selected_bulk_items:
            st.warning("Please select at least one item for bulk analysis.")
            return
        
        # Bulk analysis parameters
        st.subheader("⚙️ Bulk Analysis Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bulk_lead_time = st.number_input(
                "Default Lead Time (days)", 
                min_value=1, 
                max_value=365, 
                value=14,
                key="bulk_lead_time"
            )
        
        with col2:
            bulk_safety_stock = st.number_input(
                "Default Safety Stock (days)", 
                min_value=0, 
                max_value=90, 
                value=7,
                key="bulk_safety_stock"
            )
        
        with col3:
            analysis_period = st.selectbox(
                "Analysis Period",
                ["All available data", "Last 180 days", "Last 90 days", "Last 60 days", "Last 30 days"],
                index=0
            )
        
        # Run bulk analysis
        if st.button("🚀 Run Bulk Analysis", type="primary"):
            with st.spinner("Analyzing items... This may take a moment."):
                
                # Filter by analysis period
                if analysis_period != "All available data":
                    days = int(analysis_period.split()[1])
                    cutoff_date = datetime.now() - timedelta(days=days)
                    analysis_df = region_df[pd.to_datetime(region_df["Creation Date"], errors='coerce') >= cutoff_date]
                else:
                    analysis_df = region_df
                
                bulk_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, item in enumerate(selected_bulk_items):
                    try:
                        status_text.text(f"Analyzing item {idx + 1} of {len(selected_bulk_items)}: {item}")
                        progress_bar.progress((idx + 1) / len(selected_bulk_items))
                        
                        # Filter data for current item
                        item_df = analysis_df[analysis_df["Item"] == item].copy()
                        
                        if item_df.empty or "Creation Date" not in item_df.columns or "Qty Delivered" not in item_df.columns:
                            continue
                        
                        # Clean and process data
                        item_df = item_df.dropna(subset=["Creation Date", "Qty Delivered"])
                        if item_df.empty:
                            continue
                        
                        item_df["Creation Date"] = pd.to_datetime(item_df["Creation Date"], errors='coerce')
                        item_df = item_df.dropna(subset=["Creation Date"])
                        
                        if item_df.empty:
                            continue
                        
                        # Calculate daily demand
                        daily_demand = item_df.groupby(item_df["Creation Date"].dt.date)["Qty Delivered"].sum()
                        
                        if daily_demand.empty or daily_demand.sum() <= 0:
                            continue
                        
                        # Calculate metrics using the same formula
                        avg_daily_demand = daily_demand.mean()
                        demand_std = daily_demand.std() if len(daily_demand) > 1 else 0
                        
                        # Reorder Point Formula: Average Daily Demand × Lead Time + Safety Stock
                        lead_time_demand = avg_daily_demand * bulk_lead_time
                        safety_stock = avg_daily_demand * bulk_safety_stock
                        reorder_point = lead_time_demand + safety_stock
                        
                        # Statistical alternative
                        if demand_std > 0:
                            statistical_safety_stock = demand_std * np.sqrt(bulk_lead_time) * 1.65
                            statistical_reorder_point = lead_time_demand + statistical_safety_stock
                        else:
                            statistical_reorder_point = reorder_point
                        
                        # Risk assessment
                        if reorder_point > statistical_reorder_point * 1.2:
                            risk_level = "Conservative (High Stock)"
                        elif reorder_point < statistical_reorder_point * 0.8:
                            risk_level = "Aggressive (Low Stock)"
                        else:
                            risk_level = "Balanced"
                        
                        bulk_results.append({
                            "Item": item,
                            "Avg Daily Demand": round(avg_daily_demand, 2),
                            "Lead Time (days)": bulk_lead_time,
                            "Safety Stock (days)": bulk_safety_stock,
                            "Safety Stock (qty)": round(safety_stock, 2),
                            "Reorder Point": round(reorder_point, 2),
                            "Statistical Alternative": round(statistical_reorder_point, 2),
                            "Min Daily Demand": round(daily_demand.min(), 2),
                            "Max Daily Demand": round(daily_demand.max(), 2),
                            "Demand Std Dev": round(demand_std, 2),
                            "Total Demand": round(daily_demand.sum(), 2),
                            "Analysis Days": len(daily_demand),
                            "Risk Assessment": risk_level,
                            "Currency": selected_currency
                        })
                        
                    except Exception as e:
                        continue
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if bulk_results:
                    results_df = pd.DataFrame(bulk_results)
                    
                    st.success(f"✅ Successfully analyzed {len(results_df)} out of {len(selected_bulk_items)} items")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Items Analyzed", len(results_df))
                    
                    with col2:
                        avg_reorder_point = results_df["Reorder Point"].mean()
                        st.metric("Avg Reorder Point", f"{avg_reorder_point:.2f}")
                    
                    with col3:
                        conservative_count = len(results_df[results_df["Risk Assessment"] == "Conservative (High Stock)"])
                        st.metric("Conservative Items", conservative_count)
                    
                    with col4:
                        total_avg_demand = results_df["Avg Daily Demand"].sum()
                        st.metric("Total Daily Demand", f"{total_avg_demand:.2f}")
                    
                    # Risk assessment distribution
                    st.subheader("🎯 Risk Assessment Distribution")
                    
                    risk_counts = results_df["Risk Assessment"].value_counts()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.bar_chart(risk_counts)
                    
                    with col2:
                        for risk, count in risk_counts.items():
                            percentage = (count / len(results_df)) * 100
                            st.metric(risk, f"{count} ({percentage:.1f}%)")
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    
                    # Interactive filters
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        risk_filter = st.multiselect(
                            "Filter by Risk Assessment:",
                            results_df["Risk Assessment"].unique(),
                            default=results_df["Risk Assessment"].unique()
                        )
                    
                    with col2:
                        sort_by = st.selectbox(
                            "Sort by:",
                            ["Reorder Point", "Avg Daily Demand", "Risk Assessment", "Item"]
                        )
                    
                    # Apply filters and sorting
                    filtered_results = results_df[results_df["Risk Assessment"].isin(risk_filter)]
                    
                    if sort_by in ["Reorder Point", "Avg Daily Demand"]:
                        filtered_results = filtered_results.sort_values(sort_by, ascending=False)
                    else:
                        filtered_results = filtered_results.sort_values(sort_by)
                    
                    # Display results
                    st.dataframe(
                        filtered_results,
                        use_container_width=True,
                        column_config={
                            "Avg Daily Demand": st.column_config.NumberColumn(format="%.2f"),
                            "Reorder Point": st.column_config.NumberColumn(format="%.2f"),
                            "Statistical Alternative": st.column_config.NumberColumn(format="%.2f"),
                        }
                    )
                    
                    # Export bulk results
                    st.subheader("📥 Export Bulk Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Excel export
                        def create_excel_file(df):
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df.to_excel(writer, sheet_name='Bulk Reorder Analysis', index=False)
                                
                                # Add summary sheet
                                summary_data = {
                                    'Metric': ['Total Items', 'Conservative Items', 'Balanced Items', 'Aggressive Items', 'Average Reorder Point', 'Total Daily Demand'],
                                    'Value': [
                                        len(df),
                                        len(df[df["Risk Assessment"] == "Conservative (High Stock)"]),
                                        len(df[df["Risk Assessment"] == "Balanced"]),
                                        len(df[df["Risk Assessment"] == "Aggressive (Low Stock)"]),
                                        f"{df['Reorder Point'].mean():.2f}",
                                        f"{df['Avg Daily Demand'].sum():.2f}"
                                    ]
                                }
                                summary_df = pd.DataFrame(summary_data)
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            return output.getvalue()
                        
                        excel_data = create_excel_file(filtered_results)
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_data,
                            file_name=f"bulk_reorder_analysis_{selected_region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    with col2:
                        # CSV export
                        csv_data = filtered_results.to_csv(index=False)
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv_data,
                            file_name=f"bulk_reorder_analysis_{selected_region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.error("❌ No items could be analyzed. Please check your data format and selection.")

# Example usage (for testing)
if __name__ == "__main__":
    # Sample data structure
    sample_data = {
        'Region': ['AGC'] * 100 + ['KSA'] * 100,
        'Item': [f'Item_{i%20}' for i in range(200)],
        'Vendor': [f'Vendor_{i%5}' for i in range(200)],
        'Creation Date': pd.date_range('2024-01-01', periods=200, freq='D'),
        'Qty Delivered': np.random.randint(5, 50, 200),
        'Unit Price': np.random.uniform(10, 100, 200)
    }
    df = pd.DataFrame(sample_data)
    display(df)
