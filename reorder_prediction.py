import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

def perform_bulk_analysis(df, items, lead_time_days, safety_stock_days, analysis_period, currency):
    """Perform bulk analysis on multiple items"""
    results = []
    
    if not items:
        st.error("No items provided for analysis")
        return None
    
    # Calculate date filter based on analysis period
    end_date = datetime.now()
    start_date = None
    
    try:
        if analysis_period == "Last 30 days":
            start_date = end_date - timedelta(days=30)
        elif analysis_period == "Last 60 days":
            start_date = end_date - timedelta(days=60)
        elif analysis_period == "Last 90 days":
            start_date = end_date - timedelta(days=90)
        elif analysis_period == "Last 180 days":
            start_date = end_date - timedelta(days=180)
        # For "All available data", start_date remains None
    except Exception as e:
        st.error(f"Error setting date filter: {str(e)}")
        return None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, item in enumerate(items):
        try:
            status_text.text(f"Analyzing item {idx + 1} of {len(items)}: {item}")
            progress_bar.progress((idx + 1) / len(items))
            
            # Filter data for current item
            item_df = df[df["Item"] == item].copy()
            
            if item_df.empty:
                st.warning(f"No data found for item: {item}")
                continue
            
            # Check for required columns
            if "Creation Date" not in item_df.columns or "Qty Delivered" not in item_df.columns:
                st.warning(f"Missing required columns for item: {item}")
                continue
            
            # Convert Creation Date to datetime
            try:
                item_df["Creation Date"] = pd.to_datetime(item_df["Creation Date"], errors='coerce')
                item_df = item_df.dropna(subset=["Creation Date"])
            except Exception as e:
                st.warning(f"Date conversion error for item {item}: {str(e)}")
                continue
            
            if item_df.empty:
                st.warning(f"No valid dates for item: {item}")
                continue
            
            # Apply date filter if specified
            if start_date:
                item_df = item_df[item_df["Creation Date"] >= start_date]
            
            if item_df.empty:
                st.warning(f"No data in selected period for item: {item}")
                continue
            
            # Convert and validate quantities
            try:
                item_df["Qty Delivered"] = pd.to_numeric(item_df["Qty Delivered"], errors='coerce')
                item_df = item_df.dropna(subset=["Qty Delivered"])
                item_df = item_df[item_df["Qty Delivered"] > 0]  # Remove zero/negative quantities
            except Exception as e:
                st.warning(f"Quantity conversion error for item {item}: {str(e)}")
                continue
            
            if item_df.empty:
                st.warning(f"No valid quantities for item: {item}")
                continue
            
            # Calculate daily demand
            daily_demand = item_df.groupby(item_df["Creation Date"].dt.date)["Qty Delivered"].sum()
            
            if daily_demand.empty or daily_demand.sum() <= 0:
                st.warning(f"No valid demand data for item: {item}")
                continue
            
            # Calculate metrics
            avg_daily_demand = daily_demand.mean()
            demand_std = daily_demand.std() if len(daily_demand) > 1 else 0
            min_demand = daily_demand.min()
            max_demand = daily_demand.max()
            total_demand = daily_demand.sum()
            days_with_data = len(daily_demand)
            
            # Calculate reorder points
            lead_time_demand = avg_daily_demand * lead_time_days
            safety_stock = avg_daily_demand * safety_stock_days
            reorder_point = lead_time_demand + safety_stock
            
            # Statistical reorder point (95% service level)
            if demand_std > 0 and len(daily_demand) > 1:
                statistical_safety_stock = demand_std * np.sqrt(lead_time_days) * 1.65
                statistical_reorder_point = lead_time_demand + statistical_safety_stock
            else:
                statistical_safety_stock = safety_stock
                statistical_reorder_point = reorder_point
            
            # Calculate different scenarios
            optimistic_rp = avg_daily_demand * lead_time_days * 0.8  # 20% less
            conservative_rp = reorder_point * 1.3  # 30% more
            
            # Risk assessment
            if reorder_point > statistical_reorder_point * 1.2:
                risk_level = "Low Risk (High Stock)"
            elif reorder_point < statistical_reorder_point * 0.8:
                risk_level = "High Risk (Low Stock)"
            else:
                risk_level = "Balanced"
            
            results.append({
                "Item": item,
                "Avg Daily Demand": round(avg_daily_demand, 2),
                "Demand Std Dev": round(demand_std, 2),
                "Min Daily Demand": round(min_demand, 2),
                "Max Daily Demand": round(max_demand, 2),
                "Total Demand": round(total_demand, 2),
                "Days with Data": days_with_data,
                "Lead Time (days)": lead_time_days,
                "Safety Stock": round(safety_stock, 2),
                "Reorder Point": round(reorder_point, 2),
                "Statistical Reorder Point": round(statistical_reorder_point, 2),
                "Optimistic Scenario": round(optimistic_rp, 2),
                "Conservative Scenario": round(conservative_rp, 2),
                "Risk Level": risk_level,
                "Currency": currency
            })
            
        except Exception as e:
            st.error(f"Unexpected error analyzing item {item}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if results:
        st.success(f"✅ Successfully analyzed {len(results)} out of {len(items)} items")
        return pd.DataFrame(results)
    else:
        st.error("❌ No items could be analyzed. Please check your data format and filters.")
        return None

def display_bulk_results(results_df, currency):
    """Display bulk analysis results with visualizations and export options"""
    
    st.subheader("📊 Bulk Analysis Results")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Items Analyzed", len(results_df))
    
    with col2:
        avg_reorder_point = results_df["Reorder Point"].mean()
        st.metric("Avg Reorder Point", f"{avg_reorder_point:.2f}")
    
    with col3:
        high_risk_items = len(results_df[results_df["Risk Level"] == "High Risk (Low Stock)"])
        st.metric("High Risk Items", high_risk_items)
    
    with col4:
        total_avg_demand = results_df["Avg Daily Demand"].sum()
        st.metric("Total Daily Demand", f"{total_avg_demand:.2f}")
    
    # Risk Level Distribution
    st.subheader("🎯 Risk Level Distribution")
    risk_counts = results_df["Risk Level"].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(risk_counts)
    
    with col2:
        for risk, count in risk_counts.items():
            percentage = (count / len(results_df)) * 100
            st.metric(risk, f"{count} ({percentage:.1f}%)")
    
    # Top items by reorder point
    st.subheader("🔝 Top Items by Reorder Point")
    top_items = results_df.nlargest(10, "Reorder Point")[["Item", "Reorder Point", "Avg Daily Demand", "Risk Level"]]
    st.dataframe(top_items, use_container_width=True)
    
    # Interactive filters for detailed view
    st.subheader("🔍 Detailed Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level:",
            results_df["Risk Level"].unique(),
            default=results_df["Risk Level"].unique()
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Reorder Point", "Avg Daily Demand", "Risk Level", "Item"]
        )
    
    # Apply filters
    filtered_results = results_df[results_df["Risk Level"].isin(risk_filter)]
    
    # Sort results
    if sort_by in ["Reorder Point", "Avg Daily Demand"]:
        filtered_results = filtered_results.sort_values(sort_by, ascending=False)
    else:
        filtered_results = filtered_results.sort_values(sort_by)
    
    # Display filtered results
    st.dataframe(
        filtered_results,
        use_container_width=True,
        column_config={
            "Avg Daily Demand": st.column_config.NumberColumn(format="%.2f"),
            "Reorder Point": st.column_config.NumberColumn(format="%.2f"),
            "Statistical Reorder Point": st.column_config.NumberColumn(format="%.2f"),
        }
    )
    
    # Export functionality
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Excel export
        def create_excel_file(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Bulk Analysis Results', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Items', 'High Risk Items', 'Low Risk Items', 'Balanced Items', 
                              'Average Reorder Point', 'Total Daily Demand'],
                    'Value': [
                        len(df),
                        len(df[df["Risk Level"] == "High Risk (Low Stock)"]),
                        len(df[df["Risk Level"] == "Low Risk (High Stock)"]),
                        len(df[df["Risk Level"] == "Balanced"]),
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
            file_name=f"bulk_reorder_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # CSV export
        csv_data = filtered_results.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"bulk_reorder_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # JSON export
        json_data = filtered_results.to_json(orient='records', indent=2)
        st.download_button(
            label="🔗 Download JSON",
            data=json_data,
            file_name=f"bulk_reorder_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def display(df):
    st.header("Smart Reorder Point Prediction")
    
    # Data Overview Section
    if st.checkbox("🔍 Show Data Overview", help="View summary of available data"):
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Unique Items", df["Item"].nunique() if "Item" in df.columns else "N/A")
        with col3:
            if "Creation Date" in df.columns:
                try:
                    date_range = pd.to_datetime(df["Creation Date"]).dt.date
                    st.metric("Date Range", f"{date_range.min()} to {date_range.max()}")
                except:
                    st.metric("Date Range", "Invalid dates")
            else:
                st.metric("Date Range", "No date column")
        
        # Show column info
        with st.expander("📊 Available Columns"):
            st.write("**Columns in dataset:**")
            for i, col in enumerate(df.columns):
                col_info = f"{i+1}. **{col}** - {df[col].dtype}"
                if col in ["Item", "Region", "Vendor"]:
                    unique_count = df[col].nunique()
                    col_info += f" ({unique_count} unique values)"
                st.write(col_info)
        
        # Show sample data
        with st.expander("👀 Sample Data"):
            st.dataframe(df.head(10))
    
    # Region and Currency Selection
    st.subheader("🌍 Region & Currency Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique regions from the dataframe
        regions = sorted(df["Region"].dropna().unique()) if "Region" in df.columns else ["AGC", "UAE", "KSA", "USA", "Europe"]
        selected_region = st.selectbox("Select Business Region", regions)
    
    with col2:
        # Get all available currencies
        currency_options = [
            "AED", "SAR", "USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CNY", 
            "INR", "SGD", "HKD", "CHF", "NOK", "SEK", "DKK", "PLN", "CZK",
            "BRL", "MXN", "ZAR", "NGN", "KES", "EGP", "QAR", "KWD", "BHD", "OMR"
        ]
        
        # Auto-detect currency based on region
        currency_map = {
            # Middle East & Africa
            "AGC": "AED", "UAE": "AED", "Dubai": "AED", "Abu Dhabi": "AED",
            "KSA": "SAR", "Saudi Arabia": "SAR", "Riyadh": "SAR", "Jeddah": "SAR",
            "Qatar": "QAR", "Kuwait": "KWD", "Bahrain": "BHD", "Oman": "OMR",
            "Egypt": "EGP", "South Africa": "ZAR", "Nigeria": "NGN", "Kenya": "KES",
            
            # Americas
            "USA": "USD", "US": "USD", "Canada": "CAD", "Mexico": "MXN", "Brazil": "BRL",
            
            # Europe
            "Europe": "EUR", "Germany": "EUR", "France": "EUR", "UK": "GBP", 
            "Switzerland": "CHF", "Norway": "NOK", "Sweden": "SEK", "Denmark": "DKK",
            
            # Asia Pacific
            "China": "CNY", "Japan": "JPY", "India": "INR", "Singapore": "SGD",
            "Hong Kong": "HKD", "Australia": "AUD"
        }
        
        # Get default currency for selected region
        default_currency = currency_map.get(selected_region, "USD")
        default_index = currency_options.index(default_currency) if default_currency in currency_options else 0
        
        selected_currency = st.selectbox("Select Currency", currency_options, index=default_index)
    
    # Filter data by selected region
    if "Region" in df.columns:
        region_df = df[df["Region"] == selected_region].copy()
    else:
        region_df = df.copy()  # Use all data if no region column
    
    # Display region info boxes (matching contracting module style)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Region:** {selected_region}")
    
    with col2:
        st.info(f"**Currency:** {selected_currency}")
    
    with col3:
        st.info(f"**Records:** {len(region_df)}")
    
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
    
    # Analysis Mode Selection
    st.subheader("📊 Analysis Mode")
    analysis_mode = st.radio(
        "Choose Analysis Type:",
        ["🎯 Single Item Analysis", "📋 Bulk Analysis"],
        horizontal=True
    )
    
    available_items = region_df["Item"].dropna().unique()
    if len(available_items) == 0:
        st.warning("No items available for the selected filters.")
        return
    
    if analysis_mode == "🎯 Single Item Analysis":
        # Single Item Analysis with Search
        st.subheader("📦 Single Item Analysis")
        
        # Search functionality
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
        
        # Single item analysis (existing code will continue here)
        
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
            filtered_bulk_items = list(available_items)  # Convert to list to avoid issues
        
        if not filtered_bulk_items:
            st.warning(f"No items found matching '{search_term_bulk}'")
            return
        
        # Bulk selection controls
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("Select All Items"):
                st.session_state.selected_bulk_items = filtered_bulk_items.copy()
        
        with col2:
            if st.button("Clear Selection"):
                st.session_state.selected_bulk_items = []
        
        # Initialize session state for bulk items
        if 'selected_bulk_items' not in st.session_state:
            st.session_state.selected_bulk_items = filtered_bulk_items[:10] if len(filtered_bulk_items) >= 10 else filtered_bulk_items.copy()
        
        # Ensure selected items are still valid after filtering
        valid_selections = [item for item in st.session_state.selected_bulk_items if item in filtered_bulk_items]
        if len(valid_selections) != len(st.session_state.selected_bulk_items):
            st.session_state.selected_bulk_items = valid_selections
        
        selected_bulk_items = st.multiselect(
            "Select Items for Bulk Analysis:",
            filtered_bulk_items,
            default=st.session_state.selected_bulk_items,
            key="bulk_items_multiselect",
            help="Select multiple items to analyze simultaneously"
        )
        
        # Update session state
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
                ["Last 30 days", "Last 60 days", "Last 90 days", "Last 180 days", "All available data"],
                index=2
            )
        
        # Run bulk analysis
        if st.button("🚀 Run Bulk Analysis", type="primary"):
            with st.spinner("Analyzing items... This may take a moment."):
                bulk_results = perform_bulk_analysis(
                    region_df, selected_bulk_items, bulk_lead_time, 
                    bulk_safety_stock, analysis_period, selected_currency
                )
                
                if bulk_results is not None:
                    display_bulk_results(bulk_results, selected_currency)
        
        selected_item = st.selectbox("Select Item", filtered_items)
        
        # Filter the dataset for selected item
        item_df = region_df[region_df["Item"] == selected_item].copy()
        
        # Debug information
        st.write(f"**🔍 Data Filtering Debug Info:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"Total rows in region: {len(region_df)}")
        with col2:
            st.write(f"Rows for selected item: {len(item_df)}")
        with col3:
            if not item_df.empty:
                st.write(f"Date range: {len(item_df)} records")
            else:
                st.write("No records found")
        with col4:
            if not item_df.empty and "Qty Delivered" in item_df.columns:
                valid_qty = item_df["Qty Delivered"].notna().sum()
                st.write(f"Valid qty records: {valid_qty}")
        
        if item_df.empty:
            st.error(f"❌ No data available for item: **{selected_item}**")
            
            # Show available items for debugging
            with st.expander("🔍 Debug: Available Items in Current Filter"):
                available_debug = region_df["Item"].value_counts().head(10)
                st.write("Top 10 items in current region/vendor filter:")
                st.dataframe(available_debug)
            
            return
        
        # Check for required columns
        required_columns = ["Creation Date", "Qty Delivered"]
        missing_columns = [col for col in required_columns if col not in item_df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.write("Available columns:", list(item_df.columns))
            return
        
        # Check for valid data
        item_df = item_df.dropna(subset=["Creation Date", "Qty Delivered"])
        
        if item_df.empty:
            st.error(f"❌ No valid data found for item: **{selected_item}**")
            st.write("All records have missing Creation Date or Qty Delivered values.")
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
        st.subheader("📊 Key Performance Indicators")
        
        try:
            # Convert Creation Date to datetime with better error handling
            st.write("**📅 Processing Date Information...**")
            
            # Show sample of raw data
            with st.expander("🔍 Sample Raw Data"):
                st.write(f"Sample of {len(item_df)} records:")
                st.dataframe(item_df.head(10))
            
            # Try to convert dates
            try:
                item_df["Creation Date"] = pd.to_datetime(item_df["Creation Date"], errors='coerce')
                invalid_dates = item_df["Creation Date"].isna().sum()
                if invalid_dates > 0:
                    st.warning(f"⚠️ Found {invalid_dates} invalid dates that will be excluded")
                    item_df = item_df.dropna(subset=["Creation Date"])
            except Exception as e:
                st.error(f"❌ Error converting dates: {str(e)}")
                return
            
            if item_df.empty:
                st.error("❌ No valid dates remaining after date conversion")
                return
            
            # Check quantity data
            try:
                item_df["Qty Delivered"] = pd.to_numeric(item_df["Qty Delivered"], errors='coerce')
                invalid_qty = item_df["Qty Delivered"].isna().sum()
                if invalid_qty > 0:
                    st.warning(f"⚠️ Found {invalid_qty} invalid quantities that will be excluded")
                    item_df = item_df.dropna(subset=["Qty Delivered"])
            except Exception as e:
                st.error(f"❌ Error converting quantities: {str(e)}")
                return
            
            if item_df.empty:
                st.error("❌ No valid quantities remaining after data cleaning")
                return
            
            # Calculate daily demand
            st.write("**📊 Calculating Daily Demand...**")
            daily_demand = item_df.groupby(item_df["Creation Date"].dt.date)["Qty Delivered"].sum()
            
            if daily_demand.empty:
                st.error("❌ No daily demand data could be calculated")
                return
            
            # Show daily demand summary
            st.write(f"✅ Successfully calculated daily demand for {len(daily_demand)} days")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Days", len(daily_demand))
            with col2:
                st.metric("Total Quantity", f"{daily_demand.sum():.2f}")
            with col3:
                st.metric("Date Range", f"{daily_demand.index.min()} to {daily_demand.index.max()}")
            
            # Calculate average daily demand
            avg_daily_demand = daily_demand.mean()
            
            if avg_daily_demand <= 0:
                st.error("❌ Average daily demand is zero or negative - cannot calculate reorder point")
                return
            
            # Calculate demand standard deviation for safety stock calculation
            demand_std = daily_demand.std()
            
            # Enhanced Reorder Point Calculation
            # Formula: Average Daily Demand × Lead Time + Safety Stock
            lead_time_demand = avg_daily_demand * lead_time_days
            safety_stock = avg_daily_demand * safety_stock_days
            reorder_point = lead_time_demand + safety_stock
            
            # Alternative safety stock using statistical method
            if demand_std > 0:
                statistical_safety_stock = demand_std * np.sqrt(lead_time_days) * 1.65  # 95% service level
                statistical_reorder_point = lead_time_demand + statistical_safety_stock
            else:
                statistical_safety_stock = safety_stock
                statistical_reorder_point = reorder_point
                st.info("ℹ️ Demand has no variation - using simple safety stock calculation")
            
            st.success("✅ Analysis completed successfully!")
            
            # Display Key Metrics (matching the screenshot style)
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
                    help="Recommended reorder point"
                )
            
            # Visual Analysis Section (matching screenshot)
            st.subheader("📈 Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Daily Demand Trend Analysis**")
                
                # Prepare data for trend analysis
                daily_demand_df = daily_demand.reset_index()
                daily_demand_df.columns = ['Date', 'Daily Demand']
                daily_demand_df['Date'] = pd.to_datetime(daily_demand_df['Date'])
                
                # Calculate 7-day moving average
                daily_demand_df['7-Day Moving Average'] = daily_demand_df['Daily Demand'].rolling(window=7, center=True).mean()
                
                # Create chart data
                chart_data = daily_demand_df.set_index('Date')[['Daily Demand', '7-Day Moving Average']]
                st.line_chart(chart_data)
            
            with col2:
                st.write("**🎯 Reorder Point Scenarios Comparison**")
                
                # Calculate different scenarios (matching screenshot)
                optimistic_rp = reorder_point * 0.6  # More optimistic
                likely_rp = reorder_point  # Current calculation
                conservative_rp = reorder_point * 1.4  # More conservative
                
                # Create scenario comparison data
                scenarios_data = {
                    'Optimistic': optimistic_rp,
                    'Likely': likely_rp,
                    'Conservative': conservative_rp
                }
                
                scenarios_df = pd.DataFrame(list(scenarios_data.items()), columns=['Scenario Type', 'Reorder Point Quantity'])
                
                # Display as bar chart
                st.bar_chart(scenarios_df.set_index('Scenario Type'))
                
                # Display scenario values
                st.write("**Scenario Values:**")
                for scenario, value in scenarios_data.items():
                    st.write(f"• {scenario}: {value:.1f}")
            
            # Additional Statistics
            st.subheader("📋 Detailed Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Minimum Daily Demand", f"{daily_demand.min():.2f}")
                st.metric("Maximum Daily Demand", f"{daily_demand.max():.2f}")
                st.metric("Total Analysis Days", len(daily_demand))
            
            with col2:
                st.metric("Demand Standard Deviation", f"{demand_std:.2f}")
                st.metric("Statistical Reorder Point", f"{statistical_reorder_point:.2f}")
                st.metric("Demand Variability", f"{(demand_std/avg_daily_demand)*100:.1f}%")
            
            # Monthly aggregation for additional insight
            item_df["Month"] = item_df["Creation Date"].dt.to_period("M")
            monthly_demand = item_df.groupby("Month")["Qty Delivered"].sum()
            
            if len(monthly_demand) > 1:
                st.subheader("📅 Monthly Demand Pattern")
                monthly_df = monthly_demand.reset_index()
                monthly_df['Month'] = monthly_df['Month'].astype(str)
                monthly_df.columns = ['Month', 'Monthly Demand']
                st.bar_chart(monthly_df.set_index('Month'))
            
            # Summary Table
            st.subheader("📋 Analysis Summary")
            
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
                    "Statistical Reorder Point",
                    "Risk Assessment"
                ],
                "Value": [
                    selected_region,
                    selected_currency,
                    selected_item,
                    f"{daily_demand.index.min()} to {daily_demand.index.max()}",
                    len(daily_demand),
                    f"{avg_daily_demand:.2f}",
                    lead_time_days,
                    safety_stock_days,
                    f"{reorder_point:.2f}",
                    f"{statistical_reorder_point:.2f}",
                    "Balanced" if abs(reorder_point - statistical_reorder_point) < statistical_reorder_point * 0.2 else "Review Needed"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            
            # Recommendations
            st.subheader("💡 Smart Recommendations")
            
            if reorder_point > statistical_reorder_point * 1.2:
                st.info("🔍 Your current safety stock setting is conservative. Consider the statistical reorder point for cost optimization.")
            elif reorder_point < statistical_reorder_point * 0.8:
                st.warning("⚠️ Your safety stock might be too low. Consider increasing it to avoid stockouts.")
            else:
                st.success("✅ Your reorder point settings appear balanced between cost and service level.")
            
            # Export single item results
            st.subheader("📥 Export Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Create single item export data
                export_data = {
                    'Item': [selected_item],
                    'Region': [selected_region],
                    'Currency': [selected_currency],
                    'Average Daily Demand': [avg_daily_demand],
                    'Lead Time (days)': [lead_time_days],
                    'Safety Stock': [safety_stock],
                    'Reorder Point': [reorder_point],
                    'Statistical Reorder Point': [statistical_reorder_point],
                    'Analysis Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                }
                export_df = pd.DataFrame(export_data)
                
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Analysis (CSV)",
                    data=csv_data,
                    file_name=f"reorder_analysis_{selected_item}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create detailed export with daily data
                detailed_export = daily_demand_df.copy()
                detailed_export['Item'] = selected_item
                detailed_export['Region'] = selected_region
                detailed_export['Currency'] = display_currency
                detailed_export['Reorder Point'] = reorder_point
                
                detailed_csv = detailed_export.to_csv(index=False)
                st.download_button(
                    label="📊 Download Detailed Data (CSV)",
                    data=detailed_csv,
                    file_name=f"detailed_analysis_{selected_item}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
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
