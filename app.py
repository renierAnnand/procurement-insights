import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_and_clean_data, forecast_demand

# Import all modules
import contracting_opportunities
import cross_region
import duplicates
import lot_size_optimization
import reorder_prediction
import seasonal_price_optimization
import spend_categorization_anomaly

st.set_page_config(page_title="Smart Procurement Analytics Suite", layout="wide")

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
    """Enhanced demand forecasting functionality with vendor filtering"""
    st.header("🔮 Demand Forecasting")
    st.markdown("AI-powered demand forecasting for optimal inventory planning and procurement decisions.")
    
    # Show filtered data info
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
    
    # Display cleaned data sample
    st.subheader("📋 Cleaned PO Data Sample")
    
    # Enhanced data display with better formatting
    display_df = df.head(10)
    if not display_df.empty:
        # Format numeric columns
        numeric_columns = display_df.select_dtypes(include=['float64', 'int64']).columns
        formatted_df = display_df.copy()
        
        for col in numeric_columns:
            if 'price' in col.lower() or 'total' in col.lower():
                formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
            elif 'qty' in col.lower() or 'quantity' in col.lower():
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        
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
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'Unit Price' in df.columns:
            avg_price = df['Unit Price'].mean()
            st.metric("Avg Unit Price", f"${avg_price:,.2f}")
    with col2:
        if 'Qty Delivered' in df.columns:
            total_qty = df['Qty Delivered'].sum()
            st.metric("Total Quantity", f"{total_qty:,.0f}")
    with col3:
        if 'Unit Price' in df.columns and 'Qty Delivered' in df.columns:
            total_value = (df['Unit Price'] * df['Qty Delivered']).sum()
            st.metric("Total Value", f"${total_value:,.0f}")
    with col4:
        if len(df) > 0:
            avg_order_size = df['Qty Delivered'].mean() if 'Qty Delivered' in df.columns else 0
            st.metric("Avg Order Size", f"{avg_order_size:,.1f}")
    
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
                
                # Enhanced plotting with multiple visualizations
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Main forecast plot
                ts.plot(label="Historical Demand", ax=ax1, linewidth=2, color='#1f77b4')
                forecast.plot(label="Forecasted Demand", linestyle="--", ax=ax1, linewidth=2, color='#ff7f0e')
                
                ax1.set_title("30-Day Demand Forecast", fontsize=14, fontweight='bold')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Quantity")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Add forecast statistics
                forecast_stats = f"Avg Daily: {forecast.mean():.1f}\nTotal: {forecast.sum():.0f}\nPeak: {forecast.max():.1f}"
                ax1.text(0.02, 0.98, forecast_stats, transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                # Historical trend analysis
                if len(ts) > 7:
                    ts.rolling(window=7).mean().plot(ax=ax2, label="7-day Moving Average", color='green', linewidth=2)
                    ax2.set_title("Historical Trend Analysis")
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("Quantity (7-day avg)")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                # Demand distribution
                ts.hist(ax=ax3, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.set_title("Historical Demand Distribution")
                ax3.set_xlabel("Daily Demand")
                ax3.set_ylabel("Frequency")
                ax3.grid(True, alpha=0.3)
                
                # Forecast vs Historical comparison
                comparison_data = pd.DataFrame({
                    'Historical_Avg': [ts.mean()],
                    'Forecast_Avg': [forecast.mean()],
                    'Historical_Peak': [ts.max()],
                    'Forecast_Peak': [forecast.max()]
                })
                
                x_pos = range(len(comparison_data.columns))
                ax4.bar(x_pos, comparison_data.iloc[0], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                ax4.set_title("Historical vs Forecast Comparison")
                ax4.set_ylabel("Quantity")
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(['Hist Avg', 'Fcst Avg', 'Hist Peak', 'Fcst Peak'], rotation=45)
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
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
                
                # Format and display with styling
                st.dataframe(
                    forecast_df.style.format({
                        'Forecasted_Quantity': '{:.1f}',
                        'Lower_Bound': '{:.1f}',
                        'Upper_Bound': '{:.1f}',
                        'Cumulative_Forecast': '{:.0f}'
                    }).background_gradient(subset=['Forecasted_Quantity'], cmap='Blues'),
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
        'Unit Price': [25.50, 1150.00, 75.25],
        'Qty Delivered': [100, 2, 50],
        'Line Total': [2550.00, 2300.00, 3762.50],
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
        """)
    
    with col2:
        st.markdown("""
        **🔧 Optional Columns:**
        - 📝 **Item Description** - Detailed item description
        - 🏪 **W/H** - Warehouse or location code
        - 💵 **Line Total** - Total amount (auto-calculated if missing)
        - 📋 **Category** - Item category classification
        - 🚚 **Lead Time** - Delivery lead time in days
        """)
    
    # Module overview with detailed descriptions
    st.subheader("🛠️ Analytics Modules Overview")
    
    modules_detailed = [
        {
            "name": "🔮 Demand Forecasting",
            "description": "AI-powered demand forecasting with time series analysis",
            "features": ["30-90 day forecasts", "Trend analysis", "Seasonality detection", "Risk assessment"],
            "use_case": "Optimize inventory levels and prevent stockouts"
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
        """)
    
    with col2:
        st.markdown("""
        **🎯 Analysis Tips:**
        - Start with demand forecasting for quick wins
        - Use vendor selection to focus on key suppliers
        - Combine multiple modules for comprehensive insights
        - Regular data updates improve forecast accuracy
        - Export results for presentation to stakeholders
        """)
    
    # Sample data download
    st.subheader("📁 Sample Data Template")
    st.markdown("Download a sample CSV template to understand the expected data format:")
    
    # Create extended sample data
    extended_sample = {
        'Creation Date': pd.date_range('2024-01-01', periods=50, freq='D'),
        'Vendor Name': ['LLORI LLERENA', '33 Designs', 'AHMED ALI SALE', 'A.T. Kearney Sau', 'AAA WORLD WID'] * 10,
        'Item': [f'ITEM_{i:03d}' for i in range(1, 51)],
        'Item Description': ['Office Supplies', 'IT Equipment', 'Raw Materials', 'Professional Services', 'Facilities'] * 10,
        'Unit Price': [round(price, 2) for price in (50 + 200 * pd.Series(range(50)).apply(lambda x: x % 10) / 10)],
        'Qty Delivered': [int(qty) for qty in (10 + 90 * pd.Series(range(50)).apply(lambda x: (x % 7) / 7))],
        'W/H': ['Warehouse A', 'Warehouse B', 'Warehouse C'] * 17
    }
    
    extended_df = pd.DataFrame(extended_sample)
    extended_df['Line Total'] = extended_df['Unit Price'] * extended_df['Qty Delivered']
    
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
    """)

if __name__ == "__main__":
    main()
