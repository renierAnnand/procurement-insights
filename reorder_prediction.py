import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def calculate_demand_statistics(demand_series):
    """Calculate comprehensive demand statistics"""
    return {
        'mean': demand_series.mean(),
        'std': demand_series.std(),
        'median': demand_series.median(),
        'min': demand_series.min(),
        'max': demand_series.max(),
        'cv': demand_series.std() / demand_series.mean() if demand_series.mean() > 0 else 0,
        'trend': calculate_trend(demand_series)
    }

def calculate_trend(demand_series):
    """Calculate demand trend using linear regression"""
    if len(demand_series) < 3:
        return 0
    
    x = np.arange(len(demand_series)).reshape(-1, 1)
    y = demand_series.values
    
    try:
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_[0]
    except:
        return 0

def calculate_reorder_points(demand_series, lead_time_days=30, service_level=0.95):
    """Calculate reorder points using multiple methods"""
    stats = calculate_demand_statistics(demand_series)
    
    # Method 1: Statistical (Mean + Safety Stock)
    z_score = 1.96 if service_level == 0.95 else 1.65  # 95% or 90% service level
    safety_stock = z_score * stats['std'] * np.sqrt(lead_time_days / 30)  # Adjust for lead time
    reorder_statistical = stats['mean'] + safety_stock
    
    # Method 2: Fixed Percentage (125% of average)
    reorder_fixed = stats['mean'] * 1.25
    
    # Method 3: Dynamic (considers trend)
    trend_adjustment = stats['trend'] * lead_time_days if stats['trend'] > 0 else 0
    reorder_dynamic = stats['mean'] + safety_stock + trend_adjustment
    
    # Method 4: Min-Max based
    reorder_minmax = stats['min'] + (stats['max'] - stats['min']) * 0.3
    
    return {
        'statistical': max(0, reorder_statistical),
        'fixed_percentage': max(0, reorder_fixed),
        'dynamic': max(0, reorder_dynamic),
        'min_max': max(0, reorder_minmax),
        'safety_stock': safety_stock,
        'stats': stats
    }

def simulate_inventory(demand_series, reorder_point, lead_time_days=30, initial_stock=None):
    """Simulate inventory levels to validate reorder point"""
    if initial_stock is None:
        initial_stock = reorder_point * 2
    
    inventory_levels = [initial_stock]
    stockouts = 0
    orders_placed = 0
    
    for demand in demand_series:
        current_inventory = inventory_levels[-1] - demand
        
        # Check for stockout
        if current_inventory < 0:
            stockouts += 1
            current_inventory = 0
        
        # Check if we need to reorder (simplified - immediate delivery for simulation)
        if current_inventory <= reorder_point:
            orders_placed += 1
            current_inventory += reorder_point * 1.5  # Order enough to last
        
        inventory_levels.append(current_inventory)
    
    return {
        'inventory_levels': inventory_levels[:-1],  # Remove last element
        'stockouts': stockouts,
        'orders_placed': orders_placed,
        'service_level': (len(demand_series) - stockouts) / len(demand_series) if len(demand_series) > 0 else 0
    }

def forecast_demand(demand_series, periods=6):
    """Simple demand forecasting using linear regression and moving average"""
    if len(demand_series) < 2:
        # For very limited data, use the available average
        avg_demand = demand_series.mean() if len(demand_series) > 0 else 0
        return pd.Series([avg_demand] * periods)
    
    # Linear trend forecast
    x = np.arange(len(demand_series)).reshape(-1, 1)
    y = demand_series.values
    
    try:
        model = LinearRegression()
        model.fit(x, y)
        
        future_x = np.arange(len(demand_series), len(demand_series) + periods).reshape(-1, 1)
        forecast = model.predict(future_x)
        
        # Ensure non-negative forecasts and apply smoothing for limited data
        forecast = np.maximum(forecast, 0)
        
        # For limited data, blend with historical average to reduce volatility
        if len(demand_series) <= 3:
            historical_avg = demand_series.mean()
            # Blend 70% forecast, 30% historical average
            forecast = forecast * 0.7 + historical_avg * 0.3
        
        return pd.Series(forecast)
    except:
        # Fallback to moving average or simple average
        if len(demand_series) >= 2:
            recent_avg = demand_series.tail(2).mean()
        else:
            recent_avg = demand_series.mean()
        
        return pd.Series([recent_avg] * periods)

def calculate_abc_classification(df):
    """Calculate ABC classification based on annual value"""
    if 'Line Total' not in df.columns:
        return df
    
    item_values = df.groupby('Item')['Line Total'].sum().sort_values(ascending=False)
    total_value = item_values.sum()
    
    cumulative_value = item_values.cumsum()
    cumulative_percent = (cumulative_value / total_value) * 100
    
    # ABC Classification
    a_items = cumulative_percent[cumulative_percent <= 80].index
    b_items = cumulative_percent[(cumulative_percent > 80) & (cumulative_percent <= 95)].index
    c_items = cumulative_percent[cumulative_percent > 95].index
    
    classification = {}
    for item in item_values.index:
        if item in a_items:
            classification[item] = 'A'
        elif item in b_items:
            classification[item] = 'B'
        else:
            classification[item] = 'C'
    
    return classification

def display(df):
    st.header("📈 Smart Reorder Point Prediction")
    st.markdown("Advanced inventory optimization using multiple prediction methods and demand analysis.")
    
    # Data validation
    required_columns = ['Item', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean and prepare data
    df_clean = df.dropna(subset=['Item', 'Qty Delivered'])
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # ABC Classification
    abc_classification = calculate_abc_classification(df_clean)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Item Analysis", "📊 Demand Forecasting", "🔄 Bulk Analysis", "📈 ABC Analysis", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Individual Item Analysis")
        
        # Item selection with search
        col1, col2 = st.columns([2, 1])
        
        with col1:
            item_options = sorted(df_clean["Item"].dropna().unique())
            selected_item = st.selectbox("Select Item for Analysis", item_options, key="item_analysis")
        
        with col2:
            # Show ABC classification
            if selected_item in abc_classification:
                abc_class = abc_classification[selected_item]
                color_map = {'A': 'red', 'B': 'orange', 'C': 'green'}
                st.markdown(f"**ABC Class:** <span style='color: {color_map[abc_class]}; font-weight: bold;'>{abc_class}</span>", unsafe_allow_html=True)
        
        if selected_item:
            # Filter data for selected item
            item_df = df_clean[df_clean["Item"] == selected_item].copy()
            
            # Item information
            item_desc = item_df['Item Description'].iloc[0] if 'Item Description' in item_df.columns else "N/A"
            total_orders = len(item_df)
            total_qty = item_df['Qty Delivered'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Orders", total_orders)
            with col2:
                st.metric("Total Quantity", f"{total_qty:,.1f}")
            with col3:
                avg_order_size = total_qty / total_orders if total_orders > 0 else 0
                st.metric("Avg Order Size", f"{avg_order_size:.1f}")
            
            st.info(f"**Item Description:** {item_desc}")
            
            # Prepare demand data
            item_df['Month'] = item_df['Creation Date'].dt.to_period("M")
            demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)
            
            if len(demand_by_month) < 2:
                st.warning("Insufficient data for analysis. Need at least 2 months of data.")
                return
            
            # Configuration
            col1, col2, col3 = st.columns(3)
            with col1:
                lead_time = st.number_input("Lead Time (days)", min_value=1, max_value=365, value=30)
            with col2:
                service_level = st.selectbox("Service Level", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{x:.0%}")
            with col3:
                forecast_periods = st.number_input("Forecast Periods", min_value=3, max_value=12, value=6)
            
            # Calculate reorder points
            reorder_results = calculate_reorder_points(demand_by_month, lead_time, service_level)
            
            # Display results
            st.subheader("🎯 Recommended Reorder Points")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Statistical Method", f"{reorder_results['statistical']:.1f}")
            with col2:
                st.metric("Fixed % Method", f"{reorder_results['fixed_percentage']:.1f}")
            with col3:
                st.metric("Dynamic Method", f"{reorder_results['dynamic']:.1f}")
            with col4:
                st.metric("Min-Max Method", f"{reorder_results['min_max']:.1f}")
            
            # Demand statistics
            st.subheader("📊 Demand Analysis")
            stats = reorder_results['stats']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Demand Statistics:**")
                st.write(f"• Mean: {stats['mean']:.2f}")
                st.write(f"• Standard Deviation: {stats['std']:.2f}")
                st.write(f"• Coefficient of Variation: {stats['cv']:.2%}")
                st.write(f"• Trend: {stats['trend']:.2f} units/month")
            
            with col2:
                st.write("**Safety Stock:**")
                st.metric("Safety Stock", f"{reorder_results['safety_stock']:.1f}")
                
                volatility = "High" if stats['cv'] > 0.5 else "Medium" if stats['cv'] > 0.2 else "Low"
                st.write(f"**Demand Volatility:** {volatility}")
            
            # Demand visualization
            st.subheader("📈 Historical Demand & Forecast")
            
            # Forecast future demand
            forecast = forecast_demand(demand_by_month, forecast_periods)
            
            # Create comprehensive chart
            fig = go.Figure()
            
            # Historical demand
            fig.add_trace(go.Scatter(
                x=[str(period) for period in demand_by_month.index],
                y=demand_by_month.values,
                mode='lines+markers',
                name='Historical Demand',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            future_periods = pd.period_range(
                start=demand_by_month.index[-1] + 1,
                periods=forecast_periods,
                freq='M'
            )
            
            fig.add_trace(go.Scatter(
                x=[str(period) for period in future_periods],
                y=forecast.values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            # Reorder point lines
            all_periods = list(demand_by_month.index) + list(future_periods)
            reorder_line = [reorder_results['statistical']] * len(all_periods)
            
            fig.add_trace(go.Scatter(
                x=[str(period) for period in all_periods],
                y=reorder_line,
                mode='lines',
                name=f'Reorder Point ({reorder_results["statistical"]:.1f})',
                line=dict(color='red', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title=f"Demand Analysis for Item {selected_item}",
                xaxis_title="Month",
                yaxis_title="Quantity",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Simulation
            st.subheader("🔬 Inventory Simulation")
            
            simulation_results = {}
            methods = ['statistical', 'fixed_percentage', 'dynamic', 'min_max']
            method_names = ['Statistical', 'Fixed %', 'Dynamic', 'Min-Max']
            
            for method, name in zip(methods, method_names):
                sim_result = simulate_inventory(demand_by_month, reorder_results[method], lead_time)
                simulation_results[name] = sim_result
            
            # Display simulation results
            sim_df = pd.DataFrame({
                'Method': method_names,
                'Service Level': [simulation_results[name]['service_level'] for name in method_names],
                'Orders Placed': [simulation_results[name]['orders_placed'] for name in method_names],
                'Stockouts': [simulation_results[name]['stockouts'] for name in method_names]
            })
            
            st.dataframe(
                sim_df.style.format({
                    'Service Level': '{:.1%}',
                    'Orders Placed': '{:.0f}',
                    'Stockouts': '{:.0f}'
                }),
                use_container_width=True
            )
    
    with tab2:
        st.subheader("📊 Demand Forecasting Dashboard")
        
        # Show data summary first
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Items", df_clean['Item'].nunique())
        with col2:
            date_range = df_clean['Creation Date'].max() - df_clean['Creation Date'].min()
            st.metric("Data Range (Days)", f"{date_range.days}")
        with col3:
            months_of_data = len(df_clean.groupby(df_clean['Creation Date'].dt.to_period('M')))
            st.metric("Months of Data", months_of_data)
        
        # Auto-analyze top items by volume
        st.subheader("🎯 Top Items Analysis")
        
        # Get top items by total quantity
        top_items_by_qty = df_clean.groupby('Item')['Qty Delivered'].sum().nlargest(20)
        
        # Analyze data availability for top items
        item_analysis = []
        for item in top_items_by_qty.index:
            item_df = df_clean[df_clean["Item"] == item].copy()
            item_df['Month'] = item_df['Creation Date'].dt.to_period("M")
            demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)
            
            item_desc = item_df['Item Description'].iloc[0] if 'Item Description' in item_df.columns else "N/A"
            
            item_analysis.append({
                'Item': item,
                'Item Description': item_desc,
                'Total Qty': top_items_by_qty[item],
                'Months of Data': len(demand_by_month),
                'Avg Monthly Demand': demand_by_month.mean(),
                'Can Forecast': len(demand_by_month) >= 2  # Lowered threshold
            })
        
        analysis_df = pd.DataFrame(item_analysis)
        
        # Show analysis results
        st.dataframe(
            analysis_df.style.format({
                'Total Qty': '{:,.1f}',
                'Avg Monthly Demand': '{:.1f}'
            }),
            use_container_width=True
        )
        
        # Multi-item selection from analyzable items
        forecastable_items = analysis_df[analysis_df['Can Forecast']]['Item'].tolist()
        
        if len(forecastable_items) > 0:
            st.subheader("📈 Demand Forecasting")
            
            # Auto-select top 5 items or let user choose
            col1, col2 = st.columns(2)
            with col1:
                auto_select = st.checkbox("Auto-select top 5 items", value=True)
            with col2:
                forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=12, value=3)
            
            if auto_select:
                selected_items = forecastable_items[:5]
                st.info(f"Auto-selected top 5 items: {selected_items}")
            else:
                selected_items = st.multiselect(
                    "Select Items for Forecasting",
                    options=forecastable_items,
                    default=forecastable_items[:3] if len(forecastable_items) >= 3 else forecastable_items,
                    max_selections=10,
                    key="forecast_items"
                )
            
            if selected_items:
                forecast_results = {}
                
                # Process each selected item
                for item in selected_items:
                    item_df = df_clean[df_clean["Item"] == item].copy()
                    item_df['Month'] = item_df['Creation Date'].dt.to_period("M")
                    demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)
                    
                    # Generate forecast
                    forecast = forecast_demand(demand_by_month, forecast_periods)
                    forecast_results[item] = {
                        'historical': demand_by_month,
                        'forecast': forecast,
                        'total_forecast': forecast.sum(),
                        'item_desc': item_df['Item Description'].iloc[0] if 'Item Description' in item_df.columns else f"Item {item}"
                    }
                
                if forecast_results:
                    # Summary table
                    summary_data = []
                    for item, result in forecast_results.items():
                        historical_trend = 'Increasing' if len(result['historical']) > 1 and result['historical'].iloc[-1] > result['historical'].iloc[0] else 'Stable/Decreasing'
                        
                        summary_data.append({
                            'Item': item,
                            'Description': result['item_desc'][:50] + "..." if len(result['item_desc']) > 50 else result['item_desc'],
                            'Historical Avg': result['historical'].mean(),
                            f'{forecast_periods}-Period Forecast': result['total_forecast'],
                            'Monthly Avg Forecast': result['forecast'].mean(),
                            'Historical Trend': historical_trend,
                            'Data Points': len(result['historical'])
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    st.write("**Forecast Summary:**")
                    st.dataframe(
                        summary_df.style.format({
                            'Historical Avg': '{:.1f}',
                            f'{forecast_periods}-Period Forecast': '{:.1f}',
                            'Monthly Avg Forecast': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Forecast visualization
                    fig = go.Figure()
                    
                    colors = px.colors.qualitative.Set1
                    
                    for i, (item, result) in enumerate(forecast_results.items()):
                        color = colors[i % len(colors)]
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=[str(p) for p in result['historical'].index],
                            y=result['historical'].values,
                            mode='lines+markers',
                            name=f'Item {item} (Historical)',
                            line=dict(width=3, color=color),
                            marker=dict(size=8)
                        ))
                        
                        # Forecast data
                        future_periods = pd.period_range(
                            start=result['historical'].index[-1] + 1,
                            periods=forecast_periods,
                            freq='M'
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=[str(p) for p in future_periods],
                            y=result['forecast'].values,
                            mode='lines+markers',
                            name=f'Item {item} (Forecast)',
                            line=dict(width=3, color=color, dash='dash'),
                            marker=dict(size=8, symbol='diamond')
                        ))
                    
                    fig.update_layout(
                        title=f"Demand Forecast for Top {len(selected_items)} Items",
                        xaxis_title="Month",
                        yaxis_title="Quantity Delivered",
                        height=600,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional insights
                    st.subheader("🔍 Forecast Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Items with increasing demand
                        increasing_items = [item for item, result in forecast_results.items() 
                                          if result['forecast'].mean() > result['historical'].mean()]
                        
                        if increasing_items:
                            st.write("**📈 Items with Increasing Demand:**")
                            for item in increasing_items:
                                historical_avg = forecast_results[item]['historical'].mean()
                                forecast_avg = forecast_results[item]['forecast'].mean()
                                increase_pct = ((forecast_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
                                st.write(f"• Item {item}: +{increase_pct:.1f}% increase expected")
                        else:
                            st.write("**📉 No items showing significant demand increase**")
                    
                    with col2:
                        # Volatility analysis
                        volatile_items = []
                        for item, result in forecast_results.items():
                            if len(result['historical']) > 1:
                                cv = result['historical'].std() / result['historical'].mean() if result['historical'].mean() > 0 else 0
                                if cv > 0.5:  # High volatility
                                    volatile_items.append((item, cv))
                        
                        if volatile_items:
                            st.write("**⚠️ High Volatility Items (CV > 50%):**")
                            for item, cv in sorted(volatile_items, key=lambda x: x[1], reverse=True):
                                st.write(f"• Item {item}: {cv:.1%} volatility")
                        else:
                            st.write("**✅ All items show stable demand patterns**")
                    
                    # Export forecast data
                    if st.button("📥 Export Forecast Data"):
                        export_data = []
                        for item, result in forecast_results.items():
                            # Historical data
                            for period, qty in result['historical'].items():
                                export_data.append({
                                    'Item': item,
                                    'Period': str(period),
                                    'Type': 'Historical',
                                    'Quantity': qty
                                })
                            
                            # Forecast data
                            future_periods = pd.period_range(
                                start=result['historical'].index[-1] + 1,
                                periods=forecast_periods,
                                freq='M'
                            )
                            for period, qty in zip(future_periods, result['forecast']):
                                export_data.append({
                                    'Item': item,
                                    'Period': str(period),
                                    'Type': 'Forecast',
                                    'Quantity': qty
                                })
                        
                        export_df = pd.DataFrame(export_data)
                        csv = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download Forecast Data",
                            data=csv,
                            file_name=f"demand_forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.warning("No items could be processed for forecasting.")
            else:
                st.info("Select items to generate demand forecasts.")
        else:
            st.warning("No items have sufficient data for forecasting. Items need at least 2 months of historical data.")
    
    with tab3:
        st.subheader("🔄 Bulk Reorder Point Analysis")
        
        # Configuration for bulk analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            bulk_lead_time = st.number_input("Default Lead Time (days)", min_value=1, value=30, key="bulk_lead_time")
        with col2:
            bulk_service_level = st.selectbox("Default Service Level", [0.90, 0.95, 0.99], index=1, key="bulk_service_level", format_func=lambda x: f"{x:.0%}")
        with col3:
            min_data_points = st.number_input("Min Data Points Required", min_value=2, value=3, key="min_data_points")
        
        if st.button("🔄 Calculate Bulk Reorder Points", type="primary"):
            with st.spinner("Calculating reorder points for all items..."):
                bulk_results = []
                
                # Progress bar
                progress_bar = st.progress(0)
                items = df_clean['Item'].unique()
                
                for i, item in enumerate(items):
                    progress_bar.progress((i + 1) / len(items))
                    
                    item_df = df_clean[df_clean["Item"] == item].copy()
                    item_df['Month'] = item_df['Creation Date'].dt.to_period("M")
                    demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)
                    
                    if len(demand_by_month) >= min_data_points:
                        reorder_results = calculate_reorder_points(demand_by_month, bulk_lead_time, bulk_service_level)
                        
                        item_desc = item_df['Item Description'].iloc[0] if 'Item Description' in item_df.columns else "N/A"
                        abc_class = abc_classification.get(item, 'C')
                        
                        bulk_results.append({
                            'Item': item,
                            'Item Description': item_desc,
                            'ABC Class': abc_class,
                            'Statistical ROP': reorder_results['statistical'],
                            'Dynamic ROP': reorder_results['dynamic'],
                            'Fixed % ROP': reorder_results['fixed_percentage'],
                            'Mean Demand': reorder_results['stats']['mean'],
                            'Demand Std': reorder_results['stats']['std'],
                            'CV': reorder_results['stats']['cv'],
                            'Trend': reorder_results['stats']['trend'],
                            'Safety Stock': reorder_results['safety_stock'],
                            'Data Points': len(demand_by_month)
                        })
                
                progress_bar.empty()
                
                if bulk_results:
                    bulk_df = pd.DataFrame(bulk_results)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Items Analyzed", len(bulk_df))
                    with col2:
                        st.metric("A-Class Items", len(bulk_df[bulk_df['ABC Class'] == 'A']))
                    with col3:
                        st.metric("High Volatility Items", len(bulk_df[bulk_df['CV'] > 0.5]))
                    with col4:
                        st.metric("Growing Trend Items", len(bulk_df[bulk_df['Trend'] > 0]))
                    
                    # Display results with filtering
                    st.subheader("📋 Reorder Point Results")
                    
                    # Filters
                    col1, col2 = st.columns(2)
                    with col1:
                        abc_filter = st.multiselect("Filter by ABC Class", ['A', 'B', 'C'], default=['A', 'B', 'C'])
                    with col2:
                        volatility_filter = st.selectbox("Volatility Filter", ["All", "Low (CV < 0.2)", "Medium (0.2 ≤ CV < 0.5)", "High (CV ≥ 0.5)"])
                    
                    # Apply filters
                    filtered_df = bulk_df[bulk_df['ABC Class'].isin(abc_filter)]
                    
                    if volatility_filter == "Low (CV < 0.2)":
                        filtered_df = filtered_df[filtered_df['CV'] < 0.2]
                    elif volatility_filter == "Medium (0.2 ≤ CV < 0.5)":
                        filtered_df = filtered_df[(filtered_df['CV'] >= 0.2) & (filtered_df['CV'] < 0.5)]
                    elif volatility_filter == "High (CV ≥ 0.5)":
                        filtered_df = filtered_df[filtered_df['CV'] >= 0.5]
                    
                    # Sort by ABC class and statistical ROP
                    filtered_df = filtered_df.sort_values(['ABC Class', 'Statistical ROP'], ascending=[True, False])
                    
                    st.dataframe(
                        filtered_df.style.format({
                            'Statistical ROP': '{:.1f}',
                            'Dynamic ROP': '{:.1f}',
                            'Fixed % ROP': '{:.1f}',
                            'Mean Demand': '{:.1f}',
                            'Demand Std': '{:.1f}',
                            'CV': '{:.2%}',
                            'Trend': '{:.2f}',
                            'Safety Stock': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Export option
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Reorder Points",
                        data=csv,
                        file_name=f"reorder_points_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Store in session state for other tabs
                    st.session_state['bulk_results'] = bulk_df
                
                else:
                    st.warning("No items met the minimum data requirements for analysis.")
    
    with tab4:
        st.subheader("📈 ABC Analysis & Inventory Strategy")
        
        if 'bulk_results' in st.session_state:
            bulk_df = st.session_state['bulk_results']
            
            # ABC distribution
            abc_counts = bulk_df['ABC Class'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=abc_counts.values,
                    names=abc_counts.index,
                    title="ABC Classification Distribution",
                    color_discrete_map={'A': '#ff6b6b', 'B': '#ffa726', 'C': '#66bb6a'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ABC class statistics
                abc_stats = bulk_df.groupby('ABC Class').agg({
                    'Statistical ROP': 'mean',
                    'CV': 'mean',
                    'Mean Demand': 'mean',
                    'Safety Stock': 'mean'
                }).round(2)
                
                st.write("**ABC Class Statistics:**")
                st.dataframe(abc_stats, use_container_width=True)
            
            # Inventory strategy recommendations
            st.subheader("💡 Inventory Strategy Recommendations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🔴 A-Class Items")
                a_items = bulk_df[bulk_df['ABC Class'] == 'A']
                st.write(f"**Count:** {len(a_items)} items")
                st.write("**Strategy:**")
                st.write("• Tight inventory control")
                st.write("• Daily monitoring")
                st.write("• Higher service levels (99%)")
                st.write("• JIT delivery when possible")
            
            with col2:
                st.markdown("#### 🟡 B-Class Items")
                b_items = bulk_df[bulk_df['ABC Class'] == 'B']
                st.write(f"**Count:** {len(b_items)} items")
                st.write("**Strategy:**")
                st.write("• Moderate control")
                st.write("• Weekly monitoring")
                st.write("• Standard service levels (95%)")
                st.write("• Economic order quantities")
            
            with col3:
                st.markdown("#### 🟢 C-Class Items")
                c_items = bulk_df[bulk_df['ABC Class'] == 'C']
                st.write(f"**Count:** {len(c_items)} items")
                st.write("**Strategy:**")
                st.write("• Simple controls")
                st.write("• Monthly monitoring")
                st.write("• Lower service levels (90%)")
                st.write("• Bulk ordering")
            
            # Risk analysis
            st.subheader("⚠️ Inventory Risk Analysis")
            
            # High volatility items
            high_vol_items = bulk_df[bulk_df['CV'] > 0.5]
            
            if len(high_vol_items) > 0:
                st.write("**High Volatility Items (CV > 50%):**")
                risk_display = high_vol_items[['Item', 'Item Description', 'ABC Class', 'CV', 'Statistical ROP']].copy()
                risk_display = risk_display.sort_values('CV', ascending=False)
                
                st.dataframe(
                    risk_display.style.format({
                        'CV': '{:.1%}',
                        'Statistical ROP': '{:.1f}'
                    }),
                    use_container_width=True
                )
            else:
                st.success("No high volatility items identified.")
        
        else:
            st.info("Run bulk analysis first to see ABC analysis.")
    
    with tab5:
        st.subheader("⚙️ Advanced Settings & Configuration")
        
        # Model parameters
        st.write("**Reorder Point Calculation Settings:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Service Level Guidelines:**")
            st.write("• 90% - Cost-sensitive items")
            st.write("• 95% - Standard items")
            st.write("• 99% - Critical/high-value items")
            
            st.write("**Lead Time Considerations:**")
            st.write("• Include safety buffer for delays")
            st.write("• Consider supplier reliability")
            st.write("• Account for internal processing time")
        
        with col2:
            st.write("**Demand Volatility Interpretation:**")
            st.write("• CV < 20%: Low volatility")
            st.write("• CV 20-50%: Medium volatility")
            st.write("• CV > 50%: High volatility")
            
            st.write("**Data Quality Requirements:**")
            st.write("• Minimum 3 months of data")
            st.write("• Regular demand patterns")
            st.write("• Clean, validated data")
        
        # Export configuration
        st.subheader("📤 Export & Integration Settings")
        
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
        include_charts = st.checkbox("Include Visualization Data")
        
        if st.button("🔧 Generate Configuration Template"):
            config_template = {
                'default_lead_time_days': 30,
                'default_service_level': 0.95,
                'abc_thresholds': {'A': 80, 'B': 95, 'C': 100},
                'volatility_thresholds': {'low': 0.2, 'high': 0.5},
                'min_data_points': 3,
                'forecast_periods': 6,
                'safety_stock_multiplier': 1.96
            }
            
            config_df = pd.DataFrame([config_template])
            csv = config_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Configuration",
                data=csv,
                file_name="reorder_config.csv",
                mime="text/csv"
            )
        
        # Data refresh settings
        st.subheader("🔄 Data Refresh Settings")
        
        st.write("**Recommended Refresh Frequency:**")
        st.write("• A-Class items: Daily")
        st.write("• B-Class items: Weekly")
        st.write("• C-Class items: Monthly")
        
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared successfully!")
