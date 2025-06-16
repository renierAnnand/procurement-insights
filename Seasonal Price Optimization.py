import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_seasonal_indices(price_data):
    """Calculate seasonal indices for price optimization"""
    if len(price_data) < 4:  # Need at least 4 periods
        return None
    
    # Calculate overall average
    overall_avg = price_data.mean()
    
    # Calculate seasonal indices
    seasonal_indices = {}
    for period, price in price_data.items():
        seasonal_indices[period] = (price / overall_avg) * 100
    
    return seasonal_indices

def identify_optimal_timing(seasonal_indices, price_volatility):
    """Identify optimal buying periods"""
    if not seasonal_indices:
        return {}
    
    sorted_periods = sorted(seasonal_indices.items(), key=lambda x: x[1])
    
    return {
        'best_months': sorted_periods[:3],  # 3 cheapest months
        'worst_months': sorted_periods[-3:],  # 3 most expensive months
        'volatility_level': 'High' if price_volatility > 0.15 else 'Medium' if price_volatility > 0.08 else 'Low',
        'savings_potential': (sorted_periods[-1][1] - sorted_periods[0][1]) / 100
    }

def calculate_price_forecasting(historical_prices, forecast_periods=6):
    """Forecast future prices based on seasonal patterns"""
    if len(historical_prices) < 3:
        return pd.Series([historical_prices.mean()] * forecast_periods)
    
    # Simple seasonal forecast using historical averages
    historical_months = [pd.Timestamp(p).month for p in historical_prices.index]
    month_averages = {}
    
    for month, price in zip(historical_months, historical_prices.values):
        if month not in month_averages:
            month_averages[month] = []
        month_averages[month].append(price)
    
    # Calculate average price per month
    for month in month_averages:
        month_averages[month] = np.mean(month_averages[month])
    
    # Generate forecasts
    last_date = pd.Timestamp(historical_prices.index[-1])
    forecasts = []
    
    for i in range(forecast_periods):
        future_date = last_date + pd.DateOffset(months=i+1)
        future_month = future_date.month
        
        if future_month in month_averages:
            forecasts.append(month_averages[future_month])
        else:
            forecasts.append(historical_prices.mean())
    
    return pd.Series(forecasts)

def display(df):
    st.header("🌟 Seasonal Price Optimization")
    st.markdown("Analyze price patterns and identify optimal timing for purchases to maximize cost savings.")
    
    # Data validation
    required_columns = ['Item', 'Unit Price', 'Creation Date', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date'])
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Seasonality", "🎯 Optimal Timing", "📈 Price Forecasting", "💰 Savings Calculator", "📋 Recommendations"])
    
    with tab1:
        st.subheader("📊 Price Seasonality Analysis")
        
        # Item selection
        item_options = sorted(df_clean["Item"].dropna().unique())
        selected_item = st.selectbox("Select Item for Analysis", item_options, key="seasonal_item")
        
        if selected_item:
            item_df = df_clean[df_clean["Item"] == selected_item].copy()
            
            # Show item info
            item_desc = item_df['Item Description'].iloc[0] if 'Item Description' in item_df.columns else "N/A"
            total_spend = (item_df['Unit Price'] * item_df['Qty Delivered']).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Orders", len(item_df))
            with col2:
                st.metric("Total Spend", f"{total_spend:,.2f}")
            with col3:
                avg_price = item_df['Unit Price'].mean()
                st.metric("Average Price", f"{avg_price:.2f}")
            
            st.info(f"**Item Description:** {item_desc}")
            
            # Calculate monthly price trends
            item_df['Month'] = item_df['Creation Date'].dt.to_period('M')
            item_df['Quarter'] = item_df['Creation Date'].dt.to_period('Q')
            item_df['MonthName'] = item_df['Creation Date'].dt.month_name()
            
            # Monthly analysis
            monthly_prices = item_df.groupby('Month')['Unit Price'].mean()
            monthly_volumes = item_df.groupby('Month')['Qty Delivered'].sum()
            
            # Quarterly analysis
            quarterly_prices = item_df.groupby('Quarter')['Unit Price'].mean()
            quarterly_volumes = item_df.groupby('Quarter')['Qty Delivered'].sum()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📅 Monthly Price Trends")
                
                if len(monthly_prices) > 1:
                    fig = go.Figure()
                    
                    # Price line
                    fig.add_trace(go.Scatter(
                        x=[str(m) for m in monthly_prices.index],
                        y=monthly_prices.values,
                        mode='lines+markers',
                        name='Average Price',
                        yaxis='y',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Volume bars
                    fig.add_trace(go.Bar(
                        x=[str(m) for m in monthly_volumes.index],
                        y=monthly_volumes.values,
                        name='Volume',
                        yaxis='y2',
                        opacity=0.7,
                        marker=dict(color='lightblue')
                    ))
                    
                    fig.update_layout(
                        title=f"Price vs Volume Trends - Item {selected_item}",
                        xaxis_title="Month",
                        yaxis=dict(title="Price", side="left"),
                        yaxis2=dict(title="Volume", side="right", overlaying="y"),
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate price volatility
                    price_volatility = monthly_prices.std() / monthly_prices.mean()
                    st.metric("Price Volatility (CV)", f"{price_volatility:.2%}")
                else:
                    st.info("Insufficient data for monthly analysis.")
            
            with col2:
                st.subheader("📊 Seasonal Patterns")
                
                # Month name analysis for seasonal patterns
                if len(item_df) > 0:
                    month_analysis = item_df.groupby('MonthName').agg({
                        'Unit Price': ['mean', 'std', 'count'],
                        'Qty Delivered': 'sum'
                    }).round(2)
                    
                    month_analysis.columns = ['Avg_Price', 'Price_Std', 'Order_Count', 'Total_Volume']
                    month_analysis = month_analysis.reset_index()
                    
                    # Calculate seasonal index
                    overall_avg = month_analysis['Avg_Price'].mean()
                    month_analysis['Seasonal_Index'] = (month_analysis['Avg_Price'] / overall_avg * 100).round(1)
                    
                    # Sort by month order
                    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                  'July', 'August', 'September', 'October', 'November', 'December']
                    month_analysis['Month_Order'] = month_analysis['MonthName'].map(
                        {month: i for i, month in enumerate(month_order)}
                    )
                    month_analysis = month_analysis.sort_values('Month_Order')
                    
                    st.dataframe(
                        month_analysis[['MonthName', 'Avg_Price', 'Seasonal_Index', 'Order_Count', 'Total_Volume']].style.format({
                            'Avg_Price': '{:.2f}',
                            'Seasonal_Index': '{:.1f}%',
                            'Total_Volume': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Seasonal index visualization
                    fig = px.bar(
                        month_analysis,
                        x='MonthName',
                        y='Seasonal_Index',
                        title="Seasonal Price Index by Month",
                        labels={'Seasonal_Index': 'Seasonal Index (%)', 'MonthName': 'Month'},
                        color='Seasonal_Index',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig.add_hline(y=100, line_dash="dash", line_color="black", 
                                 annotation_text="Average (100%)")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Optimal Timing Analysis")
        
        # Multi-item analysis for timing
        selected_items = st.multiselect(
            "Select Items for Timing Analysis",
            options=sorted(df_clean["Item"].dropna().unique()),
            default=sorted(df_clean["Item"].dropna().unique())[:5],
            key="timing_items"
        )
        
        if selected_items:
            timing_results = []
            
            for item in selected_items:
                item_data = df_clean[df_clean["Item"] == item].copy()
                item_data['MonthName'] = item_data['Creation Date'].dt.month_name()
                
                monthly_prices = item_data.groupby('MonthName')['Unit Price'].mean()
                
                if len(monthly_prices) >= 2:
                    # Calculate seasonal indices
                    seasonal_indices = calculate_seasonal_indices(monthly_prices)
                    price_volatility = monthly_prices.std() / monthly_prices.mean()
                    
                    optimal_timing = identify_optimal_timing(seasonal_indices, price_volatility)
                    
                    item_desc = item_data['Item Description'].iloc[0] if 'Item Description' in item_data.columns else f"Item {item}"
                    annual_spend = (item_data['Unit Price'] * item_data['Qty Delivered']).sum()
                    
                    timing_results.append({
                        'Item': item,
                        'Description': item_desc[:40] + "..." if len(item_desc) > 40 else item_desc,
                        'Annual Spend': annual_spend,
                        'Best Month': optimal_timing['best_months'][0][0] if optimal_timing['best_months'] else 'N/A',
                        'Worst Month': optimal_timing['worst_months'][-1][0] if optimal_timing['worst_months'] else 'N/A',
                        'Savings Potential': optimal_timing['savings_potential'],
                        'Volatility': optimal_timing['volatility_level'],
                        'Price Volatility %': price_volatility * 100
                    })
            
            if timing_results:
                timing_df = pd.DataFrame(timing_results)
                timing_df = timing_df.sort_values('Savings Potential', ascending=False)
                
                st.subheader("📅 Optimal Buying Calendar")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    high_savings_items = len(timing_df[timing_df['Savings Potential'] > 0.1])
                    st.metric("High Savings Items", high_savings_items)
                with col2:
                    total_potential_savings = (timing_df['Annual Spend'] * timing_df['Savings Potential']).sum()
                    st.metric("Total Savings Potential", f"{total_potential_savings:,.0f}")
                with col3:
                    avg_volatility = timing_df['Price Volatility %'].mean()
                    st.metric("Average Price Volatility", f"{avg_volatility:.1f}%")
                
                # Display timing recommendations
                st.dataframe(
                    timing_df.style.format({
                        'Annual Spend': '{:,.0f}',
                        'Savings Potential': '{:.1%}',
                        'Price Volatility %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Best months visualization
                best_months_count = timing_df['Best Month'].value_counts()
                
                if len(best_months_count) > 0:
                    fig = px.bar(
                        x=best_months_count.index,
                        y=best_months_count.values,
                        title="Most Frequent Best Buying Months",
                        labels={'x': 'Month', 'y': 'Number of Items'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select items to analyze optimal timing.")
    
    with tab3:
        st.subheader("📈 Price Forecasting")
        
        # Item selection for forecasting
        forecast_item = st.selectbox(
            "Select Item for Price Forecasting",
            options=sorted(df_clean["Item"].dropna().unique()),
            key="forecast_item"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            forecast_periods = st.number_input("Forecast Periods (months)", min_value=1, max_value=12, value=6)
        with col2:
            confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
        
        if forecast_item:
            forecast_data = df_clean[df_clean["Item"] == forecast_item].copy()
            forecast_data['Month'] = forecast_data['Creation Date'].dt.to_period('M')
            
            monthly_prices = forecast_data.groupby('Month')['Unit Price'].mean()
            
            if len(monthly_prices) >= 2:
                # Generate price forecast
                price_forecast = calculate_price_forecasting(monthly_prices, forecast_periods)
                
                # Calculate confidence intervals (simplified)
                historical_std = monthly_prices.std()
                z_scores = {90: 1.65, 95: 1.96, 99: 2.58}
                z_score = z_scores[confidence_level]
                
                upper_bound = price_forecast + (z_score * historical_std)
                lower_bound = price_forecast - (z_score * historical_std)
                
                # Create forecast visualization
                fig = go.Figure()
                
                # Historical prices
                fig.add_trace(go.Scatter(
                    x=[str(p) for p in monthly_prices.index],
                    y=monthly_prices.values,
                    mode='lines+markers',
                    name='Historical Prices',
                    line=dict(color='blue', width=3)
                ))
                
                # Forecast
                future_periods = pd.period_range(
                    start=monthly_prices.index[-1] + 1,
                    periods=forecast_periods,
                    freq='M'
                )
                
                fig.add_trace(go.Scatter(
                    x=[str(p) for p in future_periods],
                    y=price_forecast.values,
                    mode='lines+markers',
                    name='Price Forecast',
                    line=dict(color='orange', width=3, dash='dash')
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=[str(p) for p in future_periods],
                    y=upper_bound.values,
                    mode='lines',
                    name=f'{confidence_level}% CI Upper',
                    line=dict(color='lightgray'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=[str(p) for p in future_periods],
                    y=lower_bound.values,
                    mode='lines',
                    name=f'{confidence_level}% CI Lower',
                    line=dict(color='lightgray'),
                    fill='tonexty',
                    fillcolor='rgba(211,211,211,0.3)',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title=f"Price Forecast - Item {forecast_item}",
                    xaxis_title="Month",
                    yaxis_title="Unit Price",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_price = monthly_prices.iloc[-1]
                    st.metric("Current Price", f"{current_price:.2f}")
                with col2:
                    forecast_avg = price_forecast.mean()
                    st.metric("Forecast Average", f"{forecast_avg:.2f}")
                with col3:
                    price_change = ((forecast_avg - current_price) / current_price) * 100
                    st.metric("Expected Change", f"{price_change:+.1f}%")
                
                # Forecast table
                forecast_table = pd.DataFrame({
                    'Month': [str(p) for p in future_periods],
                    'Forecast Price': price_forecast.values,
                    'Lower Bound': lower_bound.values,
                    'Upper Bound': upper_bound.values
                })
                
                st.write("**Detailed Forecast:**")
                st.dataframe(
                    forecast_table.style.format({
                        'Forecast Price': '{:.2f}',
                        'Lower Bound': '{:.2f}',
                        'Upper Bound': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.warning("Insufficient historical data for price forecasting.")
    
    with tab4:
        st.subheader("💰 Savings Calculator")
        
        # Calculate potential savings from optimal timing
        st.write("**Calculate potential savings by optimizing purchase timing:**")
        
        # User inputs
        col1, col2 = st.columns(2)
        with col1:
            annual_budget = st.number_input("Annual Procurement Budget", min_value=0, value=1000000, step=10000)
        with col2:
            optimization_rate = st.slider("Expected Optimization Success Rate", 0, 100, 70, help="Percentage of identified opportunities you can actually implement")
        
        if st.button("🔍 Calculate Portfolio Savings", type="primary"):
            with st.spinner("Calculating savings across all items..."):
                # Analyze all items for savings potential
                all_items_savings = []
                
                for item in df_clean['Item'].unique():
                    item_data = df_clean[df_clean["Item"] == item].copy()
                    
                    if len(item_data) < 2:
                        continue
                    
                    item_data['MonthName'] = item_data['Creation Date'].dt.month_name()
                    monthly_prices = item_data.groupby('MonthName')['Unit Price'].mean()
                    
                    if len(monthly_prices) >= 2:
                        annual_spend = (item_data['Unit Price'] * item_data['Qty Delivered']).sum()
                        price_range = monthly_prices.max() - monthly_prices.min()
                        savings_potential = (price_range / monthly_prices.mean()) if monthly_prices.mean() > 0 else 0
                        
                        all_items_savings.append({
                            'Item': item,
                            'Annual Spend': annual_spend,
                            'Savings Potential %': savings_potential * 100,
                            'Potential Savings': annual_spend * savings_potential
                        })
                
                if all_items_savings:
                    savings_df = pd.DataFrame(all_items_savings)
                    savings_df = savings_df.sort_values('Potential Savings', ascending=False)
                    
                    # Calculate totals
                    total_spend = savings_df['Annual Spend'].sum()
                    total_potential_savings = savings_df['Potential Savings'].sum()
                    adjusted_savings = total_potential_savings * (optimization_rate / 100)
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Analyzed Spend", f"{total_spend:,.0f}")
                    with col2:
                        st.metric("Potential Savings", f"{total_potential_savings:,.0f}")
                    with col3:
                        st.metric("Adjusted Savings", f"{adjusted_savings:,.0f}")
                    with col4:
                        roi_percent = (adjusted_savings / total_spend) * 100 if total_spend > 0 else 0
                        st.metric("ROI", f"{roi_percent:.1f}%")
                    
                    # Top opportunities
                    st.subheader("🎯 Top Savings Opportunities")
                    top_opportunities = savings_df.head(20)
                    
                    st.dataframe(
                        top_opportunities.style.format({
                            'Annual Spend': '{:,.0f}',
                            'Savings Potential %': '{:.1f}%',
                            'Potential Savings': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Savings distribution
                    fig = px.scatter(
                        savings_df,
                        x='Savings Potential %',
                        y='Annual Spend',
                        size='Potential Savings',
                        title="Savings Opportunities: Potential vs Current Spend",
                        labels={'Savings Potential %': 'Savings Potential (%)', 'Annual Spend': 'Annual Spend'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("No items found with sufficient data for savings calculation.")
    
    with tab5:
        st.subheader("📋 Strategic Recommendations")
        
        # Generate actionable recommendations
        st.write("**Seasonal Price Optimization Strategy:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 **Immediate Actions**")
            st.write("1. **Identify High-Impact Items**")
            st.write("   • Focus on items with >10% seasonal price variation")
            st.write("   • Prioritize high-spend categories first")
            st.write("")
            st.write("2. **Implement Buying Calendar**")
            st.write("   • Schedule purchases during optimal months")
            st.write("   • Set up price alerts for target items")
            st.write("")
            st.write("3. **Negotiate Seasonal Contracts**")
            st.write("   • Lock in prices during low-cost periods")
            st.write("   • Build buffer inventory when prices are favorable")
        
        with col2:
            st.markdown("#### 📈 **Long-term Strategy**")
            st.write("1. **Market Intelligence**")
            st.write("   • Monitor commodity price indices")
            st.write("   • Track supplier seasonal patterns")
            st.write("")
            st.write("2. **Demand Planning Integration**")
            st.write("   • Align procurement with production schedules")
            st.write("   • Balance inventory costs vs price savings")
            st.write("")
            st.write("3. **Supplier Diversification**")
            st.write("   • Reduce dependency on single suppliers")
            st.write("   • Develop regional supplier networks")
        
        # Implementation timeline
        st.subheader("🗓️ Implementation Timeline")
        
        timeline_data = [
            {"Phase": "Phase 1 (0-3 months)", "Activities": "Data analysis, item prioritization, quick wins identification"},
            {"Phase": "Phase 2 (3-6 months)", "Activities": "Buying calendar implementation, supplier negotiations, process setup"},
            {"Phase": "Phase 3 (6-12 months)", "Activities": "Full optimization, performance tracking, continuous improvement"}
        ]
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
        
        # Key success metrics
        st.subheader("📊 Success Metrics")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.write("**Financial Metrics:**")
            st.write("• Cost savings achieved")
            st.write("• Price variance reduction")
            st.write("• Budget performance improvement")
            st.write("• ROI on optimization efforts")
        
        with metrics_col2:
            st.write("**Operational Metrics:**")
            st.write("• Forecast accuracy improvement")
            st.write("• Inventory turnover optimization")
            st.write("• Supplier performance scores")
            st.write("• Process efficiency gains")
        
        # Export recommendations
        if st.button("📥 Export Strategic Plan"):
            recommendations = {
                'recommendation_type': ['Immediate Action', 'Immediate Action', 'Immediate Action', 
                                      'Long-term Strategy', 'Long-term Strategy', 'Long-term Strategy'],
                'category': ['High-Impact Items', 'Buying Calendar', 'Seasonal Contracts',
                           'Market Intelligence', 'Demand Planning', 'Supplier Diversification'],
                'description': [
                    'Focus on items with >10% seasonal price variation and high spend',
                    'Schedule purchases during optimal months with price alerts',
                    'Lock in prices during low-cost periods with buffer inventory',
                    'Monitor commodity indices and supplier seasonal patterns',
                    'Align procurement with production and balance inventory costs',
                    'Reduce supplier dependency and develop regional networks'
                ],
                'timeline': ['0-3 months', '0-3 months', '3-6 months', 
                           '3-6 months', '6-12 months', '6-12 months'],
                'priority': ['High', 'High', 'Medium', 'Medium', 'Medium', 'Low']
            }
            
            recommendations_df = pd.DataFrame(recommendations)
            csv = recommendations_df.to_csv(index=False)
            
            st.download_button(
                label="Download Strategic Plan",
                data=csv,
                file_name=f"seasonal_optimization_plan_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
