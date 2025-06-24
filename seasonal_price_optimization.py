import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def display(df):
    """Enhanced Seasonal Price Optimization Module - Simplified Version"""
    st.header("🌟 Enhanced Seasonal Price Optimization")
    st.markdown("Advanced procurement intelligence with multi-dimensional analysis and actionable insights.")
    
    # Enhanced data validation
    required_columns = ['Creation Date', 'Unit Price', 'Item']
    optional_columns = ['Vendor', 'Region', 'Qty Delivered', 'Lead Time Days']
    
    missing_required = [col for col in required_columns if col not in df.columns]
    available_optional = [col for col in optional_columns if col in df.columns]
    
    if missing_required:
        st.error(f"Missing required columns: {', '.join(missing_required)}")
        return
    
    # Enhanced data preparation
    df_clean = prepare_enhanced_data(df, available_optional)
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Show enhanced data coverage summary
    show_enhanced_coverage(df_clean, available_optional)
    
    # Enhanced tabs with more comprehensive analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Advanced Patterns", 
        "🔮 Price Forecasting", 
        "🌍 Multi-Dimensional Analysis",
        "💰 Optimization Engine",
        "⚡ Action Center"
    ])
    
    with tab1:
        advanced_pattern_analysis(df_clean)
    
    with tab2:
        price_forecasting(df_clean)
    
    with tab3:
        multi_dimensional_analysis(df_clean, available_optional)
    
    with tab4:
        optimization_engine(df_clean, available_optional)
    
    with tab5:
        action_center(df_clean, available_optional)

def prepare_enhanced_data(df, available_optional):
    """Enhanced data preparation with validation and enrichment"""
    df_clean = df.copy()
    
    # Convert date and clean basic data
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date', 'Unit Price', 'Item'])
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    # Add comprehensive time components
    df_clean['Year'] = df_clean['Creation Date'].dt.year
    df_clean['Month'] = df_clean['Creation Date'].dt.month
    df_clean['Quarter'] = df_clean['Creation Date'].dt.quarter
    df_clean['Week'] = df_clean['Creation Date'].dt.isocalendar().week
    df_clean['DayOfYear'] = df_clean['Creation Date'].dt.dayofyear
    df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name()
    df_clean['Quarter_Name'] = 'Q' + df_clean['Quarter'].astype(str)
    df_clean['Season'] = df_clean['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Handle optional dimensions with defaults
    if 'Vendor' not in df_clean.columns:
        df_clean['Vendor'] = 'Unknown'
    if 'Region' not in df_clean.columns:
        df_clean['Region'] = 'Unknown'
    if 'Lead Time Days' not in df_clean.columns:
        df_clean['Lead Time Days'] = 30
    
    # Calculate enhanced metrics
    if 'Qty Delivered' in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    else:
        df_clean['Qty Delivered'] = 1
        df_clean['Line Total'] = df_clean['Unit Price']
    
    # Add rolling averages for trend analysis
    df_clean = df_clean.sort_values(['Item', 'Creation Date'])
    for window in [30, 90, 180]:  # 1, 3, 6 month rolling averages
        df_clean[f'Price_MA_{window}d'] = df_clean.groupby('Item')['Unit Price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return df_clean

def show_enhanced_coverage(df_clean, available_optional):
    """Display comprehensive data coverage analysis"""
    st.subheader("📈 Enhanced Data Coverage Analysis")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df_clean):,}")
        st.metric("Unique Items", f"{df_clean['Item'].nunique():,}")
    
    with col2:
        date_range = df_clean['Creation Date'].max() - df_clean['Creation Date'].min()
        st.metric("Date Range", f"{date_range.days} days")
        st.metric("Years Covered", f"{df_clean['Year'].nunique()}")
    
    with col3:
        if 'Vendor' in available_optional:
            st.metric("Unique Vendors", f"{df_clean['Vendor'].nunique()}")
        if 'Region' in available_optional:
            st.metric("Unique Regions", f"{df_clean['Region'].nunique()}")
    
    with col4:
        total_spend = df_clean['Line Total'].sum()
        st.metric("Total Spend", f"${total_spend:,.0f}")
        avg_price = df_clean['Unit Price'].mean()
        st.metric("Avg Unit Price", f"${avg_price:.2f}")
    
    # Data quality assessment
    st.subheader("🎯 Data Quality Assessment")
    
    quality_metrics = []
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        # Calculate quality score
        record_count_score = min(len(item_data) / 52, 1) * 25  # 52 weeks = full score
        date_range_score = min((item_data['Creation Date'].max() - item_data['Creation Date'].min()).days / 365, 1) * 25
        price_consistency_score = max(0, 25 - (item_data['Unit Price'].std() / item_data['Unit Price'].mean() * 100))
        completeness_score = 25  # Base score for having required fields
        
        total_quality_score = record_count_score + date_range_score + price_consistency_score + completeness_score
        
        quality_metrics.append({
            'Item': item,
            'Records': len(item_data),
            'Date_Range_Days': (item_data['Creation Date'].max() - item_data['Creation Date'].min()).days,
            'Price_Volatility_%': item_data['Unit Price'].std() / item_data['Unit Price'].mean() * 100,
            'Quality_Score': total_quality_score,
            'Quality_Grade': 'A' if total_quality_score >= 80 else 'B' if total_quality_score >= 60 else 'C'
        })
    
    quality_df = pd.DataFrame(quality_metrics)
    quality_df = quality_df.sort_values('Quality_Score', ascending=False)
    
    # Show quality distribution
    quality_counts = quality_df['Quality_Grade'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=quality_counts.values, names=quality_counts.index,
                    title="Data Quality Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Top Quality Items:**")
        st.dataframe(
            quality_df.head(10)[['Item', 'Records', 'Quality_Score', 'Quality_Grade']].style.format({
                'Quality_Score': '{:.0f}'
            }),
            use_container_width=True
        )

def advanced_pattern_analysis(df_clean):
    """Advanced seasonal pattern analysis"""
    st.subheader("📊 Advanced Seasonal Pattern Analysis")
    
    # Item selection with enhanced filtering
    col1, col2 = st.columns(2)
    
    with col1:
        min_records = st.slider("Minimum Records Required", 5, 50, 10)
    with col2:
        quality_filter = st.selectbox("Quality Filter", ['All', 'A Grade Only', 'B+ Grade'])
    
    # Filter items based on criteria
    eligible_items = []
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        if len(item_data) >= min_records:
            eligible_items.append(item)
    
    if not eligible_items:
        st.warning("No items meet the minimum record criteria.")
        return
    
    selected_item = st.selectbox("Select Item for Analysis", eligible_items)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item].copy()
        
        # Enhanced seasonal analysis
        st.subheader(f"🔍 Detailed Analysis: {selected_item}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = item_data['Unit Price'].mean()
            st.metric("Average Price", f"${avg_price:.2f}")
        
        with col2:
            price_volatility = item_data['Unit Price'].std() / avg_price * 100
            st.metric("Price Volatility", f"{price_volatility:.1f}%")
        
        with col3:
            total_spend = item_data['Line Total'].sum()
            st.metric("Total Spend", f"${total_spend:,.0f}")
        
        with col4:
            records_count = len(item_data)
            st.metric("Data Points", f"{records_count}")
        
        # Multi-level seasonal analysis
        st.subheader("📈 Multi-Level Seasonal Patterns")
        
        # Monthly patterns
        monthly_stats = item_data.groupby('Month_Name')['Unit Price'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).reset_index()
        
        # Order months correctly
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_stats['Month_Order'] = monthly_stats['Month_Name'].apply(
            lambda x: month_order.index(x) if x in month_order else 12
        )
        monthly_stats = monthly_stats.sort_values('Month_Order')
        
        # Create comprehensive visualization
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=monthly_stats['Month_Name'], y=monthly_stats['mean'],
            mode='lines+markers', name='Average Price',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Add min/max range
        fig.add_trace(go.Scatter(
            x=monthly_stats['Month_Name'], y=monthly_stats['max'],
            mode='lines', line=dict(width=0), showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['Month_Name'], y=monthly_stats['min'],
            mode='lines', line=dict(width=0), showlegend=False,
            fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
            name='Min-Max Range'
        ))
        
        fig.update_layout(
            title=f'Monthly Price Patterns - {selected_item}',
            xaxis_title='Month',
            yaxis_title='Unit Price ($)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal insights
        best_month = monthly_stats.loc[monthly_stats['mean'].idxmin(), 'Month_Name']
        worst_month = monthly_stats.loc[monthly_stats['mean'].idxmax(), 'Month_Name']
        price_range = monthly_stats['mean'].max() - monthly_stats['mean'].min()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Best Month:** {best_month}")
        with col2:
            st.warning(f"**Worst Month:** {worst_month}")
        with col3:
            st.error(f"**Price Range:** ${price_range:.2f}")
        
        # Quarterly analysis
        quarterly_stats = item_data.groupby('Quarter_Name')['Unit Price'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        fig = px.bar(quarterly_stats, x='Quarter_Name', y='mean',
                    title=f"Quarterly Price Patterns - {selected_item}",
                    labels={'mean': 'Average Unit Price ($)', 'Quarter_Name': 'Quarter'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis using rolling averages
        st.subheader("📈 Trend Analysis")
        
        # Create time series plot with multiple moving averages
        item_data_sorted = item_data.sort_values('Creation Date')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=item_data_sorted['Creation Date'], y=item_data_sorted['Unit Price'],
            mode='markers', name='Actual Prices',
            marker=dict(color='lightblue', size=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=item_data_sorted['Creation Date'], y=item_data_sorted['Price_MA_30d'],
            mode='lines', name='30-Day MA',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=item_data_sorted['Creation Date'], y=item_data_sorted['Price_MA_90d'],
            mode='lines', name='90-Day MA',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=f'Price Trends with Moving Averages - {selected_item}',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def price_forecasting(df_clean):
    """Simple price forecasting using statistical methods"""
    st.subheader("🔮 Price Forecasting")
    
    # Simple forecasting approach using historical patterns
    items_with_data = []
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        if len(item_data) >= 12:  # Need at least 12 data points
            items_with_data.append(item)
    
    if not items_with_data:
        st.warning("Not enough historical data for forecasting.")
        return
    
    selected_item = st.selectbox("Select Item for Forecasting", items_with_data)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item].copy()
        
        # Create monthly aggregation
        monthly_data = item_data.groupby(item_data['Creation Date'].dt.to_period('M')).agg({
            'Unit Price': 'mean',
            'Qty Delivered': 'sum'
        }).reset_index()
        
        monthly_data['Date'] = monthly_data['Creation Date'].dt.to_timestamp()
        monthly_data = monthly_data.sort_values('Date')
        
        # Simple forecasting using seasonal patterns and trends
        if len(monthly_data) >= 12:
            
            # Calculate seasonal indices
            monthly_data['Month'] = monthly_data['Date'].dt.month
            seasonal_indices = monthly_data.groupby('Month')['Unit Price'].mean()
            overall_mean = monthly_data['Unit Price'].mean()
            seasonal_indices = seasonal_indices / overall_mean
            
            # Simple trend calculation
            monthly_data['Period'] = range(len(monthly_data))
            trend_slope = np.polyfit(monthly_data['Period'], monthly_data['Unit Price'], 1)[0]
            
            # Generate forecasts
            forecast_months = 6
            last_period = monthly_data['Period'].max()
            last_date = monthly_data['Date'].max()
            
            forecasts = []
            for i in range(1, forecast_months + 1):
                future_date = last_date + pd.DateOffset(months=i)
                future_period = last_period + i
                
                # Apply trend
                trend_value = overall_mean + (trend_slope * future_period)
                
                # Apply seasonal adjustment
                seasonal_factor = seasonal_indices.get(future_date.month, 1.0)
                forecasted_price = trend_value * seasonal_factor
                
                forecasts.append({
                    'Date': future_date,
                    'Forecasted_Price': forecasted_price,
                    'Month': future_date.strftime('%B %Y')
                })
            
            forecast_df = pd.DataFrame(forecasts)
            
            # Display forecast
            st.subheader(f"📅 6-Month Price Forecast - {selected_item}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    forecast_df[['Month', 'Forecasted_Price']].style.format({
                        'Forecasted_Price': '${:.2f}'
                    }),
                    use_container_width=True
                )
            
            with col2:
                # Forecast visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=monthly_data['Date'], y=monthly_data['Unit Price'],
                    mode='lines+markers', name='Historical',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'], y=forecast_df['Forecasted_Price'],
                    mode='lines+markers', name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title='Price Forecast',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Forecast insights
            current_price = monthly_data['Unit Price'].iloc[-1]
            avg_forecast = forecast_df['Forecasted_Price'].mean()
            price_direction = "increasing" if avg_forecast > current_price else "decreasing"
            
            st.info(f"**Forecast Summary:** Prices are expected to be {price_direction} over the next 6 months.")

def multi_dimensional_analysis(df_clean, available_optional):
    """Enhanced multi-dimensional analysis"""
    st.subheader("🌍 Multi-Dimensional Analysis")
    
    # Vendor Analysis
    if 'Vendor' in available_optional and df_clean['Vendor'].nunique() > 1:
        st.subheader("🏢 Vendor Performance Analysis")
        
        vendor_metrics = []
        for vendor in df_clean['Vendor'].unique():
            vendor_data = df_clean[df_clean['Vendor'] == vendor]
            
            vendor_metrics.append({
                'Vendor': vendor,
                'Items': vendor_data['Item'].nunique(),
                'Total_Spend': vendor_data['Line Total'].sum(),
                'Avg_Price': vendor_data['Unit Price'].mean(),
                'Price_Volatility': vendor_data['Unit Price'].std() / vendor_data['Unit Price'].mean() * 100,
                'Records': len(vendor_data)
            })
        
        vendor_df = pd.DataFrame(vendor_metrics)
        vendor_df = vendor_df.sort_values('Total_Spend', ascending=False)
        
        st.dataframe(
            vendor_df.style.format({
                'Total_Spend': '${:,.0f}',
                'Avg_Price': '${:.2f}',
                'Price_Volatility': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        # Vendor comparison for specific items
        st.subheader("📊 Item-Level Vendor Comparison")
        
        # Find items available from multiple vendors
        item_vendor_counts = df_clean.groupby('Item')['Vendor'].nunique()
        multi_vendor_items = item_vendor_counts[item_vendor_counts > 1].index.tolist()
        
        if multi_vendor_items:
            selected_item_vendor = st.selectbox("Select Item for Vendor Comparison", multi_vendor_items)
            
            item_vendor_data = df_clean[df_clean['Item'] == selected_item_vendor]
            vendor_comparison = item_vendor_data.groupby('Vendor')['Unit Price'].agg([
                'mean', 'std', 'count', 'min', 'max'
            ]).reset_index()
            
            fig = px.box(item_vendor_data, x='Vendor', y='Unit Price',
                        title=f"Price Distribution by Vendor - {selected_item_vendor}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Vendor savings opportunities
            best_vendor = vendor_comparison.loc[vendor_comparison['mean'].idxmin(), 'Vendor']
            worst_vendor = vendor_comparison.loc[vendor_comparison['mean'].idxmax(), 'Vendor']
            potential_savings = vendor_comparison['mean'].max() - vendor_comparison['mean'].min()
            
            st.success(f"**Best Vendor:** {best_vendor} | **Potential Savings:** ${potential_savings:.2f} per unit")
    
    # Regional Analysis
    if 'Region' in available_optional and df_clean['Region'].nunique() > 1:
        st.subheader("🗺️ Regional Analysis")
        
        regional_metrics = []
        for region in df_clean['Region'].unique():
            region_data = df_clean[df_clean['Region'] == region]
            
            regional_metrics.append({
                'Region': region,
                'Items': region_data['Item'].nunique(),
                'Total_Spend': region_data['Line Total'].sum(),
                'Avg_Price': region_data['Unit Price'].mean(),
                'Price_Volatility': region_data['Unit Price'].std() / region_data['Unit Price'].mean() * 100
            })
        
        regional_df = pd.DataFrame(regional_metrics)
        
        fig = px.bar(regional_df, x='Region', y='Total_Spend',
                    title="Total Spend by Region")
        st.plotly_chart(fig, use_container_width=True)

def optimization_engine(df_clean, available_optional):
    """Comprehensive optimization engine"""
    st.subheader("💰 Procurement Optimization Engine")
    
    # Calculate optimization scores for each item
    optimization_results = []
    
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        if len(item_data) >= 6:
            # Seasonal analysis
            monthly_avg = item_data.groupby('Month')['Unit Price'].mean()
            overall_avg = item_data['Unit Price'].mean()
            
            if len(monthly_avg) > 0:
                best_month = monthly_avg.idxmin()
                worst_month = monthly_avg.idxmax()
                seasonal_savings_pct = ((monthly_avg[worst_month] - monthly_avg[best_month]) / monthly_avg[worst_month]) * 100
            else:
                best_month = 1
                worst_month = 1
                seasonal_savings_pct = 0
            
            # Calculate metrics
            annual_spend = item_data['Line Total'].sum()
            potential_savings = annual_spend * (seasonal_savings_pct / 100)
            price_volatility = item_data['Unit Price'].std() / item_data['Unit Price'].mean() * 100
            
            # Optimization score calculation
            spend_score = min(np.log10(annual_spend + 1) * 10, 50)
            savings_score = min(seasonal_savings_pct * 2, 30)
            volatility_score = min(price_volatility, 20)
            
            optimization_score = spend_score + savings_score + volatility_score
            
            optimization_results.append({
                'Item': item,
                'Annual_Spend': annual_spend,
                'Best_Month': pd.to_datetime(f'2024-{best_month:02d}-01').strftime('%B'),
                'Worst_Month': pd.to_datetime(f'2024-{worst_month:02d}-01').strftime('%B'),
                'Seasonal_Savings_%': seasonal_savings_pct,
                'Potential_Savings_$': potential_savings,
                'Price_Volatility_%': price_volatility,
                'Optimization_Score': optimization_score,
                'Priority': 'High' if optimization_score >= 70 else 'Medium' if optimization_score >= 40 else 'Low'
            })
    
    if optimization_results:
        optimization_df = pd.DataFrame(optimization_results)
        optimization_df = optimization_df.sort_values('Optimization_Score', ascending=False)
        
        # Summary metrics
        total_potential_savings = optimization_df['Potential_Savings_$'].sum()
        total_spend = optimization_df['Annual_Spend'].sum()
        high_priority_items = len(optimization_df[optimization_df['Priority'] == 'High'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Potential Savings", f"${total_potential_savings:,.0f}")
        with col2:
            st.metric("Savings Rate", f"{(total_potential_savings/total_spend*100):.1f}%")
        with col3:
            st.metric("High Priority Items", high_priority_items)
        
        # Top opportunities
        st.subheader("🎯 Top Optimization Opportunities")
        
        display_cols = ['Item', 'Annual_Spend', 'Seasonal_Savings_%', 'Potential_Savings_$', 
                       'Best_Month', 'Priority', 'Optimization_Score']
        
        st.dataframe(
            optimization_df.head(15)[display_cols].style.format({
                'Annual_Spend': '${:,.0f}',
                'Seasonal_Savings_%': '{:.1f}%',
                'Potential_Savings_$': '${:,.0f}',
                'Optimization_Score': '{:.1f}'
            }),
            use_container_width=True
        )

def action_center(df_clean, available_optional):
    """Comprehensive action center with specific recommendations"""
    st.subheader("⚡ Procurement Action Center")
    
    current_month = datetime.now().month
    current_month_name = datetime.now().strftime('%B')
    
    # Immediate actions
    st.subheader("🚨 This Month's Actions")
    
    current_month_actions = []
    
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        if len(item_data) >= 6:
            monthly_avg = item_data.groupby('Month')['Unit Price'].mean()
            
            if current_month in monthly_avg.index:
                current_month_price = monthly_avg[current_month]
                min_price = monthly_avg.min()
                max_price = monthly_avg.max()
                
                # Determine action
                price_percentile = (current_month_price - min_price) / (max_price - min_price) * 100
                
                if price_percentile <= 25:
                    action = "🟢 BUY NOW"
                    priority = "High"
                elif price_percentile <= 50:
                    action = "🟡 CONSIDER"
                    priority = "Medium"
                else:
                    action = "🔴 WAIT"
                    priority = "Low"
                
                current_month_actions.append({
                    'Item': item,
                    'Action': action,
                    'Priority': priority,
                    'Current_Price_Percentile': price_percentile,
                    'Best_Month': pd.to_datetime(f'2024-{monthly_avg.idxmin():02d}-01').strftime('%B')
                })
    
    if current_month_actions:
        action_df = pd.DataFrame(current_month_actions)
        
        # Show high priority actions
        high_priority = action_df[action_df['Priority'] == 'High']
        if len(high_priority) > 0:
            st.success(f"🎉 {len(high_priority)} items are at optimal prices this month!")
            st.dataframe(high_priority[['Item', 'Action', 'Best_Month']], use_container_width=True)
        
        # Show all actions
        st.subheader("📋 All Current Month Actions")
        st.dataframe(
            action_df.style.format({'Current_Price_Percentile': '{:.0f}%'}),
            use_container_width=True
        )
    
    # Procurement calendar
    st.subheader("📅 Annual Procurement Calendar")
    
    calendar_data = []
    for month in range(1, 13):
        month_name = pd.to_datetime(f'2024-{month:02d}-01').strftime('%B')
        optimal_items = []
        
        for item in df_clean['Item'].unique():
            item_data = df_clean[df_clean['Item'] == item]
            if len(item_data) >= 6:
                monthly_avg = item_data.groupby('Month')['Unit Price'].mean()
                if len(monthly_avg) > 0:
                    best_month = monthly_avg.idxmin()
                    if month == best_month:
                        optimal_items.append(item)
        
        calendar_data.append({
            'Month': month_name,
            'Optimal_Items_Count': len(optimal_items),
            'Items': ', '.join(optimal_items[:3]) + ('...' if len(optimal_items) > 3 else '')
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    
    fig = px.bar(calendar_df, x='Month', y='Optimal_Items_Count',
                title="Optimal Purchase Timing Calendar",
                labels={'Optimal_Items_Count': 'Number of Items at Best Price'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    st.subheader("📥 Export Analysis")
    
    if st.button("Generate Complete Report"):
        # Create comprehensive report
        report_data = []
        
        for item in df_clean['Item'].unique():
            item_data = df_clean[df_clean['Item'] == item]
            if len(item_data) >= 6:
                monthly_avg = item_data.groupby('Month')['Unit Price'].mean()
                
                if len(monthly_avg) > 0:
                    report_data.append({
                        'Item': item,
                        'Records': len(item_data),
                        'Avg_Price': item_data['Unit Price'].mean(),
                        'Min_Price': item_data['Unit Price'].min(),
                        'Max_Price': item_data['Unit Price'].max(),
                        'Best_Month': pd.to_datetime(f'2024-{monthly_avg.idxmin():02d}-01').strftime('%B'),
                        'Worst_Month': pd.to_datetime(f'2024-{monthly_avg.idxmax():02d}-01').strftime('%B'),
                        'Annual_Spend': item_data['Line Total'].sum(),
                        'Seasonal_Savings_%': ((monthly_avg.max() - monthly_avg.min()) / monthly_avg.max()) * 100
                    })
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Seasonal Analysis Report",
                data=csv,
                file_name=f"seasonal_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Test the module
if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced Seasonal Price Optimization", layout="wide")
    
    # Sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='W')
    
    sample_data = []
    items = ['Widget A', 'Widget B', 'Widget C', 'Component X', 'Material Y']
    vendors = ['Supplier Inc', 'Global Parts', 'Premium Supply']
    regions = ['North America', 'Europe', 'Asia Pacific']
    
    for date in dates:
        for item in np.random.choice(items, size=np.random.randint(1, 3), replace=False):
            # Add realistic seasonal patterns
            month = date.month
            if item in ['Widget A', 'Component X']:
                seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)  # Peak in summer
            else:
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * (month - 9) / 12)  # Peak in winter
            
            base_prices = {'Widget A': 45, 'Widget B': 78, 'Widget C': 32, 
                          'Component X': 125, 'Material Y': 28}
            
            base_price = base_prices[item]
            price = base_price * seasonal_factor * np.random.uniform(0.85, 1.15)
            
            sample_data.append({
                'Creation Date': date,
                'Item': item,
                'Unit Price': round(price, 2),
                'Qty Delivered': np.random.randint(5, 100),
                'Vendor': np.random.choice(vendors),
                'Region': np.random.choice(regions),
                'Lead Time Days': np.random.randint(10, 45)
            })
    
    df = pd.DataFrame(sample_data)
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    
    display(df)
