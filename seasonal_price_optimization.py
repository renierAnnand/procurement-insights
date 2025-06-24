import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Region-specific currency and formatting
REGION_CONFIG = {
    'North America': {'currency': 'USD', 'symbol': '$', 'format': '${:,.2f}'},
    'Europe': {'currency': 'EUR', 'symbol': '€', 'format': '€{:,.2f}'},
    'Asia Pacific': {'currency': 'USD', 'symbol': '$', 'format': '${:,.2f}'},
    'Latin America': {'currency': 'USD', 'symbol': '$', 'format': '${:,.2f}'},
    'Middle East': {'currency': 'USD', 'symbol': '$', 'format': '${:,.2f}'},
    'Africa': {'currency': 'USD', 'symbol': '$', 'format': '${:,.2f}'}
}

def format_currency(value, region):
    """Format currency based on region"""
    if pd.isna(value):
        return "N/A"
    config = REGION_CONFIG.get(region, REGION_CONFIG['North America'])
    return config['format'].format(value)

def detect_seasonal_pattern(data, min_data_points=12):
    """
    Detect strong seasonal patterns using coefficient of variation and statistical tests
    Returns pattern strength score (0-100) and confidence level
    """
    try:
        if len(data) < min_data_points:
            return 0, "Insufficient Data"
        
        # Ensure Unit Price is numeric
        data = data.copy()
        data['Unit Price'] = pd.to_numeric(data['Unit Price'], errors='coerce')
        data = data.dropna(subset=['Unit Price'])
        
        if len(data) == 0:
            return 0, "No Valid Data"
        
        # Calculate coefficient of variation
        price_mean = data['Unit Price'].mean()
        price_std = data['Unit Price'].std()
        cv = (price_std / price_mean * 100) if price_mean > 0 else 0
        
        # Statistical tests for seasonality
        statistical_significance = 0
        try:
            # Ensure Month is numeric
            data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
            data = data.dropna(subset=['Month'])
            
            # Kruskal-Wallis test for monthly differences
            monthly_groups = []
            for month_num, group in data.groupby('Month'):
                if len(group) >= 2:
                    monthly_groups.append(group['Unit Price'].values)
            
            if len(monthly_groups) >= 3:
                h_stat, p_value = stats.kruskal(*monthly_groups)
                statistical_significance = 1 - p_value if p_value < 0.05 else 0
        except Exception as e:
            statistical_significance = 0
        
        # Pattern strength calculation
        cv_score = min(cv / 20 * 50, 50)  # CV contributes up to 50 points
        stat_score = statistical_significance * 50  # Statistical significance contributes up to 50 points
        
        pattern_strength = cv_score + stat_score
        
        # Confidence level
        if pattern_strength >= 70:
            confidence = "High"
        elif pattern_strength >= 40:
            confidence = "Medium"
        elif pattern_strength >= 20:
            confidence = "Low"
        else:
            confidence = "None"
        
        return float(pattern_strength), str(confidence)
        
    except Exception as e:
        st.error(f"Error in pattern detection: {str(e)}")
        return 0, "Error"

def calculate_optimal_timing(data, region):
    """Calculate optimal buying months and potential savings"""
    try:
        # Ensure data types are correct
        data = data.copy()
        data['Unit Price'] = pd.to_numeric(data['Unit Price'], errors='coerce')
        data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
        data = data.dropna(subset=['Unit Price', 'Month', 'Month_Name'])
        
        if len(data) == 0:
            return {
                'monthly_stats': pd.DataFrame(),
                'optimal_months': pd.DataFrame(),
                'worst_months': pd.DataFrame(),
                'max_savings_pct': 0,
                'realistic_savings_pct': 0,
                'best_price': 0,
                'worst_price': 0,
                'avg_price': 0
            }
        
        monthly_stats = data.groupby(['Month', 'Month_Name'])['Unit Price'].agg([
            'mean', 'std', 'count', 'median'
        ]).reset_index()
        
        # Handle NaN values in std
        monthly_stats['std'] = monthly_stats['std'].fillna(0)
        
        # Calculate seasonal indices
        overall_mean = data['Unit Price'].mean()
        if overall_mean > 0:
            monthly_stats['Seasonal_Index'] = (monthly_stats['mean'] / overall_mean) * 100
            monthly_stats['Price_Advantage'] = 100 - monthly_stats['Seasonal_Index']
        else:
            monthly_stats['Seasonal_Index'] = 100
            monthly_stats['Price_Advantage'] = 0
        
        # Find optimal months (lowest prices)
        optimal_months = monthly_stats.nsmallest(min(3, len(monthly_stats)), 'Seasonal_Index')
        worst_months = monthly_stats.nlargest(min(3, len(monthly_stats)), 'Seasonal_Index')
        
        # Calculate potential savings
        best_price = monthly_stats['mean'].min() if len(monthly_stats) > 0 else 0
        worst_price = monthly_stats['mean'].max() if len(monthly_stats) > 0 else 0
        avg_price = monthly_stats['mean'].mean() if len(monthly_stats) > 0 else 0
        
        # Different savings scenarios
        max_savings_pct = ((worst_price - best_price) / worst_price * 100) if worst_price > 0 else 0
        realistic_savings_pct = ((avg_price - best_price) / avg_price * 100) if avg_price > 0 else 0
        
        return {
            'monthly_stats': monthly_stats,
            'optimal_months': optimal_months,
            'worst_months': worst_months,
            'max_savings_pct': float(max_savings_pct),
            'realistic_savings_pct': float(realistic_savings_pct),
            'best_price': float(best_price),
            'worst_price': float(worst_price),
            'avg_price': float(avg_price)
        }
        
    except Exception as e:
        st.error(f"Error in timing calculation: {str(e)}")
        return {
            'monthly_stats': pd.DataFrame(),
            'optimal_months': pd.DataFrame(),
            'worst_months': pd.DataFrame(),
            'max_savings_pct': 0,
            'realistic_savings_pct': 0,
            'best_price': 0,
            'worst_price': 0,
            'avg_price': 0
        }

def display(df):
    """Enhanced Seasonal Price Optimization Module"""
    st.header("🌟 Enhanced Seasonal Price Optimization")
    st.markdown("Advanced seasonal analysis with pattern detection, regional filtering, and precise savings estimation.")
    
    # Data validation
    required_columns = ['Creation Date', 'Unit Price', 'Item']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("This module requires: Creation Date, Unit Price, and Item columns")
        return
    
    # Regional filtering
    st.sidebar.header("🌍 Regional Filters")
    
    # Check if Region column exists
    has_region = 'Region' in df.columns
    if has_region:
        regions = ['All Regions'] + sorted(df['Region'].dropna().unique().tolist())
        selected_region = st.sidebar.selectbox("Select Region", regions)
        
        # Filter data by region
        if selected_region != 'All Regions':
            df_filtered = df[df['Region'] == selected_region].copy()
            currency_region = selected_region
        else:
            df_filtered = df.copy()
            currency_region = 'North America'  # Default
    else:
        st.sidebar.info("💡 Add a 'Region' column to enable regional filtering")
        df_filtered = df.copy()
        currency_region = st.sidebar.selectbox("Select Currency Region", list(REGION_CONFIG.keys()))
    
    # Data cleaning and preparation
    df_clean = df_filtered.copy()
    
    try:
        # Convert dates and clean data
        df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
        df_clean['Unit Price'] = pd.to_numeric(df_clean['Unit Price'], errors='coerce')
        
        # Remove invalid data
        df_clean = df_clean.dropna(subset=['Creation Date', 'Unit Price', 'Item'])
        df_clean = df_clean[df_clean['Unit Price'] > 0]
        
        if len(df_clean) == 0:
            st.warning("No valid data found for analysis in selected region.")
            return
        
        # Add date components with proper data types
        df_clean['Year'] = df_clean['Creation Date'].dt.year.astype(int)
        df_clean['Month'] = df_clean['Creation Date'].dt.month.astype(int)
        df_clean['Quarter'] = df_clean['Creation Date'].dt.quarter.astype(int)
        df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name().astype(str)
        
        # Ensure Item column is string
        df_clean['Item'] = df_clean['Item'].astype(str)
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return
    
    # Display regional summary
    if has_region and selected_region != 'All Regions':
        st.sidebar.success(f"📊 Analyzing {len(df_clean)} records from {selected_region}")
        st.sidebar.metric("Currency", REGION_CONFIG[currency_region]['currency'])
    
    # Advanced filtering options
    st.sidebar.subheader("📋 Advanced Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df_clean['Creation Date'].min().date(), df_clean['Creation Date'].max().date()),
        min_value=df_clean['Creation Date'].min().date(),
        max_value=df_clean['Creation Date'].max().date()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_clean = df_clean[
            (df_clean['Creation Date'].dt.date >= start_date) & 
            (df_clean['Creation Date'].dt.date <= end_date)
        ]
    
    # Minimum data points filter
    min_data_points = st.sidebar.slider("Minimum data points per item", 5, 50, 12)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Pattern Detection", 
        "📊 Price Analysis", 
        "⏰ Optimal Timing", 
        "💰 Savings Calculator"
    ])
    
    with tab1:
        st.subheader("🔍 Seasonal Pattern Detection")
        
        # Analyze all items for seasonal patterns
        pattern_results = []
        
        for item in df_clean['Item'].unique():
            try:
                item_data = df_clean[df_clean['Item'] == item].copy()
                if len(item_data) >= min_data_points:
                    pattern_strength, confidence = detect_seasonal_pattern(item_data)
                    
                    # Calculate metrics safely
                    unit_prices = pd.to_numeric(item_data['Unit Price'], errors='coerce').dropna()
                    if len(unit_prices) > 0:
                        avg_price = float(unit_prices.mean())
                        price_std = float(unit_prices.std())
                        cv_percent = float((price_std / avg_price * 100)) if avg_price > 0 else 0.0
                        price_range = float(unit_prices.max() - unit_prices.min())
                    else:
                        avg_price = 0.0
                        cv_percent = 0.0
                        price_range = 0.0
                    
                    pattern_results.append({
                        'Item': str(item),
                        'Data_Points': int(len(item_data)),
                        'Pattern_Strength': float(pattern_strength),
                        'Confidence': str(confidence),
                        'CV_Percent': cv_percent,
                        'Price_Range': price_range,
                        'Avg_Price': avg_price
                    })
            except Exception as e:
                st.warning(f"Error analyzing item {item}: {str(e)}")
                continue
        
        if pattern_results:
            pattern_df = pd.DataFrame(pattern_results)
            pattern_df = pattern_df.sort_values('Pattern_Strength', ascending=False)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                strong_patterns = len(pattern_df[pattern_df['Pattern_Strength'] >= 70])
                st.metric("Strong Patterns", strong_patterns)
            with col2:
                medium_patterns = len(pattern_df[pattern_df['Pattern_Strength'].between(40, 69)])
                st.metric("Medium Patterns", medium_patterns)
            with col3:
                avg_cv = pattern_df['CV_Percent'].mean()
                st.metric("Avg Price Volatility", f"{avg_cv:.1f}%")
            with col4:
                total_items = len(pattern_df)
                st.metric("Items Analyzed", total_items)
            
            # Pattern strength visualization
            fig = px.scatter(pattern_df, 
                           x='CV_Percent', 
                           y='Pattern_Strength',
                           size='Price_Range',
                           color='Confidence',
                           hover_data=['Item', 'Data_Points'],
                           title="Seasonal Pattern Detection Results",
                           labels={
                               'CV_Percent': 'Coefficient of Variation (%)',
                               'Pattern_Strength': 'Seasonal Pattern Strength (0-100)'
                           })
            fig.add_hline(y=70, line_dash="dash", line_color="green", 
                         annotation_text="Strong Pattern Threshold")
            fig.add_hline(y=40, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Pattern Threshold")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top seasonal items table
            st.subheader("🎯 Items with Strongest Seasonal Patterns")
            
            display_pattern_df = pattern_df.head(15).copy()
            display_pattern_df['Avg_Price_Formatted'] = display_pattern_df['Avg_Price'].apply(
                lambda x: format_currency(x, currency_region)
            )
            
            st.dataframe(
                display_pattern_df[[
                    'Item', 'Pattern_Strength', 'Confidence', 'CV_Percent', 
                    'Data_Points', 'Avg_Price_Formatted'
                ]].style.format({
                    'Pattern_Strength': '{:.1f}',
                    'CV_Percent': '{:.1f}%'
                }).background_gradient(subset=['Pattern_Strength'], cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            st.warning("No items found with sufficient data points for pattern analysis.")
    
    with tab2:
        st.subheader("📊 Detailed Price Analysis")
        
        # Item selection with pattern strength info
        if 'pattern_df' in locals() and len(pattern_df) > 0:
            # Sort items by pattern strength for selection
            items_with_patterns = pattern_df.set_index('Item')['Pattern_Strength'].to_dict()
            items = sorted(items_with_patterns.keys(), key=lambda x: items_with_patterns[x], reverse=True)
        else:
            items = sorted(df_clean['Item'].unique())
        
        selected_item = st.selectbox("Select Item for Detailed Analysis", items)
        
        if selected_item:
            item_data = df_clean[df_clean['Item'] == selected_item]
            
            if len(item_data) < min_data_points:
                st.warning(f"Not enough data points for {selected_item} (need at least {min_data_points})")
            else:
                # Display pattern strength if available
                if 'pattern_df' in locals():
                    item_pattern = pattern_df[pattern_df['Item'] == selected_item]
                    if len(item_pattern) > 0:
                        pattern_info = item_pattern.iloc[0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pattern Strength", f"{pattern_info['Pattern_Strength']:.1f}/100")
                        with col2:
                            st.metric("Confidence Level", pattern_info['Confidence'])
                        with col3:
                            st.metric("Price Volatility", f"{pattern_info['CV_Percent']:.1f}%")
                
                # Monthly analysis
                monthly_analysis = calculate_optimal_timing(item_data, currency_region)
                monthly_stats = monthly_analysis['monthly_stats']
                
                # Price metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_price = item_data['Unit Price'].mean()
                    st.metric("Average Price", format_currency(avg_price, currency_region))
                with col2:
                    best_month = monthly_stats.loc[monthly_stats['mean'].idxmin(), 'Month_Name']
                    best_price = monthly_stats['mean'].min()
                    st.metric("Best Month", best_month, format_currency(best_price, currency_region))
                with col3:
                    worst_month = monthly_stats.loc[monthly_stats['mean'].idxmax(), 'Month_Name']
                    worst_price = monthly_stats['mean'].max()
                    st.metric("Worst Month", worst_month, format_currency(worst_price, currency_region))
                with col4:
                    savings_potential = monthly_analysis['realistic_savings_pct']
                    st.metric("Savings Potential", f"{savings_potential:.1f}%")
                
                # Enhanced price trend chart with confidence intervals
                fig = go.Figure()
                
                # Add mean line
                fig.add_trace(go.Scatter(
                    x=monthly_stats['Month_Name'],
                    y=monthly_stats['mean'],
                    mode='lines+markers',
                    name='Average Price',
                    line=dict(width=3, color='blue'),
                    marker=dict(size=8)
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=monthly_stats['Month_Name'],
                    y=monthly_stats['mean'] + monthly_stats['std'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=monthly_stats['Month_Name'],
                    y=monthly_stats['mean'] - monthly_stats['std'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='±1 Std Dev',
                    fillcolor='rgba(0,100,80,0.2)'
                ))
                
                fig.update_layout(
                    title=f"Monthly Price Trends - {selected_item}",
                    xaxis_title="Month",
                    yaxis_title=f"Price ({REGION_CONFIG[currency_region]['currency']})",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal index chart
                fig2 = px.bar(monthly_stats, 
                             x='Month_Name', 
                             y='Price_Advantage',
                             color='Price_Advantage',
                             color_continuous_scale='RdYlGn',
                             title="Price Advantage by Month (% below average)",
                             labels={'Price_Advantage': 'Price Advantage (%)'})
                fig2.add_hline(y=0, line_dash="dash", line_color="black")
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("⏰ Optimal Purchase Timing")
        
        # Generate recommendations for all items with strong patterns
        if 'pattern_df' in locals():
            strong_pattern_items = pattern_df[pattern_df['Pattern_Strength'] >= 40]['Item'].tolist()
        else:
            strong_pattern_items = df_clean['Item'].unique()
        
        timing_recommendations = []
        
        for item in strong_pattern_items:
            try:
                item_data = df_clean[df_clean['Item'] == item].copy()
                if len(item_data) >= min_data_points:
                    timing_analysis = calculate_optimal_timing(item_data, currency_region)
                    
                    optimal_months = timing_analysis['optimal_months']
                    
                    if len(optimal_months) > 0:
                        # Safely get month names
                        best_month_1 = str(optimal_months.iloc[0]['Month_Name']) if len(optimal_months) > 0 else 'N/A'
                        best_month_2 = str(optimal_months.iloc[1]['Month_Name']) if len(optimal_months) > 1 else 'N/A'
                        best_month_3 = str(optimal_months.iloc[2]['Month_Name']) if len(optimal_months) > 2 else 'N/A'
                        
                        # Get pattern strength safely
                        pattern_strength = 0.0
                        if 'pattern_df' in locals() and len(pattern_df) > 0:
                            pattern_match = pattern_df[pattern_df['Item'] == item]
                            if len(pattern_match) > 0:
                                pattern_strength = float(pattern_match['Pattern_Strength'].iloc[0])
                        
                        timing_recommendations.append({
                            'Item': str(item),
                            'Best_Month_1': best_month_1,
                            'Best_Month_2': best_month_2,
                            'Best_Month_3': best_month_3,
                            'Avg_Price': float(item_data['Unit Price'].mean()),
                            'Best_Price': float(timing_analysis['best_price']),
                            'Savings_Potential': float(timing_analysis['realistic_savings_pct']),
                            'Pattern_Strength': pattern_strength
                        })
            except Exception as e:
                st.warning(f"Error processing timing for item {item}: {str(e)}")
                continue
        
        if timing_recommendations:
            timing_df = pd.DataFrame(timing_recommendations)
            timing_df = timing_df.sort_values('Savings_Potential', ascending=False)
            
            # Current month recommendations
            current_month = datetime.now().strftime('%B')
            current_month_items = timing_df[
                (timing_df['Best_Month_1'] == current_month) |
                (timing_df['Best_Month_2'] == current_month) |
                (timing_df['Best_Month_3'] == current_month)
            ]
            
            if len(current_month_items) > 0:
                st.success(f"🎉 **Perfect Timing!** {len(current_month_items)} items are optimal to buy in {current_month}")
                
                current_display = current_month_items.copy()
                current_display['Avg_Price_Formatted'] = current_display['Avg_Price'].apply(
                    lambda x: format_currency(x, currency_region)
                )
                current_display['Best_Price_Formatted'] = current_display['Best_Price'].apply(
                    lambda x: format_currency(x, currency_region)
                )
                
                st.dataframe(
                    current_display[[
                        'Item', 'Avg_Price_Formatted', 'Best_Price_Formatted', 'Savings_Potential'
                    ]].style.format({'Savings_Potential': '{:.1f}%'}),
                    use_container_width=True
                )
            else:
                st.info(f"ℹ️ No optimal buying opportunities in {current_month}")
            
            # Full timing calendar
            st.subheader("📅 Complete Timing Calendar")
            
            # Create a calendar heatmap
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            
            calendar_data = []
            for month in months:
                month_items = timing_df[
                    (timing_df['Best_Month_1'] == month) |
                    (timing_df['Best_Month_2'] == month) |
                    (timing_df['Best_Month_3'] == month)
                ]
                
                calendar_data.append({
                    'Month': month,
                    'Optimal_Items': len(month_items),
                    'Total_Savings_Potential': month_items['Savings_Potential'].sum() if len(month_items) > 0 else 0
                })
            
            calendar_df = pd.DataFrame(calendar_data)
            
            fig = px.bar(calendar_df, 
                        x='Month', 
                        y='Optimal_Items',
                        color='Total_Savings_Potential',
                        color_continuous_scale='Greens',
                        title="Optimal Buying Calendar",
                        labels={'Optimal_Items': 'Number of Items', 'Total_Savings_Potential': 'Total Savings %'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed recommendations table
            st.subheader("📋 Detailed Timing Recommendations")
            
            display_timing_df = timing_df.copy()
            display_timing_df['Avg_Price_Formatted'] = display_timing_df['Avg_Price'].apply(
                lambda x: format_currency(x, currency_region)
            )
            display_timing_df['Best_Price_Formatted'] = display_timing_df['Best_Price'].apply(
                lambda x: format_currency(x, currency_region)
            )
            
            st.dataframe(
                display_timing_df[[
                    'Item', 'Best_Month_1', 'Best_Month_2', 'Best_Month_3', 
                    'Avg_Price_Formatted', 'Best_Price_Formatted', 'Savings_Potential'
                ]].style.format({'Savings_Potential': '{:.1f}%'}).background_gradient(
                    subset=['Savings_Potential'], cmap='RdYlGn'
                ),
                use_container_width=True
            )
        else:
            st.info("No items with sufficient seasonal patterns found for timing recommendations.")
    
    with tab4:
        st.subheader("💰 Advanced Savings Calculator")
        
        # Calculate comprehensive savings analysis
        if 'timing_df' in locals() and len(timing_df) > 0:
            
            # Enhanced savings calculation with spending data
            savings_analysis = []
            
            for _, item_row in timing_df.iterrows():
                try:
                    item = str(item_row['Item'])
                    item_data = df_clean[df_clean['Item'] == item].copy()
                    
                    # Calculate annual spending safely
                    annual_spend = 0.0
                    if 'Line Total' in item_data.columns:
                        line_totals = pd.to_numeric(item_data['Line Total'], errors='coerce').fillna(0)
                        annual_spend = float(line_totals.sum())
                    elif 'Qty Delivered' in item_data.columns:
                        unit_prices = pd.to_numeric(item_data['Unit Price'], errors='coerce').fillna(0)
                        quantities = pd.to_numeric(item_data['Qty Delivered'], errors='coerce').fillna(1)
                        annual_spend = float((unit_prices * quantities).sum())
                    else:
                        # Estimate based on average order frequency
                        unit_prices = pd.to_numeric(item_data['Unit Price'], errors='coerce').fillna(0)
                        avg_price = float(unit_prices.mean()) if len(unit_prices) > 0 else 0.0
                        orders_per_year = int(len(item_data))
                        annual_spend = avg_price * orders_per_year
                    
                    # Calculate different savings scenarios
                    savings_potential = float(item_row['Savings_Potential'])
                    realistic_savings = annual_spend * (savings_potential / 100.0)
                    conservative_savings = realistic_savings * 0.7  # 70% achievement rate
                    optimistic_savings = realistic_savings * 1.3   # 130% achievement rate
                    
                    # Get best months safely
                    best_month_1 = str(item_row['Best_Month_1'])
                    best_month_2 = str(item_row['Best_Month_2'])
                    best_months_str = f"{best_month_1}, {best_month_2}" if best_month_2 != 'N/A' else best_month_1
                    
                    savings_analysis.append({
                        'Item': item,
                        'Annual_Spend': float(annual_spend),
                        'Realistic_Savings': float(realistic_savings),
                        'Conservative_Savings': float(conservative_savings),
                        'Optimistic_Savings': float(optimistic_savings),
                        'Savings_Potential_%': savings_potential,
                        'Best_Months': best_months_str,
                        'Pattern_Strength': float(item_row.get('Pattern_Strength', 0.0))
                    })
                    
                except Exception as e:
                    st.warning(f"Error calculating savings for item {item_row.get('Item', 'Unknown')}: {str(e)}")
                    continue
            
            savings_df = pd.DataFrame(savings_analysis)
            savings_df = savings_df.sort_values('Realistic_Savings', ascending=False)
            
            # Summary dashboard
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_realistic = savings_df['Realistic_Savings'].sum()
                st.metric("Total Realistic Savings", format_currency(total_realistic, currency_region))
            with col2:
                total_conservative = savings_df['Conservative_Savings'].sum()
                st.metric("Conservative Estimate", format_currency(total_conservative, currency_region))
            with col3:
                total_optimistic = savings_df['Optimistic_Savings'].sum()
                st.metric("Optimistic Estimate", format_currency(total_optimistic, currency_region))
            with col4:
                avg_savings_pct = savings_df['Savings_Potential_%'].mean()
                st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
            
            # ROI Analysis
            st.subheader("📈 Return on Investment Analysis")
            
            # Assume implementation cost (can be customized)
            implementation_cost = st.number_input(
                f"Implementation Cost ({REGION_CONFIG[currency_region]['currency']})",
                min_value=0.0,
                value=50000.0,
                help="Estimated cost for implementing seasonal purchasing strategy"
            )
            
            payback_months = (implementation_cost / (total_realistic / 12)) if total_realistic > 0 else 0
            annual_roi = ((total_realistic - implementation_cost) / implementation_cost * 100) if implementation_cost > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Payback Period", f"{payback_months:.1f} months")
            with col2:
                st.metric("Annual ROI", f"{annual_roi:.1f}%")
            with col3:
                net_benefit = total_realistic - implementation_cost
                st.metric("Net Annual Benefit", format_currency(net_benefit, currency_region))
            
            # Savings by item visualization
            fig = px.bar(savings_df.head(15), 
                        x='Realistic_Savings', 
                        y='Item',
                        orientation='h',
                        color='Pattern_Strength',
                        color_continuous_scale='Viridis',
                        title="Top 15 Items by Savings Potential",
                        labels={'Realistic_Savings': f'Annual Savings ({REGION_CONFIG[currency_region]["currency"]})'})
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed savings table
            st.subheader("📊 Detailed Savings Breakdown")
            
            display_savings_df = savings_df.copy()
            for col in ['Annual_Spend', 'Realistic_Savings', 'Conservative_Savings', 'Optimistic_Savings']:
                display_savings_df[f'{col}_Formatted'] = display_savings_df[col].apply(
                    lambda x: format_currency(x, currency_region)
                )
            
            st.dataframe(
                display_savings_df[[
                    'Item', 'Annual_Spend_Formatted', 'Realistic_Savings_Formatted',
                    'Conservative_Savings_Formatted', 'Optimistic_Savings_Formatted',
                    'Savings_Potential_%', 'Best_Months'
                ]].style.format({'Savings_Potential_%': '{:.1f}%'}).background_gradient(
                    subset=['Savings_Potential_%'], cmap='RdYlGn'
                ),
                use_container_width=True
            )
            
            # Implementation roadmap
            st.subheader("🗺️ Implementation Strategy")
            
            # Prioritize items by impact and pattern strength
            savings_df['Priority_Score'] = (
                savings_df['Realistic_Savings'] * 0.7 + 
                savings_df['Pattern_Strength'] * 0.3
            )
            savings_df = savings_df.sort_values('Priority_Score', ascending=False)
            
            phase1_items = savings_df.head(5)
            phase2_items = savings_df.iloc[5:15] if len(savings_df) > 5 else pd.DataFrame()
            phase3_items = savings_df.iloc[15:] if len(savings_df) > 15 else pd.DataFrame()
            
            st.markdown(f"""
            **Phase 1 (Months 1-3): High-Impact Quick Wins**
            - Focus on top {len(phase1_items)} items with strongest patterns
            - Potential savings: {format_currency(phase1_items['Realistic_Savings'].sum(), currency_region)}
            - Items: {', '.join(phase1_items['Item'].head(3).tolist())}{'...' if len(phase1_items) > 3 else ''}
            
            **Phase 2 (Months 4-9): Systematic Expansion**
            - Extend to {len(phase2_items)} additional items
            - Additional savings: {format_currency(phase2_items['Realistic_Savings'].sum(), currency_region) if len(phase2_items) > 0 else format_currency(0, currency_region)}
            - Develop automated monitoring systems
            
            **Phase 3 (Months 10+): Advanced Optimization**
            - Remaining {len(phase3_items)} items with seasonal patterns
            - Complete optimization potential: {format_currency(phase3_items['Realistic_Savings'].sum(), currency_region) if len(phase3_items) > 0 else format_currency(0, currency_region)}
            - Integrate with ERP and procurement systems
            """)
            
            # Export functionality
            if st.button("📥 Export Complete Analysis"):
                # Combine all analysis results
                export_data = {
                    'Pattern_Analysis': pattern_df if 'pattern_df' in locals() else pd.DataFrame(),
                    'Timing_Recommendations': timing_df,
                    'Savings_Analysis': savings_df,
                    'Summary_Metrics': pd.DataFrame([{
                        'Region': currency_region if has_region else 'Global',
                        'Total_Items_Analyzed': len(savings_df),
                        'Total_Realistic_Savings': total_realistic,
                        'Average_Savings_Percent': avg_savings_pct,
                        'Implementation_Cost': implementation_cost,
                        'Payback_Months': payback_months,
                        'Annual_ROI_Percent': annual_roi
                    }])
                }
                
                # Create downloadable Excel file
                from io import BytesIO
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    for sheet_name, data in export_data.items():
                        if not data.empty:
                            data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                st.download_button(
                    label="Download Complete Analysis (Excel)",
                    data=output.getvalue(),
                    file_name=f"seasonal_analysis_{currency_region}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        else:
            st.info("Complete the pattern detection and timing analysis to see savings calculations.")

if __name__ == "__main__":
    # For testing standalone
    st.set_page_config(page_title="Enhanced Seasonal Price Optimization", layout="wide")
    
    # Sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='W')
    regions = ['North America', 'Europe', 'Asia Pacific']
    items = ['Widget A', 'Widget B', 'Widget C', 'Component X', 'Material Y']
    
    sample_data = []
    for date in dates:
        for item in np.random.choice(items, size=np.random.randint(1, 4), replace=False):
            # Create seasonal patterns
            month = date.month
            seasonal_multiplier = 1.0
            
            if item == 'Widget A':  # Strong winter pattern
                seasonal_multiplier = 1.3 if month in [12, 1, 2] else 0.8 if month in [6, 7, 8] else 1.0
            elif item == 'Widget B':  # Spring/Summer pattern
                seasonal_multiplier = 1.2 if month in [4, 5, 6] else 0.9 if month in [11, 12, 1] else 1.0
            
            base_price = np.random.uniform(50, 200)
            seasonal_price = base_price * seasonal_multiplier * np.random.normal(1.0, 0.1)
            
            sample_data.append({
                'Creation Date': date,
                'Item': item,
                'Unit Price': max(seasonal_price, 10),  # Minimum price
                'Qty Delivered': np.random.randint(1, 100),
                'Region': np.random.choice(regions)
            })
    
    df = pd.DataFrame(sample_data)
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    
    display(df)
