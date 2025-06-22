import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ReorderPointCalculator:
    """Advanced reorder point calculation with multiple methods"""
    
    def __init__(self, item_df, service_level=0.95):
        self.item_df = item_df.copy()
        self.service_level = service_level
        self.z_score = self._get_z_score(service_level)
        self.prepare_data()
    
    def _get_z_score(self, service_level):
        """Convert service level to Z-score"""
        z_scores = {0.50: 0.00, 0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 
                   0.95: 1.65, 0.97: 1.88, 0.99: 2.33, 0.995: 2.58}
        return z_scores.get(service_level, 1.65)
    
    def prepare_data(self):
        """Prepare and clean data for analysis"""
        self.item_df['Creation Date'] = pd.to_datetime(self.item_df['Creation Date'])
        self.item_df = self.item_df.sort_values('Creation Date')
        self.date_range = (self.item_df['Creation Date'].max() - 
                          self.item_df['Creation Date'].min()).days + 1
        
        # Create daily demand series
        self.daily_demand = self.item_df.groupby('Creation Date')['Qty Delivered'].sum()
        full_range = pd.date_range(
            start=self.item_df['Creation Date'].min(),
            end=self.item_df['Creation Date'].max(),
            freq='D'
        )
        self.daily_demand = self.daily_demand.reindex(full_range, fill_value=0)

def calculate_average_daily_demand(item_df):
    """Calculate average daily demand with multiple methods"""
    calculator = ReorderPointCalculator(item_df)
    
    # Method 1: Simple average
    simple_avg = calculator.daily_demand.mean()
    
    # Method 2: Weighted average (recent months weighted more)
    recent_data = calculator.daily_demand.tail(90)  # Last 3 months
    if len(recent_data) >= 30:
        weighted_avg = (recent_data.mean() * 0.7) + (simple_avg * 0.3)
    else:
        weighted_avg = simple_avg
    
    # Method 3: Trend-adjusted average
    if len(calculator.daily_demand) >= 60:
        slope = np.polyfit(range(len(calculator.daily_demand)), calculator.daily_demand, 1)[0]
        trend_adjusted = simple_avg + (slope * 30)  # Project 30 days ahead
    else:
        trend_adjusted = simple_avg
    
    return {
        'simple': max(simple_avg, 0),
        'weighted': max(weighted_avg, 0),
        'trend_adjusted': max(trend_adjusted, 0),
        'recommended': max(weighted_avg, 0)
    }

def estimate_lead_time(item_df):
    """Advanced lead time estimation with multiple scenarios"""
    item_df_sorted = item_df.copy().sort_values('Creation Date')
    item_df_sorted['Creation Date'] = pd.to_datetime(item_df_sorted['Creation Date'])
    
    # Calculate time differences between orders
    time_diffs = item_df_sorted['Creation Date'].diff().dropna()
    
    if len(time_diffs) == 0:
        return {'min': 7, 'avg': 14, 'max': 21, 'std': 3}
    
    # Convert to days and filter outliers
    days_between = time_diffs.dt.days
    q75, q25 = np.percentile(days_between, [75, 25])
    iqr = q75 - q25
    lower_bound = q25 - (1.5 * iqr)
    upper_bound = q75 + (1.5 * iqr)
    filtered_days = days_between[(days_between >= max(lower_bound, 1)) & 
                                (days_between <= upper_bound)]
    
    # Lead time scenarios (fraction of reorder cycle)
    reorder_cycle = filtered_days.mean() if len(filtered_days) > 0 else 14
    
    return {
        'min': max(reorder_cycle * 0.2, 3),
        'avg': max(reorder_cycle * 0.4, 7),
        'max': max(reorder_cycle * 0.6, 14),
        'std': filtered_days.std() * 0.3 if len(filtered_days) > 1 else 2
    }

def calculate_safety_stock(daily_demand, lead_time_data, z_score=1.65, method='advanced'):
    """Calculate safety stock using multiple methods"""
    if method == 'basic':
        return z_score * daily_demand['simple'] * lead_time_data['std']
    
    elif method == 'advanced':
        # Advanced method considering demand and lead time variability
        demand_std = np.std([daily_demand['simple']] * 30)  # Simulate demand variability
        lead_time_var = lead_time_data['std'] ** 2
        avg_lead_time = lead_time_data['avg']
        
        safety_stock = z_score * np.sqrt(
            (avg_lead_time * demand_std ** 2) + 
            (daily_demand['recommended'] ** 2 * lead_time_var)
        )
        return max(safety_stock, daily_demand['recommended'] * 2)
    
    elif method == 'conservative':
        return z_score * daily_demand['recommended'] * lead_time_data['max'] * 0.5

def calculate_reorder_point(daily_demand, lead_time_data, safety_stock):
    """Calculate reorder point with scenario analysis"""
    scenarios = {}
    
    # Optimistic scenario
    scenarios['optimistic'] = (daily_demand['simple'] * lead_time_data['min']) + (safety_stock * 0.5)
    
    # Most likely scenario
    scenarios['likely'] = (daily_demand['recommended'] * lead_time_data['avg']) + safety_stock
    
    # Conservative scenario
    scenarios['conservative'] = (daily_demand['trend_adjusted'] * lead_time_data['max']) + (safety_stock * 1.5)
    
    return scenarios

def detect_seasonality(daily_demand_series):
    """Detect seasonal patterns in demand"""
    if len(daily_demand_series) < 365:
        return {'has_seasonality': False, 'pattern': 'Insufficient data'}
    
    # Convert to monthly data
    monthly_data = daily_demand_series.resample('M').sum()
    
    if len(monthly_data) < 12:
        return {'has_seasonality': False, 'pattern': 'Need at least 12 months'}
    
    # Calculate coefficient of variation
    cv = monthly_data.std() / monthly_data.mean() if monthly_data.mean() > 0 else 0
    
    # Determine seasonality
    if cv > 0.5:
        peak_month = monthly_data.idxmax().month_name()
        low_month = monthly_data.idxmin().month_name()
        return {
            'has_seasonality': True,
            'pattern': f'High seasonality (CV: {cv:.2f})',
            'peak_month': peak_month,
            'low_month': low_month,
            'seasonal_factor': monthly_data.max() / monthly_data.mean()
        }
    elif cv > 0.25:
        return {'has_seasonality': True, 'pattern': f'Moderate seasonality (CV: {cv:.2f})'}
    else:
        return {'has_seasonality': False, 'pattern': f'Low seasonality (CV: {cv:.2f})'}

def create_advanced_visualizations(item_df, daily_demand_data, reorder_scenarios):
    """Create comprehensive visualizations"""
    calculator = ReorderPointCalculator(item_df)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Daily Demand Pattern', 'Demand Distribution', 
                       'Reorder Point Scenarios', 'Cumulative Demand'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Daily demand with moving averages
    dates = calculator.daily_demand.index
    fig.add_trace(
        go.Scatter(x=dates, y=calculator.daily_demand.values, 
                  name='Daily Demand', line=dict(width=1, color='lightblue')),
        row=1, col=1
    )
    
    # Add moving averages
    if len(calculator.daily_demand) >= 7:
        ma_7 = calculator.daily_demand.rolling(7).mean()
        fig.add_trace(
            go.Scatter(x=dates, y=ma_7.values, name='7-day MA', 
                      line=dict(width=2, color='orange')),
            row=1, col=1
        )
    
    if len(calculator.daily_demand) >= 30:
        ma_30 = calculator.daily_demand.rolling(30).mean()
        fig.add_trace(
            go.Scatter(x=dates, y=ma_30.values, name='30-day MA', 
                      line=dict(width=2, color='red')),
            row=1, col=1
        )
    
    # 2. Demand distribution histogram
    fig.add_trace(
        go.Histogram(x=calculator.daily_demand.values, name='Demand Distribution',
                    nbinsx=20, opacity=0.7),
        row=1, col=2
    )
    
    # 3. Reorder point scenarios
    scenarios = ['Optimistic', 'Likely', 'Conservative']
    values = [reorder_scenarios['optimistic'], reorder_scenarios['likely'], 
              reorder_scenarios['conservative']]
    colors = ['green', 'blue', 'red']
    
    fig.add_trace(
        go.Bar(x=scenarios, y=values, name='Reorder Points',
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # 4. Cumulative demand
    cumulative = calculator.daily_demand.cumsum()
    fig.add_trace(
        go.Scatter(x=dates, y=cumulative.values, name='Cumulative Demand',
                  fill='tonexty', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Comprehensive Demand Analysis")
    
    return fig

def calculate_abc_analysis(df):
    """Perform ABC analysis on all items"""
    item_summary = df.groupby('Item').agg({
        'Qty Delivered': 'sum',
        'Creation Date': 'count'
    }).rename(columns={'Creation Date': 'Order_Count'})
    
    # Calculate total value (assuming unit price = 1 for demonstration)
    item_summary['Total_Value'] = item_summary['Qty Delivered']
    item_summary = item_summary.sort_values('Total_Value', ascending=False)
    
    # Calculate cumulative percentage
    item_summary['Cumulative_Value'] = item_summary['Total_Value'].cumsum()
    item_summary['Cumulative_Pct'] = (item_summary['Cumulative_Value'] / 
                                     item_summary['Total_Value'].sum()) * 100
    
    # Assign ABC categories
    item_summary['ABC_Category'] = 'C'
    item_summary.loc[item_summary['Cumulative_Pct'] <= 80, 'ABC_Category'] = 'A'
    item_summary.loc[(item_summary['Cumulative_Pct'] > 80) & 
                    (item_summary['Cumulative_Pct'] <= 95), 'ABC_Category'] = 'B'
    
    return item_summary

def analyze_all_items(df, service_level=0.95, method='advanced'):
    """Perform bulk analysis on all items with priority ranking"""
    all_items = df["Item"].dropna().unique()
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, item in enumerate(all_items):
        status_text.text(f'Analyzing item {i+1}/{len(all_items)}: {item}')
        progress_bar.progress((i + 1) / len(all_items))
        
        try:
            item_df = df[df["Item"] == item].copy()
            
            if len(item_df) == 0:
                continue
                
            # Calculate metrics for this item
            daily_demand = calculate_average_daily_demand(item_df)
            lead_time_data = estimate_lead_time(item_df)
            
            calculator = ReorderPointCalculator(item_df, service_level)
            seasonality = detect_seasonality(calculator.daily_demand)
            
            safety_stock = calculate_safety_stock(
                daily_demand, lead_time_data, 
                calculator.z_score, method
            )
            
            reorder_scenarios = calculate_reorder_point(
                daily_demand, lead_time_data, safety_stock
            )
            
            # Calculate additional metrics for prioritization
            total_quantity = item_df['Qty Delivered'].sum()
            total_orders = len(item_df)
            date_range = (item_df['Creation Date'].max() - item_df['Creation Date'].min()).days + 1
            order_frequency = total_orders / max(date_range / 30, 1)  # Orders per month
            
            # Demand variability (coefficient of variation)
            demand_cv = calculator.daily_demand.std() / calculator.daily_demand.mean() if calculator.daily_demand.mean() > 0 else 0
            
            # Lead time risk score
            lead_time_risk = lead_time_data['std'] / lead_time_data['avg'] if lead_time_data['avg'] > 0 else 0
            
            # Priority score calculation (higher = more critical)
            priority_score = (
                (total_quantity / 1000) * 0.3 +  # Volume impact
                (daily_demand['recommended'] * 10) * 0.25 +  # Daily demand impact
                (1 / max(order_frequency, 0.1)) * 0.2 +  # Frequency impact (inverted)
                (demand_cv * 100) * 0.15 +  # Variability impact
                (lead_time_risk * 100) * 0.1  # Lead time risk impact
            )
            
            results.append({
                'Item': item,
                'Priority_Score': priority_score,
                'Daily_Demand': daily_demand['recommended'],
                'Lead_Time_Days': lead_time_data['avg'],
                'Lead_Time_Risk': lead_time_risk,
                'Safety_Stock': safety_stock,
                'Reorder_Point_Optimistic': reorder_scenarios['optimistic'],
                'Reorder_Point_Likely': reorder_scenarios['likely'],
                'Reorder_Point_Conservative': reorder_scenarios['conservative'],
                'Total_Quantity': total_quantity,
                'Total_Orders': total_orders,
                'Order_Frequency_Monthly': order_frequency,
                'Demand_Variability': demand_cv,
                'Seasonality': seasonality['pattern'],
                'Has_Seasonality': seasonality['has_seasonality'],
                'Data_Period_Days': date_range,
                'Service_Level': f"{service_level*100:.0f}%"
            })
            
        except Exception as e:
            st.warning(f"Could not analyze {item}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("No items could be analyzed successfully.")
        return None
    
    # Convert to DataFrame and sort by priority
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Priority_Score', ascending=False)
    
    # Add priority categories
    total_items = len(results_df)
    results_df['Priority_Category'] = 'Low'
    results_df.iloc[:int(total_items * 0.2), results_df.columns.get_loc('Priority_Category')] = 'Critical'
    results_df.iloc[int(total_items * 0.2):int(total_items * 0.5), results_df.columns.get_loc('Priority_Category')] = 'High'
    results_df.iloc[int(total_items * 0.5):int(total_items * 0.8), results_df.columns.get_loc('Priority_Category')] = 'Medium'
    
    return results_df

def display_bulk_analysis_results(results_df):
    """Display comprehensive bulk analysis results"""
    st.header("📊 Bulk Analysis Results - All Items")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items Analyzed", len(results_df))
    
    with col2:
        critical_items = len(results_df[results_df['Priority_Category'] == 'Critical'])
        st.metric("Critical Priority Items", critical_items)
    
    with col3:
        total_reorder_value = results_df['Reorder_Point_Likely'].sum()
        st.metric("Total Reorder Investment", f"{total_reorder_value:,.0f}")
    
    with col4:
        avg_lead_time = results_df['Lead_Time_Days'].mean()
        st.metric("Average Lead Time", f"{avg_lead_time:.1f} days")
    
    # Priority distribution
    st.subheader("🎯 Priority Distribution")
    priority_counts = results_df['Priority_Category'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        for priority in ['Critical', 'High', 'Medium', 'Low']:
            count = priority_counts.get(priority, 0)
            percentage = (count / len(results_df)) * 100
            color = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}[priority]
            st.write(f"{color} **{priority}**: {count} items ({percentage:.1f}%)")
    
    with col2:
        # Priority pie chart
        fig_pie = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            title="Items by Priority Category",
            color_discrete_map={
                'Critical': '#ff4444',
                'High': '#ff8800',
                'Medium': '#ffdd00',
                'Low': '#44ff44'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Risk analysis
    st.subheader("⚠️ Risk Analysis")
    
    high_risk_items = results_df[
        (results_df['Lead_Time_Risk'] > 0.5) | 
        (results_df['Demand_Variability'] > 0.5)
    ]
    
    seasonal_items = results_df[results_df['Has_Seasonality'] == True]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("High Risk Items", len(high_risk_items))
        if len(high_risk_items) > 0:
            st.write("Items with high lead time or demand variability:")
            risk_display = high_risk_items[['Item', 'Lead_Time_Risk', 'Demand_Variability']].head(5)
            st.dataframe(risk_display, use_container_width=True)
    
    with col2:
        st.metric("Seasonal Items", len(seasonal_items))
        if len(seasonal_items) > 0:
            st.write("Items with seasonal demand patterns:")
            seasonal_display = seasonal_items[['Item', 'Seasonality']].head(5)
            st.dataframe(seasonal_display, use_container_width=True)
    
    # Interactive data table
    st.subheader("📋 Detailed Results Table")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority_filter = st.multiselect(
            "Filter by Priority",
            ['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High', 'Medium', 'Low']
        )
    
    with col2:
        min_demand = st.number_input("Minimum Daily Demand", min_value=0.0, value=0.0)
    
    with col3:
        show_seasonal_only = st.checkbox("Show Seasonal Items Only")
    
    # Apply filters
    filtered_df = results_df[results_df['Priority_Category'].isin(priority_filter)]
    filtered_df = filtered_df[filtered_df['Daily_Demand'] >= min_demand]
    
    if show_seasonal_only:
        filtered_df = filtered_df[filtered_df['Has_Seasonality'] == True]
    
    # Display table with formatting
    display_df = filtered_df[[
        'Item', 'Priority_Category', 'Priority_Score', 
        'Daily_Demand', 'Lead_Time_Days', 'Safety_Stock',
        'Reorder_Point_Likely', 'Total_Quantity', 'Demand_Variability',
        'Order_Frequency_Monthly', 'Seasonality'
    ]].copy()
    
    # Format numeric columns
    display_df['Priority_Score'] = display_df['Priority_Score'].round(2)
    display_df['Daily_Demand'] = display_df['Daily_Demand'].round(2)
    display_df['Lead_Time_Days'] = display_df['Lead_Time_Days'].round(1)
    display_df['Safety_Stock'] = display_df['Safety_Stock'].round(2)
    display_df['Reorder_Point_Likely'] = display_df['Reorder_Point_Likely'].round(2)
    display_df['Demand_Variability'] = (display_df['Demand_Variability'] * 100).round(1)
    display_df['Order_Frequency_Monthly'] = display_df['Order_Frequency_Monthly'].round(2)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Advanced analytics
    st.subheader("📈 Advanced Analytics")
    
    # Correlation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot: Priority Score vs Daily Demand
        fig_scatter1 = px.scatter(
            results_df, 
            x='Daily_Demand', 
            y='Priority_Score',
            color='Priority_Category',
            size='Total_Quantity',
            hover_data=['Item', 'Lead_Time_Days'],
            title="Priority Score vs Daily Demand",
            color_discrete_map={
                'Critical': '#ff4444',
                'High': '#ff8800',
                'Medium': '#ffdd00',
                'Low': '#44ff44'
            }
        )
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        # Box plot: Reorder Points by Priority Category
        fig_box = px.box(
            results_df,
            x='Priority_Category',
            y='Reorder_Point_Likely',
            title="Reorder Points Distribution by Priority",
            color='Priority_Category',
            color_discrete_map={
                'Critical': '#ff4444',
                'High': '#ff8800',
                'Medium': '#ffdd00',
                'Low': '#44ff44'
            }
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Export options
    st.subheader("📤 Export Bulk Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Full export
        full_csv = results_df.to_csv(index=False)
        st.download_button(
            "📊 Download Full Analysis (CSV)",
            full_csv,
            file_name=f"bulk_reorder_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Critical items only
        critical_df = results_df[results_df['Priority_Category'] == 'Critical']
        if len(critical_df) > 0:
            critical_csv = critical_df.to_csv(index=False)
            st.download_button(
                "🔴 Download Critical Items (CSV)",
                critical_csv,
                file_name=f"critical_items_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Summary report
        summary_data = {
            'Metric': [
                'Total Items', 'Critical Items', 'High Risk Items', 
                'Seasonal Items', 'Average Lead Time', 'Total Reorder Value'
            ],
            'Value': [
                len(results_df), len(results_df[results_df['Priority_Category'] == 'Critical']),
                len(high_risk_items), len(seasonal_items),
                f"{results_df['Lead_Time_Days'].mean():.1f} days",
                f"{results_df['Reorder_Point_Likely'].sum():,.0f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            "📋 Download Summary Report (CSV)",
            summary_csv,
            file_name=f"inventory_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

def display(df):
    """Enhanced main Streamlit interface"""
    st.title("🎯 Smart Reorder Point Prediction System")
    st.markdown("Advanced procurement analytics with multiple calculation methods")
    
    # Validate data
    required_cols = ['Item', 'Qty Delivered', 'Creation Date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return
    
    if df["Item"].dropna().empty:
        st.error("❌ No items found in the dataset.")
        return
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    service_level = st.sidebar.selectbox(
        "Service Level", 
        [0.85, 0.90, 0.95, 0.97, 0.99],
        index=2,
        format_func=lambda x: f"{x*100:.0f}%"
    )
    
    calculation_method = st.sidebar.selectbox(
        "Safety Stock Method",
        ['basic', 'advanced', 'conservative'],
        index=1
    )
    
    show_abc_analysis = st.sidebar.checkbox("Show ABC Analysis", value=True)
    
    # ABC Analysis sidebar
    if show_abc_analysis:
        st.sidebar.subheader("📊 ABC Analysis")
        abc_summary = calculate_abc_analysis(df)
        abc_counts = abc_summary['ABC_Category'].value_counts()
        st.sidebar.write(f"**A Items:** {abc_counts.get('A', 0)} (High Value)")
        st.sidebar.write(f"**B Items:** {abc_counts.get('B', 0)} (Medium Value)")
        st.sidebar.write(f"**C Items:** {abc_counts.get('C', 0)} (Low Value)")
    
    # Main item selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        item = st.selectbox("🔍 Select Item for Analysis", df["Item"].dropna().unique())
    
    with col2:
        if show_abc_analysis and item in abc_summary.index:
            abc_cat = abc_summary.loc[item, 'ABC_Category']
            st.metric("ABC Category", abc_cat)
    
    # Filter data for selected item
    item_df = df[df["Item"] == item].copy()
    
    if len(item_df) == 0:
        st.warning("⚠️ No data available for the selected item.")
        return
    
    # Calculate all metrics
    daily_demand = calculate_average_daily_demand(item_df)
    lead_time_data = estimate_lead_time(item_df)
    
    calculator = ReorderPointCalculator(item_df, service_level)
    seasonality = detect_seasonality(calculator.daily_demand)
    
    safety_stock = calculate_safety_stock(
        daily_demand, lead_time_data, 
        calculator.z_score, calculation_method
    )
    
    reorder_scenarios = calculate_reorder_point(
        daily_demand, lead_time_data, safety_stock
    )
    
    # Display key metrics
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Daily Demand (Avg)", 
            f"{daily_demand['recommended']:.2f}",
            delta=f"{daily_demand['trend_adjusted'] - daily_demand['simple']:.2f}"
        )
    
    with col2:
        st.metric(
            "Lead Time (Days)", 
            f"{lead_time_data['avg']:.1f}",
            delta=f"±{lead_time_data['std']:.1f}"
        )
    
    with col3:
        st.metric(
            "Safety Stock", 
            f"{safety_stock:.2f}",
            help=f"Service Level: {service_level*100:.0f}%"
        )
    
    with col4:
        st.metric(
            "Reorder Point (Likely)", 
            f"{reorder_scenarios['likely']:.2f}",
            delta=f"{reorder_scenarios['conservative'] - reorder_scenarios['optimistic']:.1f} range"
        )
    
    with col5:
        total_orders = len(item_df)
        st.metric(
            "Total Orders", 
            total_orders,
            delta=f"{calculator.date_range} days period"
        )
    
    # Seasonality insights
    st.subheader("🔄 Seasonality Analysis")
    if seasonality['has_seasonality']:
        if 'peak_month' in seasonality:
            st.info(f"📊 {seasonality['pattern']} | Peak: {seasonality['peak_month']} | Low: {seasonality['low_month']}")
        else:
            st.info(f"📊 {seasonality['pattern']}")
    else:
        st.success(f"✅ {seasonality['pattern']} - Stable demand pattern")
    
    # Scenario analysis
    st.subheader("🎯 Reorder Point Scenarios")
    scenario_cols = st.columns(3)
    
    scenarios_data = [
        ("Optimistic", reorder_scenarios['optimistic'], "🟢", "Low demand + Short lead time"),
        ("Most Likely", reorder_scenarios['likely'], "🟡", "Expected conditions"),
        ("Conservative", reorder_scenarios['conservative'], "🔴", "High demand + Long lead time")
    ]
    
    for i, (name, value, emoji, desc) in enumerate(scenarios_data):
        with scenario_cols[i]:
            st.metric(f"{emoji} {name}", f"{value:.2f}")
            st.caption(desc)
    
    # Advanced calculations details
    with st.expander("🧮 Detailed Calculations"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Demand Analysis:**")
            st.write(f"• Simple Average: {daily_demand['simple']:.2f}")
            st.write(f"• Weighted Average: {daily_demand['weighted']:.2f}")
            st.write(f"• Trend Adjusted: {daily_demand['trend_adjusted']:.2f}")
            
            st.write("**Lead Time Scenarios:**")
            st.write(f"• Minimum: {lead_time_data['min']:.1f} days")
            st.write(f"• Average: {lead_time_data['avg']:.1f} days")
            st.write(f"• Maximum: {lead_time_data['max']:.1f} days")
        
        with col2:
            st.write("**Safety Stock Formula:**")
            if calculation_method == 'advanced':
                st.code("SS = Z × √(LT×σD² + D²×σLT²)")
            else:
                st.code("SS = Z × D × σLT")
            
            st.write("**Reorder Point Formula:**")
            st.code("ROP = (Daily Demand × Lead Time) + Safety Stock")
            
            st.write(f"**Service Level:** {service_level*100:.0f}% (Z = {calculator.z_score})")
    
    # Advanced visualizations
    st.subheader("📊 Advanced Analytics")
    
    try:
        fig = create_advanced_visualizations(item_df, daily_demand, reorder_scenarios)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        # Fallback to simple chart
        calculator = ReorderPointCalculator(item_df)
        st.line_chart(calculator.daily_demand)
    
    # Performance recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    # Data quality check
    if len(item_df) < 10:
        recommendations.append("⚠️ **Data Quality**: Limited order history. Consider gathering more data for accurate predictions.")
    
    # Seasonality recommendations
    if seasonality['has_seasonality'] and 'seasonal_factor' in seasonality:
        if seasonality['seasonal_factor'] > 2:
            recommendations.append(f"📈 **Seasonality**: High seasonal variation detected. Consider seasonal stock planning for {seasonality['peak_month']}.")
    
    # ABC category recommendations
    if show_abc_analysis and item in abc_summary.index:
        abc_cat = abc_summary.loc[item, 'ABC_Category']
        if abc_cat == 'A':
            recommendations.append("🎯 **High Priority**: This is an 'A' category item. Consider more frequent monitoring and higher service levels.")
        elif abc_cat == 'C':
            recommendations.append("📦 **Low Priority**: This is a 'C' category item. Consider bulk ordering to reduce procurement costs.")
    
    # Lead time recommendations
    if lead_time_data['std'] > lead_time_data['avg']:
        recommendations.append("⏱️ **Lead Time Variability**: High lead time variation detected. Consider backup suppliers or safety lead time buffers.")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✅ No specific recommendations. Current inventory management appears optimal for this item.")
    
    # Export functionality
    st.subheader("📤 Export Results")
    
    # Prepare export data
    export_data = {
        'Item': [item],
        'Daily_Demand_Average': [daily_demand['recommended']],
        'Lead_Time_Days': [lead_time_data['avg']],
        'Safety_Stock': [safety_stock],
        'Reorder_Point_Optimistic': [reorder_scenarios['optimistic']],
        'Reorder_Point_Likely': [reorder_scenarios['likely']],
        'Reorder_Point_Conservative': [reorder_scenarios['conservative']],
        'Service_Level': [f"{service_level*100:.0f}%"],
        'Seasonality': [seasonality['pattern']],
        'ABC_Category': [abc_summary.loc[item, 'ABC_Category'] if show_abc_analysis and item in abc_summary.index else 'N/A']
    }
    
    export_df = pd.DataFrame(export_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📊 Download Analysis (CSV)",
            export_df.to_csv(index=False),
            file_name=f"reorder_analysis_{item}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        if st.button("🔄 Analyze All Items"):
            st.info("Feature coming soon: Bulk analysis of all items with priority ranking.")
