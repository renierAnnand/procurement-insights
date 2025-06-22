import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import io

# Configure Streamlit page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Smart Reorder Point Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import openpyxl for Excel functionality
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Reorder Point Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def calculate_metrics(item_df):
    """Calculate all reorder point metrics for a given item"""
    if len(item_df) == 0:
        return None
    
    # Prepare data
    item_df = item_df.copy()
    item_df['Creation Date'] = pd.to_datetime(item_df['Creation Date'])
    item_df = item_df.sort_values('Creation Date')
    
    # Calculate date range and daily demand
    min_date = item_df['Creation Date'].min()
    max_date = item_df['Creation Date'].max()
    total_days = (max_date - min_date).days + 1
    total_quantity = item_df['Qty Delivered'].sum()
    
    # Average daily demand
    avg_daily_demand = total_quantity / total_days if total_days > 0 else 0
    
    # Estimate lead time (average time between orders)
    time_diffs = item_df['Creation Date'].diff().dropna()
    if len(time_diffs) > 0:
        avg_days_between_orders = time_diffs.dt.days.mean()
        # Use 30% of reorder cycle as lead time estimate (minimum 3 days)
        lead_time = max(avg_days_between_orders * 0.3, 3)
        lead_time_std = time_diffs.dt.days.std() * 0.3 if len(time_diffs) > 1 else 1
    else:
        lead_time = 7  # Default
        lead_time_std = 2
    
    # Safety stock calculation (Z=1.65 for 95% service level)
    z_score = 1.65
    safety_stock = z_score * avg_daily_demand * lead_time_std
    
    # Reorder point scenarios
    optimistic_rop = (avg_daily_demand * lead_time * 0.8) + (safety_stock * 0.5)
    likely_rop = (avg_daily_demand * lead_time) + safety_stock
    conservative_rop = (avg_daily_demand * lead_time * 1.2) + (safety_stock * 1.5)
    
    # Create daily demand series for visualization
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    daily_demand_series = item_df.groupby('Creation Date')['Qty Delivered'].sum()
    daily_demand_series = daily_demand_series.reindex(full_date_range, fill_value=0)
    
    return {
        'avg_daily_demand': avg_daily_demand,
        'lead_time': lead_time,
        'safety_stock': safety_stock,
        'optimistic_rop': optimistic_rop,
        'likely_rop': likely_rop,
        'conservative_rop': conservative_rop,
        'daily_demand_series': daily_demand_series,
        'total_orders': len(item_df),
        'date_range_days': total_days
    }

def create_daily_demand_chart(daily_demand_series):
    """Create enhanced daily demand trend chart"""
    fig = go.Figure()
    
    # Add daily demand line
    fig.add_trace(go.Scatter(
        x=daily_demand_series.index,
        y=daily_demand_series.values,
        mode='lines+markers',
        name='Daily Demand',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Date:</b> %{x}<br><b>Demand:</b> %{y}<extra></extra>'
    ))
    
    # Add 7-day moving average if enough data
    if len(daily_demand_series) >= 7:
        ma_7 = daily_demand_series.rolling(window=7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=ma_7.index,
            y=ma_7.values,
            mode='lines',
            name='7-Day Moving Average',
            line=dict(color='#F18F01', width=3),
            hovertemplate='<b>Date:</b> %{x}<br><b>7-Day Avg:</b> %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='📈 Daily Demand Trend Analysis',
        xaxis_title='Date',
        yaxis_title='Quantity Demanded',
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_scenarios_chart(scenarios, selected_scenario=None):
    """Create reorder point scenarios bar chart"""
    scenario_data = {
        'Scenario': ['Optimistic', 'Likely', 'Conservative'],
        'Reorder Point': [scenarios['optimistic_rop'], scenarios['likely_rop'], scenarios['conservative_rop']],
        'Colors': ['#28a745', '#007bff', '#dc3545']  # Green, Blue, Red
    }
    
    fig = go.Figure()
    
    for i, (scenario, value, color) in enumerate(zip(scenario_data['Scenario'], scenario_data['Reorder Point'], scenario_data['Colors'])):
        # Highlight selected scenario
        opacity = 1.0 if selected_scenario is None or selected_scenario == scenario else 0.6
        border_width = 3 if selected_scenario == scenario else 0
        
        fig.add_trace(go.Bar(
            x=[scenario],
            y=[value],
            name=scenario,
            marker=dict(
                color=color,
                opacity=opacity,
                line=dict(width=border_width, color='black')
            ),
            text=f'{value:.1f}',
            textposition='auto',
            hovertemplate=f'<b>{scenario}</b><br>Reorder Point: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='🎯 Reorder Point Scenarios Comparison',
        xaxis_title='Scenario Type',
        yaxis_title='Reorder Point Quantity',
        showlegend=False,
        height=400
    )
    
    return fig

def create_demand_distribution_chart(daily_demand_series):
    """Create demand distribution histogram"""
    # Remove zero values for better distribution visualization
    non_zero_demand = daily_demand_series[daily_demand_series > 0]
    
    if len(non_zero_demand) == 0:
        # Handle case with no demand
        fig = go.Figure()
        fig.add_annotation(
            text="No demand data available for distribution analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    else:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=non_zero_demand.values,
            nbinsx=min(20, len(non_zero_demand.unique())),
            name='Demand Distribution',
            marker_color='#A23B72',
            opacity=0.7,
            hovertemplate='<b>Demand Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_demand = non_zero_demand.mean()
        fig.add_vline(
            x=mean_demand,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_demand:.2f}",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title='📊 Demand Distribution Analysis',
        xaxis_title='Daily Demand Quantity',
        yaxis_title='Frequency (Days)',
        height=400
    )
    
    return fig

def create_cumulative_demand_chart(daily_demand_series):
    """Create cumulative demand chart"""
    cumulative_demand = daily_demand_series.cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative_demand.index,
        y=cumulative_demand.values,
        mode='lines',
        name='Cumulative Demand',
        line=dict(color='#9B59B6', width=3),
        fill='tonexty',
        fillcolor='rgba(155, 89, 182, 0.3)',
        hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative Demand:</b> %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='📈 Cumulative Demand Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Quantity',
        height=400
    )
    
    return fig

def perform_bulk_analysis(df, progress_callback=None):
    """Perform comprehensive bulk analysis on all items"""
    all_items = df['Item'].dropna().unique()
    results = []
    
    for i, item in enumerate(all_items):
        if progress_callback:
            progress_callback(i + 1, len(all_items), item)
        
        try:
            item_data = df[df['Item'] == item].copy()
            metrics = calculate_metrics(item_data)
            
            if metrics is None:
                continue
            
            # Calculate additional business metrics
            total_quantity = item_data['Qty Delivered'].sum()
            total_orders = len(item_data)
            
            # Calculate demand variability (coefficient of variation)
            daily_demand_std = metrics['daily_demand_series'].std()
            demand_cv = daily_demand_std / metrics['avg_daily_demand'] if metrics['avg_daily_demand'] > 0 else 0
            
            # Calculate order frequency (orders per month)
            date_range_months = max(metrics['date_range_days'] / 30, 1)
            order_frequency = total_orders / date_range_months
            
            # Lead time risk assessment
            lead_time_risk = "Low" if metrics['lead_time'] <= 7 else "Medium" if metrics['lead_time'] <= 14 else "High"
            
            # Priority classification based on volume and variability
            priority_score = (
                (total_quantity / 1000) * 0.4 +  # Volume impact
                (metrics['avg_daily_demand'] * 10) * 0.3 +  # Daily demand impact
                (demand_cv * 100) * 0.2 +  # Variability impact
                (1 / max(order_frequency, 0.1)) * 0.1  # Frequency impact (inverted)
            )
            
            # Assign priority category
            if priority_score >= 50:
                priority = "Critical"
            elif priority_score >= 25:
                priority = "High"
            elif priority_score >= 10:
                priority = "Medium"
            else:
                priority = "Low"
            
            # Seasonal analysis (simplified)
            monthly_demand = metrics['daily_demand_series'].resample('M').sum()
            is_seasonal = len(monthly_demand) >= 6 and (monthly_demand.std() / monthly_demand.mean() > 0.5) if monthly_demand.mean() > 0 else False
            
            results.append({
                'Item': item,
                'Priority': priority,
                'Priority_Score': round(priority_score, 2),
                'Total_Quantity': total_quantity,
                'Total_Orders': total_orders,
                'Avg_Daily_Demand': round(metrics['avg_daily_demand'], 2),
                'Lead_Time_Days': round(metrics['lead_time'], 1),
                'Lead_Time_Risk': lead_time_risk,
                'Safety_Stock': round(metrics['safety_stock'], 2),
                'Reorder_Point_Optimistic': round(metrics['optimistic_rop'], 2),
                'Reorder_Point_Likely': round(metrics['likely_rop'], 2),
                'Reorder_Point_Conservative': round(metrics['conservative_rop'], 2),
                'Demand_Variability_CV': round(demand_cv, 3),
                'Order_Frequency_Monthly': round(order_frequency, 2),
                'Is_Seasonal': is_seasonal,
                'Data_Period_Days': metrics['date_range_days'],
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d')
            })
            
        except Exception as e:
            st.warning(f"Could not analyze item '{item}': {str(e)}")
            continue
    
    return pd.DataFrame(results) if results else None

def create_excel_report(bulk_results_df, df):
    """Create comprehensive Excel report with multiple sheets"""
    if not EXCEL_AVAILABLE:
        st.error("❌ Excel export not available. openpyxl package is required.")
        return None
    
    # Create Excel buffer
    excel_buffer = io.BytesIO()
    
    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary
            create_executive_summary_sheet(bulk_results_df, writer)
            
            # Sheet 2: Detailed Analysis
            bulk_results_df.to_excel(writer, sheet_name='Detailed Analysis', index=False)
            
            # Sheet 3: Priority Matrix
            create_priority_matrix_sheet(bulk_results_df, writer)
            
            # Sheet 4: Risk Assessment
            create_risk_assessment_sheet(bulk_results_df, writer)
            
            # Sheet 5: Action Plan
            create_action_plan_sheet(bulk_results_df, writer)
            
            # Sheet 6: Raw Data Summary
            create_raw_data_summary_sheet(df, writer)
        
        excel_buffer.seek(0)
        return excel_buffer
    except Exception as e:
        st.error(f"❌ Error creating Excel report: {str(e)}")
        return None

def create_csv_bundle(bulk_results_df, df):
    """Create multiple CSV files as a bundle when Excel is not available"""
    csv_files = {}
    
    # Executive Summary
    summary_data = {
        'Metric': [
            'Total Items Analyzed',
            'Critical Priority Items',
            'High Priority Items', 
            'Medium Priority Items',
            'Low Priority Items',
            'Items with High Lead Time Risk',
            'Seasonal Items',
            'Total Reorder Investment (Likely)',
            'Total Reorder Investment (Conservative)',
            'Average Lead Time (Days)',
            'Items Requiring Immediate Action'
        ],
        'Value': [
            len(bulk_results_df),
            len(bulk_results_df[bulk_results_df['Priority'] == 'Critical']),
            len(bulk_results_df[bulk_results_df['Priority'] == 'High']),
            len(bulk_results_df[bulk_results_df['Priority'] == 'Medium']),
            len(bulk_results_df[bulk_results_df['Priority'] == 'Low']),
            len(bulk_results_df[bulk_results_df['Lead_Time_Risk'] == 'High']),
            len(bulk_results_df[bulk_results_df['Is_Seasonal'] == True]),
            f"{bulk_results_df['Reorder_Point_Likely'].sum():,.0f}",
            f"{bulk_results_df['Reorder_Point_Conservative'].sum():,.0f}",
            f"{bulk_results_df['Lead_Time_Days'].mean():.1f}",
            len(bulk_results_df[(bulk_results_df['Priority'].isin(['Critical', 'High'])) | 
                              (bulk_results_df['Lead_Time_Risk'] == 'High')])
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    csv_files['executive_summary'] = summary_df.to_csv(index=False)
    
    # Detailed Analysis
    csv_files['detailed_analysis'] = bulk_results_df.to_csv(index=False)
    
    # Priority Matrix
    priority_analysis = bulk_results_df.groupby('Priority').agg({
        'Item': 'count',
        'Total_Quantity': 'sum',
        'Reorder_Point_Likely': 'sum',
        'Avg_Daily_Demand': 'mean',
        'Lead_Time_Days': 'mean'
    }).round(2)
    priority_analysis.columns = ['Item_Count', 'Total_Volume', 'Total_Reorder_Investment', 
                               'Avg_Daily_Demand', 'Avg_Lead_Time']
    csv_files['priority_matrix'] = priority_analysis.to_csv()
    
    # High-risk items
    high_risk_df = bulk_results_df[
        (bulk_results_df['Lead_Time_Risk'] == 'High') |
        (bulk_results_df['Demand_Variability_CV'] > 0.5) |
        (bulk_results_df['Is_Seasonal'] == True)
    ].copy()
    
    if not high_risk_df.empty:
        risk_columns = ['Item', 'Priority', 'Lead_Time_Risk', 'Demand_Variability_CV', 
                       'Is_Seasonal', 'Reorder_Point_Conservative']
        csv_files['risk_assessment'] = high_risk_df[risk_columns].to_csv(index=False)
    
    return csv_files

def create_executive_summary_sheet(bulk_results_df, writer):
    """Create executive summary sheet"""
    if not EXCEL_AVAILABLE:
        return
        
    summary_data = {
        'Metric': [
            'Total Items Analyzed',
            'Critical Priority Items',
            'High Priority Items', 
            'Medium Priority Items',
            'Low Priority Items',
            'Items with High Lead Time Risk',
            'Seasonal Items',
            'Total Reorder Investment (Likely)',
            'Total Reorder Investment (Conservative)',
            'Average Lead Time (Days)',
            'Items Requiring Immediate Action'
        ],
        'Value': [
            len(bulk_results_df),
            len(bulk_results_df[bulk_results_df['Priority'] == 'Critical']),
            len(bulk_results_df[bulk_results_df['Priority'] == 'High']),
            len(bulk_results_df[bulk_results_df['Priority'] == 'Medium']),
            len(bulk_results_df[bulk_results_df['Priority'] == 'Low']),
            len(bulk_results_df[bulk_results_df['Lead_Time_Risk'] == 'High']),
            len(bulk_results_df[bulk_results_df['Is_Seasonal'] == True]),
            f"{bulk_results_df['Reorder_Point_Likely'].sum():,.0f}",
            f"{bulk_results_df['Reorder_Point_Conservative'].sum():,.0f}",
            f"{bulk_results_df['Lead_Time_Days'].mean():.1f}",
            len(bulk_results_df[(bulk_results_df['Priority'].isin(['Critical', 'High'])) | 
                              (bulk_results_df['Lead_Time_Risk'] == 'High')])
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)

def create_priority_matrix_sheet(bulk_results_df, writer):
    """Create priority matrix analysis sheet"""
    if not EXCEL_AVAILABLE:
        return
        
    priority_analysis = bulk_results_df.groupby('Priority').agg({
        'Item': 'count',
        'Total_Quantity': 'sum',
        'Reorder_Point_Likely': 'sum',
        'Avg_Daily_Demand': 'mean',
        'Lead_Time_Days': 'mean'
    }).round(2)
    
    priority_analysis.columns = ['Item_Count', 'Total_Volume', 'Total_Reorder_Investment', 
                               'Avg_Daily_Demand', 'Avg_Lead_Time']
    priority_analysis.to_excel(writer, sheet_name='Priority Matrix')

def create_risk_assessment_sheet(bulk_results_df, writer):
    """Create risk assessment sheet"""
    if not EXCEL_AVAILABLE:
        return
        
    # High-risk items
    high_risk_df = bulk_results_df[
        (bulk_results_df['Lead_Time_Risk'] == 'High') |
        (bulk_results_df['Demand_Variability_CV'] > 0.5) |
        (bulk_results_df['Is_Seasonal'] == True)
    ].copy()
    
    if not high_risk_df.empty:
        high_risk_df['Risk_Factors'] = ''
        for idx, row in high_risk_df.iterrows():
            factors = []
            if row['Lead_Time_Risk'] == 'High':
                factors.append('High Lead Time')
            if row['Demand_Variability_CV'] > 0.5:
                factors.append('High Demand Variability')
            if row['Is_Seasonal']:
                factors.append('Seasonal Demand')
            high_risk_df.at[idx, 'Risk_Factors'] = ', '.join(factors)
        
        risk_columns = ['Item', 'Priority', 'Lead_Time_Risk', 'Demand_Variability_CV', 
                       'Is_Seasonal', 'Risk_Factors', 'Reorder_Point_Conservative']
        high_risk_df[risk_columns].to_excel(writer, sheet_name='Risk Assessment', index=False)

def create_action_plan_sheet(bulk_results_df, writer):
    """Create action plan sheet with specific recommendations"""
    if not EXCEL_AVAILABLE:
        return
        
    action_items = []
    
    # Critical items needing immediate attention
    critical_items = bulk_results_df[bulk_results_df['Priority'] == 'Critical']
    for _, item in critical_items.iterrows():
        action_items.append({
            'Item': item['Item'],
            'Priority': 'URGENT',
            'Action': 'Review inventory levels immediately',
            'Recommended_Reorder_Point': item['Reorder_Point_Conservative'],
            'Reason': 'Critical priority item',
            'Timeline': 'Within 24 hours'
        })
    
    # High lead time risk items
    high_lead_time = bulk_results_df[bulk_results_df['Lead_Time_Risk'] == 'High']
    for _, item in high_lead_time.iterrows():
        if item['Priority'] != 'Critical':  # Avoid duplicates
            action_items.append({
                'Item': item['Item'],
                'Priority': 'HIGH',
                'Action': 'Negotiate shorter lead times or find backup suppliers',
                'Recommended_Reorder_Point': item['Reorder_Point_Conservative'],
                'Reason': 'High lead time risk',
                'Timeline': 'Within 1 week'
            })
    
    # Seasonal items
    seasonal_items = bulk_results_df[bulk_results_df['Is_Seasonal'] == True]
    for _, item in seasonal_items.iterrows():
        if item['Priority'] not in ['Critical'] and item['Lead_Time_Risk'] != 'High':  # Avoid duplicates
            action_items.append({
                'Item': item['Item'],
                'Priority': 'MEDIUM',
                'Action': 'Plan seasonal inventory buildup',
                'Recommended_Reorder_Point': item['Reorder_Point_Likely'],
                'Reason': 'Seasonal demand pattern detected',
                'Timeline': 'Within 2 weeks'
            })
    
    if action_items:
        action_df = pd.DataFrame(action_items)
        action_df.to_excel(writer, sheet_name='Action Plan', index=False)

def create_raw_data_summary_sheet(df, writer):
    """Create raw data summary sheet"""
    if not EXCEL_AVAILABLE:
        return
        
    data_summary = {
        'Metric': [
            'Total Records',
            'Date Range Start',
            'Date Range End',
            'Total Unique Items',
            'Total Quantity Delivered',
            'Average Order Size',
            'Data Collection Period (Days)'
        ],
        'Value': [
            len(df),
            df['Creation Date'].min().strftime('%Y-%m-%d') if not df.empty else 'N/A',
            df['Creation Date'].max().strftime('%Y-%m-%d') if not df.empty else 'N/A',
            df['Item'].nunique(),
            df['Qty Delivered'].sum(),
            f"{df['Qty Delivered'].mean():.2f}",
            (pd.to_datetime(df['Creation Date']).max() - pd.to_datetime(df['Creation Date']).min()).days
        ]
    }
    
    summary_df = pd.DataFrame(data_summary)
    summary_df.to_excel(writer, sheet_name='Data Summary', index=False)

def display_bulk_analysis_dashboard(bulk_results_df):
    """Display bulk analysis results dashboard"""
    st.header("📊 Bulk Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items", len(bulk_results_df))
    
    with col2:
        critical_count = len(bulk_results_df[bulk_results_df['Priority'] == 'Critical'])
        st.metric("Critical Items", critical_count)
    
    with col3:
        total_investment = bulk_results_df['Reorder_Point_Likely'].sum()
        st.metric("Total Investment", f"{total_investment:,.0f}")
    
    with col4:
        high_risk_count = len(bulk_results_df[bulk_results_df['Lead_Time_Risk'] == 'High'])
        st.metric("High Risk Items", high_risk_count)
    
    # Priority distribution
    st.subheader("🎯 Priority Distribution")
    priority_counts = bulk_results_df['Priority'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(priority_counts.to_frame('Count'))
    
    with col2:
        fig_pie = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            title="Items by Priority",
            color_discrete_map={
                'Critical': '#dc3545',
                'High': '#fd7e14', 
                'Medium': '#ffc107',
                'Low': '#28a745'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Risk analysis
    st.subheader("⚠️ Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead time risk distribution
        risk_counts = bulk_results_df['Lead_Time_Risk'].value_counts()
        fig_risk = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Lead Time Risk Distribution",
            color=risk_counts.index,
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Seasonal items
        seasonal_count = bulk_results_df['Is_Seasonal'].sum()
        seasonal_data = pd.DataFrame({
            'Type': ['Seasonal', 'Non-Seasonal'],
            'Count': [seasonal_count, len(bulk_results_df) - seasonal_count]
        })
        fig_seasonal = px.bar(
            seasonal_data,
            x='Type',
            y='Count',
            title="Seasonal vs Non-Seasonal Items",
            color='Type',
            color_discrete_map={'Seasonal': '#17a2b8', 'Non-Seasonal': '#6c757d'}
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Top items requiring attention
    st.subheader("🚨 Items Requiring Immediate Attention")
    
    attention_items = bulk_results_df[
        (bulk_results_df['Priority'].isin(['Critical', 'High'])) |
        (bulk_results_df['Lead_Time_Risk'] == 'High')
    ].sort_values(['Priority', 'Priority_Score'], ascending=[True, False])
    
    if len(attention_items) > 0:
        display_cols = ['Item', 'Priority', 'Priority_Score', 'Lead_Time_Risk', 
                       'Reorder_Point_Likely', 'Avg_Daily_Demand']
        st.dataframe(attention_items[display_cols].head(10), use_container_width=True)
    else:
        st.success("✅ No items require immediate attention!")
    
    # Interactive scatter plot
    st.subheader("📈 Priority vs Investment Analysis")
    
    fig_scatter = px.scatter(
        bulk_results_df,
        x='Avg_Daily_Demand',
        y='Reorder_Point_Likely',
        size='Total_Quantity',
        color='Priority',
        hover_data=['Item', 'Lead_Time_Days'],
        title="Daily Demand vs Reorder Point (sized by total volume)",
        color_discrete_map={
            'Critical': '#dc3545',
            'High': '#fd7e14',
            'Medium': '#ffc107', 
            'Low': '#28a745'
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Detailed data table with filters
    st.subheader("📋 Detailed Analysis Table")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority_filter = st.multiselect(
            "Filter by Priority",
            bulk_results_df['Priority'].unique(),
            default=bulk_results_df['Priority'].unique()
        )
    
    with col2:
        risk_filter = st.multiselect(
            "Filter by Lead Time Risk",
            bulk_results_df['Lead_Time_Risk'].unique(),
            default=bulk_results_df['Lead_Time_Risk'].unique()
        )
    
    with col3:
        min_demand = st.number_input(
            "Minimum Daily Demand",
            min_value=0.0,
            value=0.0,
            step=0.1
        )
    
    # Apply filters
    filtered_df = bulk_results_df[
        (bulk_results_df['Priority'].isin(priority_filter)) &
        (bulk_results_df['Lead_Time_Risk'].isin(risk_filter)) &
        (bulk_results_df['Avg_Daily_Demand'] >= min_demand)
    ]
    
    st.dataframe(filtered_df, use_container_width=True, height=400)
    """Generate simple forecast data for demonstration"""
    if len(daily_demand_series) < 7:
        return None
    
    # Simple trend-based forecast
    recent_trend = daily_demand_series.tail(30).mean()
    seasonal_factor = 1 + 0.1 * np.sin(np.arange(days_ahead) * 2 * np.pi / 365)
    
    forecast_dates = pd.date_range(
        start=daily_demand_series.index[-1] + timedelta(days=1),
        periods=days_ahead,
        freq='D'
    )
    
    # Add some randomness to make it realistic
    np.random.seed(42)
    noise = np.random.normal(0, recent_trend * 0.2, days_ahead)
    forecast_values = (recent_trend * seasonal_factor + noise).clip(min=0)
    
    return pd.Series(forecast_values, index=forecast_dates)

def generate_forecast_data(daily_demand_series, days_ahead=90):
    """Generate simple forecast data for demonstration"""
    if len(daily_demand_series) < 7:
        return None
    
    # Simple trend-based forecast
    recent_trend = daily_demand_series.tail(30).mean()
    seasonal_factor = 1 + 0.1 * np.sin(np.arange(days_ahead) * 2 * np.pi / 365)
    
    forecast_dates = pd.date_range(
        start=daily_demand_series.index[-1] + timedelta(days=1),
        periods=days_ahead,
        freq='D'
    )
    
    # Add some randomness to make it realistic
    np.random.seed(42)
    noise = np.random.normal(0, recent_trend * 0.2, days_ahead)
    forecast_values = (recent_trend * seasonal_factor + noise).clip(min=0)
    
    return pd.Series(forecast_values, index=forecast_dates)

def main():
    # Initialize session state
    if 'start_bulk_analysis' not in st.session_state:
        st.session_state['start_bulk_analysis'] = False
    if 'bulk_analysis_complete' not in st.session_state:
        st.session_state['bulk_analysis_complete'] = False
    
    # Show Excel availability warning here if needed
    if not EXCEL_AVAILABLE:
        st.sidebar.warning("⚠️ openpyxl not available. Excel export will be disabled. Install with: `pip install openpyxl`")
    
    # Header
    st.title("🎯 Smart Reorder Point Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Sample data option for demonstration
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        # Create sample data for demonstration
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        
        sample_data = []
        items = ['Widget A', 'Component B', 'Part C', 'Material D']
        
        for item in items:
            # Generate realistic order patterns
            order_dates = np.random.choice(dates, size=np.random.randint(50, 150), replace=False)
            for date in order_dates:
                qty = np.random.poisson(lam=20) + 5  # Realistic quantities
                sample_data.append({
                    'Item': item,
                    'Qty Delivered': qty,
                    'Creation Date': date
                })
        
        df = pd.DataFrame(sample_data)
        st.sidebar.success("✅ Using sample data")
    else:
        st.sidebar.info("📁 Upload your CSV file with columns: Item, Qty Delivered, Creation Date")
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Data loaded successfully")
        else:
            st.warning("⚠️ Please upload a CSV file or use sample data to continue.")
            return
    
    # Validate required columns
    required_columns = ['Item', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Main navigation
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single Item Analysis", "Bulk Analysis"],
        help="Choose between analyzing individual items or all items at once"
    )
    
    if analysis_mode == "Bulk Analysis":
        # Bulk Analysis Section
        st.header("📊 Bulk Analysis - All Items")
        st.markdown("Analyze all items in your dataset with comprehensive reporting and Excel export.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Start Bulk Analysis", type="primary"):
                st.session_state['start_bulk_analysis'] = True
        
        with col2:
            st.info(f"📦 {len(df['Item'].unique())} items will be analyzed")
        
        # Perform bulk analysis
        if st.session_state.get('start_bulk_analysis', False):
            with st.spinner("🔄 Analyzing all items... This may take a few minutes."):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total, item_name):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing item {current}/{total}: {item_name}")
                
                # Run bulk analysis
                bulk_results = perform_bulk_analysis(df, progress_callback)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if bulk_results is not None and not bulk_results.empty:
                    st.success(f"✅ Successfully analyzed {len(bulk_results)} items!")
                    
                    # Store results in session state
                    st.session_state['bulk_results'] = bulk_results
                    st.session_state['bulk_analysis_complete'] = True
                    
                    # Display dashboard
                    display_bulk_analysis_dashboard(bulk_results)
                    
                    # Excel Export Section
                    st.markdown("---")
                    st.subheader("📤 Export Options")
                    
                    if EXCEL_AVAILABLE:
                        # Excel export available
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Generate Excel report
                            excel_buffer = create_excel_report(bulk_results, df)
                            
                            if excel_buffer is not None:
                                st.download_button(
                                    label="📊 Download Complete Excel Report",
                                    data=excel_buffer.getvalue(),
                                    file_name=f"inventory_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    help="Download comprehensive Excel report with multiple sheets",
                                    type="primary"
                                )
                        
                        with col2:
                            # CSV export option
                            csv_buffer = bulk_results.to_csv(index=False)
                            st.download_button(
                                label="📋 Download CSV Data",
                                data=csv_buffer,
                                file_name=f"bulk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                help="Download raw data in CSV format"
                            )
                        
                        with col3:
                            # Critical items only export
                            critical_items = bulk_results[bulk_results['Priority'] == 'Critical']
                            if not critical_items.empty:
                                critical_csv = critical_items.to_csv(index=False)
                                st.download_button(
                                    label="🚨 Download Critical Items Only",
                                    data=critical_csv,
                                    file_name=f"critical_items_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv",
                                    help="Download only critical priority items"
                                )
                            else:
                                st.info("No critical items found")
                        
                        # Excel report contents explanation
                        with st.expander("📋 Excel Report Contents", expanded=False):
                            st.markdown("""
                            **The Excel report contains 6 sheets:**
                            
                            1. **Executive Summary** - High-level metrics and KPIs
                            2. **Detailed Analysis** - Complete analysis for all items
                            3. **Priority Matrix** - Items grouped by priority level
                            4. **Risk Assessment** - High-risk items requiring attention
                            5. **Action Plan** - Specific recommendations with timelines
                            6. **Data Summary** - Raw data statistics and quality metrics
                            
                            Perfect for sharing with management and procurement teams!
                            """)
                    
                    else:
                        # Excel not available - provide CSV alternatives
                        st.warning("📊 Excel export not available. Providing CSV alternatives:")
                        
                        csv_bundle = create_csv_bundle(bulk_results, df)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Executive Summary CSV
                            if 'executive_summary' in csv_bundle:
                                st.download_button(
                                    label="📈 Executive Summary (CSV)",
                                    data=csv_bundle['executive_summary'],
                                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv",
                                    help="High-level metrics and KPIs"
                                )
                        
                        with col2:
                            # Detailed Analysis CSV
                            if 'detailed_analysis' in csv_bundle:
                                st.download_button(
                                    label="📋 Detailed Analysis (CSV)",
                                    data=csv_bundle['detailed_analysis'],
                                    file_name=f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv",
                                    help="Complete analysis for all items"
                                )
                        
                        with col3:
                            # Priority Matrix CSV
                            if 'priority_matrix' in csv_bundle:
                                st.download_button(
                                    label="🎯 Priority Matrix (CSV)",
                                    data=csv_bundle['priority_matrix'],
                                    file_name=f"priority_matrix_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv",
                                    help="Items grouped by priority level"
                                )
                        
                        # Additional CSV downloads
                        st.markdown("**Additional Reports:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Risk Assessment CSV
                            if 'risk_assessment' in csv_bundle:
                                st.download_button(
                                    label="⚠️ Risk Assessment (CSV)",
                                    data=csv_bundle['risk_assessment'],
                                    file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv",
                                    help="High-risk items requiring attention"
                                )
                        
                        with col2:
                            # Critical items only
                            critical_items = bulk_results[bulk_results['Priority'] == 'Critical']
                            if not critical_items.empty:
                                critical_csv = critical_items.to_csv(index=False)
                                st.download_button(
                                    label="🚨 Critical Items (CSV)",
                                    data=critical_csv,
                                    file_name=f"critical_items_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv",
                                    help="Critical priority items only"
                                )
                        
                        # Installation instructions
                        with st.expander("💡 How to Enable Excel Export", expanded=False):
                            st.markdown("""
                            **To enable Excel export functionality:**
                            
                            1. **Local Installation:**
                               ```bash
                               pip install openpyxl
                               ```
                            
                            2. **Streamlit Cloud/Deployment:**
                               - Add `openpyxl` to your `requirements.txt` file
                               - Redeploy your application
                            
                            3. **Requirements.txt example:**
                               ```
                               streamlit
                               pandas
                               numpy
                               plotly
                               openpyxl
                               ```
                            
                            Once installed, restart the application for Excel export to become available.
                            """)
                
                    
                else:
                    st.error("❌ Bulk analysis failed. Please check your data format and try again.")
                    st.session_state['start_bulk_analysis'] = False
        
        # Show previous results if available
        elif st.session_state.get('bulk_analysis_complete', False):
            if st.button("🔍 Show Previous Bulk Analysis Results"):
                if 'bulk_results' in st.session_state:
                    display_bulk_analysis_dashboard(st.session_state['bulk_results'])
                    
                    # Re-enable export options
                    st.markdown("---")
                    st.subheader("📤 Export Previous Results")
                    
                    bulk_results = st.session_state['bulk_results']
                    
                    if EXCEL_AVAILABLE:
                        excel_buffer = create_excel_report(bulk_results, df)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if excel_buffer is not None:
                                st.download_button(
                                    label="📊 Download Excel Report",
                                    data=excel_buffer.getvalue(),
                                    file_name=f"inventory_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        
                        with col2:
                            csv_buffer = bulk_results.to_csv(index=False)
                            st.download_button(
                                label="📋 Download CSV Data", 
                                data=csv_buffer,
                                file_name=f"bulk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                    else:
                        # Provide CSV alternatives
                        st.warning("📊 Excel export not available. Download CSV files:")
                        
                        csv_bundle = create_csv_bundle(bulk_results, df)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if 'detailed_analysis' in csv_bundle:
                                st.download_button(
                                    label="📋 Detailed Analysis (CSV)",
                                    data=csv_bundle['detailed_analysis'],
                                    file_name=f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )
                        
                        with col2:
                            if 'executive_summary' in csv_bundle:
                                st.download_button(
                                    label="📈 Executive Summary (CSV)",
                                    data=csv_bundle['executive_summary'],
                                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )
                        
                        with col3:
                            if 'priority_matrix' in csv_bundle:
                                st.download_button(
                                    label="🎯 Priority Matrix (CSV)",
                                    data=csv_bundle['priority_matrix'],
                                    file_name=f"priority_matrix_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )
                else:
                    st.warning("Previous results not found. Please run bulk analysis again.")
        
        return  # Exit here for bulk analysis mode
    
    # Single Item Analysis Mode (existing code continues below)
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📦 Item Selection")
        selected_item = st.selectbox(
            "Choose an item to analyze:",
            options=sorted(df['Item'].unique()),
            help="Select an item from your inventory to perform reorder point analysis"
        )
    
    with col2:
        st.subheader("🎛️ Scenario Selection")
        selected_scenario = st.radio(
            "Highlight scenario:",
            options=['Optimistic', 'Likely', 'Conservative'],
            index=1,
            help="Choose which reorder scenario to emphasize in visualizations"
        )
    
    # Filter data for selected item
    item_data = df[df['Item'] == selected_item].copy()
    
    if len(item_data) == 0:
        st.error(f"❌ No data found for item: {selected_item}")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(item_data)
    
    if metrics is None:
        st.error("❌ Unable to calculate metrics for this item")
        return
    
    # Display key metrics
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Average Daily Demand",
            value=f"{metrics['avg_daily_demand']:.2f}",
            help="Total quantity delivered divided by the number of days in the analysis period"
        )
    
    with col2:
        st.metric(
            label="Average Lead Time",
            value=f"{metrics['lead_time']:.1f} days",
            help="Estimated time between placing an order and receiving delivery"
        )
    
    with col3:
        st.metric(
            label="Safety Stock",
            value=f"{metrics['safety_stock']:.2f}",
            help="Buffer stock to maintain 95% service level during demand/lead time variations"
        )
    
    with col4:
        # Show the selected scenario's ROP
        scenario_values = {
            'Optimistic': metrics['optimistic_rop'],
            'Likely': metrics['likely_rop'],
            'Conservative': metrics['conservative_rop']
        }
        st.metric(
            label=f"Reorder Point ({selected_scenario})",
            value=f"{scenario_values[selected_scenario]:.2f}",
            help=f"Suggested reorder point for {selected_scenario.lower()} scenario"
        )
    
    st.markdown("---")
    
    # Decision guidance
    st.subheader("🧠 Decision Guidance")
    
    with st.expander("📋 Scenario Explanations", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🟢 Optimistic Scenario**
            - Lower demand and shorter lead times
            - Use when: Supply chain is stable
            - Risk: Potential stockouts
            - Best for: Cost-sensitive items
            """)
        
        with col2:
            st.markdown("""
            **🔵 Likely Scenario (Recommended)**
            - Most probable demand and lead time
            - Use when: Normal business conditions
            - Risk: Balanced approach
            - Best for: Most inventory items
            """)
        
        with col3:
            st.markdown("""
            **🔴 Conservative Scenario**
            - Higher demand and longer lead times
            - Use when: Uncertain supply/demand
            - Risk: Higher holding costs
            - Best for: Critical items
            """)
    
    # Visualizations
    st.subheader("📈 Visual Analysis")
    
    # Row 1: Daily demand and scenarios
    col1, col2 = st.columns(2)
    
    with col1:
        demand_chart = create_daily_demand_chart(metrics['daily_demand_series'])
        st.plotly_chart(demand_chart, use_container_width=True)
    
    with col2:
        scenarios_chart = create_scenarios_chart(metrics, selected_scenario)
        st.plotly_chart(scenarios_chart, use_container_width=True)
    
    # Row 2: Distribution and cumulative
    col1, col2 = st.columns(2)
    
    with col1:
        distribution_chart = create_demand_distribution_chart(metrics['daily_demand_series'])
        st.plotly_chart(distribution_chart, use_container_width=True)
    
    with col2:
        cumulative_chart = create_cumulative_demand_chart(metrics['daily_demand_series'])
        st.plotly_chart(cumulative_chart, use_container_width=True)
    
    # Forecast feature (bonus)
    st.subheader("🔮 Demand Forecast (Optional)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("📊 Show 3-Month Forecast"):
            st.session_state['show_forecast'] = True
    
    with col2:
        st.info("💡 Forecast uses trend analysis and seasonal patterns from historical data")
    
    if st.session_state.get('show_forecast', False):
        forecast = generate_forecast_data(metrics['daily_demand_series'])
        
        if forecast is not None:
            # Combined historical and forecast chart
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=metrics['daily_demand_series'].index,
                y=metrics['daily_demand_series'].values,
                mode='lines',
                name='Historical Demand',
                line=dict(color='#2E86AB', width=2)
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines',
                name='3-Month Forecast',
                line=dict(color='#F18F01', width=2, dash='dash')
            ))
            
            # Add vertical line to separate historical from forecast
            fig.add_vline(
                x=metrics['daily_demand_series'].index[-1],
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Start"
            )
            
            fig.update_layout(
                title='📈 Historical Demand + 3-Month Forecast',
                xaxis_title='Date',
                yaxis_title='Daily Demand',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Insufficient data for forecasting (need at least 7 days)")
    
    # Summary and next steps
    st.markdown("---")
    st.subheader("🎯 Summary & Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **📊 Analysis Summary for {selected_item}:**
        - Data period: {metrics['date_range_days']} days
        - Total orders: {metrics['total_orders']}
        - Recommended reorder point: **{metrics['likely_rop']:.2f}** units
        - Current inventory position needed for review
        """)
    
    with col2:
        st.markdown("""
        **🎯 Recommended Actions:**
        1. ✅ Review current inventory levels
        2. 📦 Set reorder point in your system
        3. 🔄 Monitor demand patterns monthly
        4. ⚡ Consider supplier lead time agreements
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard powered by Streamlit and Plotly | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "*")

if __name__ == "__main__":
    main()
