import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def calculate_vendor_performance_score(vendor_data):
    """Calculate comprehensive vendor performance score"""
    metrics = {}
    
    # Volume consistency (coefficient of variation of monthly orders)
    vendor_data['Order_Month'] = vendor_data['Creation Date'].dt.to_period('M')
    monthly_orders = vendor_data.groupby('Order_Month').size()
    if len(monthly_orders) > 1:
        metrics['volume_consistency'] = 1 - (monthly_orders.std() / monthly_orders.mean()) if monthly_orders.mean() > 0 else 0
    else:
        metrics['volume_consistency'] = 0
    
    # Price stability (1 - coefficient of variation of unit prices)
    if len(vendor_data['Unit Price']) > 1:
        price_cv = vendor_data['Unit Price'].std() / vendor_data['Unit Price'].mean() if vendor_data['Unit Price'].mean() > 0 else 0
        metrics['price_stability'] = max(0, 1 - price_cv)
    else:
        metrics['price_stability'] = 0.5
    
    # Lead time consistency (using difference between ordered and delivered dates)
    if 'Delivered date' in vendor_data.columns and 'Ordered Date' in vendor_data.columns:
        vendor_data['lead_time_days'] = (vendor_data['Delivered date'] - vendor_data['Ordered Date']).dt.days
        lead_times = vendor_data['lead_time_days'].dropna()
        if len(lead_times) > 1:
            lt_cv = lead_times.std() / lead_times.mean() if lead_times.mean() > 0 else 0
            metrics['lead_time_consistency'] = max(0, 1 - lt_cv)
        else:
            metrics['lead_time_consistency'] = 0.5
    else:
        metrics['lead_time_consistency'] = 0.5
    
    # Delivery performance (on-time delivery)
    if 'lead_time_days' in vendor_data.columns:
        lead_times = vendor_data['lead_time_days'].dropna()
        if len(lead_times) > 0:
            # Consider on-time if delivered within 30 days
            on_time_rate = len(lead_times[lead_times <= 30]) / len(lead_times)
            metrics['delivery_performance'] = on_time_rate
        else:
            metrics['delivery_performance'] = 0.5
    else:
        metrics['delivery_performance'] = 0.5
    
    # Quality score (based on rejection rates)
    if 'Qty Rejected' in vendor_data.columns:
        total_delivered = vendor_data['Qty Delivered'].sum()
        total_rejected = vendor_data['Qty Rejected'].sum()
        if total_delivered > 0:
            rejection_rate = total_rejected / (total_delivered + total_rejected)
            metrics['quality_score'] = 1 - rejection_rate
        else:
            metrics['quality_score'] = 0.5
    else:
        metrics['quality_score'] = 0.5
    
    # Calculate weighted overall score
    weights = {
        'volume_consistency': 0.25,
        'price_stability': 0.25,
        'lead_time_consistency': 0.20,
        'delivery_performance': 0.20,
        'quality_score': 0.10
    }
    
    overall_score = sum(metrics[metric] * weights[metric] for metric in weights)
    
    return {
        'overall_score': overall_score,
        **metrics
    }

def analyze_contract_suitability(item_vendor_data, min_spend_threshold=10000, min_frequency_threshold=4):
    """Analyze suitability for contracting based on spend and frequency"""
    
    # Calculate key metrics
    total_spend = item_vendor_data['Line Total'].sum()
    order_frequency = len(item_vendor_data)
    
    # Calculate time span
    date_range = item_vendor_data['Creation Date'].max() - item_vendor_data['Creation Date'].min()
    months_span = date_range.days / 30 if date_range.days > 0 else 1
    monthly_frequency = order_frequency / months_span
    
    # Demand predictability
    monthly_demand = item_vendor_data.groupby(item_vendor_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
    if len(monthly_demand) > 1:
        demand_cv = monthly_demand.std() / monthly_demand.mean() if monthly_demand.mean() > 0 else 1
        demand_predictability = max(0, 1 - demand_cv)
    else:
        demand_predictability = 0.3
    
    # Contract suitability score
    spend_score = min(total_spend / min_spend_threshold, 1.0) if min_spend_threshold > 0 else 1.0
    frequency_score = min(monthly_frequency / (min_frequency_threshold / 12), 1.0)
    
    suitability_score = (spend_score * 0.4 + frequency_score * 0.3 + demand_predictability * 0.3)
    
    # Contract recommendation
    if suitability_score >= 0.7 and total_spend >= min_spend_threshold:
        recommendation = "High Priority"
    elif suitability_score >= 0.5 and total_spend >= min_spend_threshold * 0.5:
        recommendation = "Medium Priority"
    elif suitability_score >= 0.3:
        recommendation = "Low Priority"
    else:
        recommendation = "Not Suitable"
    
    return {
        'total_spend': total_spend,
        'order_frequency': order_frequency,
        'monthly_frequency': monthly_frequency,
        'demand_predictability': demand_predictability,
        'suitability_score': suitability_score,
        'recommendation': recommendation,
        'months_span': months_span
    }

def calculate_contract_savings_potential(historical_data, contract_terms):
    """Calculate potential savings from contracting"""
    
    # Current average price
    current_avg_price = historical_data['Unit Price'].mean()
    annual_volume = (historical_data['Qty Delivered'].sum() / 
                    ((historical_data['Creation Date'].max() - historical_data['Creation Date'].min()).days / 365))
    
    # Contract savings scenarios
    savings_scenarios = []
    
    for term in contract_terms:
        # Price reduction from volume commitment
        volume_discount = term.get('volume_discount', 0)
        contract_price = current_avg_price * (1 - volume_discount)
        
        # Administrative cost savings
        admin_savings_per_order = term.get('admin_savings', 0)
        current_orders_per_year = len(historical_data) / ((historical_data['Creation Date'].max() - historical_data['Creation Date'].min()).days / 365)
        contract_orders_per_year = term.get('orders_per_year', current_orders_per_year)
        
        total_admin_savings = (current_orders_per_year - contract_orders_per_year) * admin_savings_per_order
        
        # Total annual savings
        price_savings = (current_avg_price - contract_price) * annual_volume
        total_savings = price_savings + total_admin_savings
        
        savings_scenarios.append({
            'contract_term': term['name'],
            'contract_price': contract_price,
            'price_savings': price_savings,
            'admin_savings': total_admin_savings,
            'total_savings': total_savings,
            'savings_percent': (total_savings / (current_avg_price * annual_volume)) * 100 if current_avg_price * annual_volume > 0 else 0
        })
    
    return savings_scenarios

def display(df):
    st.header("🤝 Contracting Opportunities")
    st.markdown("Identify optimal contracting opportunities based on spend analysis, vendor performance, and demand predictability.")
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Line Total', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Line Total'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date'])
    
    # Calculate unit price from line total and quantity
    df_clean['Unit Price'] = df_clean['Line Total'] / df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Contract Identification", "📊 Vendor Performance", "💰 Savings Analysis", "📋 Contract Portfolio", "⚙️ Contract Strategy"])
    
    with tab1:
        st.subheader("🎯 Contract Opportunity Identification")
        
        # Configuration parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_spend = st.number_input("Min Annual Spend Threshold (SAR)", min_value=0, value=50000, step=10000)
        with col2:
            min_frequency = st.number_input("Min Annual Order Frequency", min_value=1, value=6, step=1)
        with col3:
            analysis_period = st.selectbox("Analysis Period", ["All Data", "Last 12 Months", "Last 6 Months"])
        
        # Filter data by period
        if analysis_period == "Last 12 Months":
            cutoff_date = df_clean['Creation Date'].max() - timedelta(days=365)
            analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
        elif analysis_period == "Last 6 Months":
            cutoff_date = df_clean['Creation Date'].max() - timedelta(days=180)
            analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
        else:
            analysis_df = df_clean
        
        if st.button("🔍 Identify Contract Opportunities", type="primary"):
            try:
                with st.spinner("Analyzing contract opportunities..."):
                    contract_opportunities = []
                    
                    # Analyze vendor-item combinations
                    vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
                    
                    for (vendor, item), group_data in vendor_item_combinations:
                        try:
                            suitability_analysis = analyze_contract_suitability(
                                group_data, min_spend, min_frequency
                            )
                            
                            if suitability_analysis['recommendation'] != "Not Suitable":
                                vendor_performance = calculate_vendor_performance_score(group_data)
                                
                                item_desc = group_data['Item Description'].iloc[0] if 'Item Description' in group_data.columns else f"Item {item}"
                                
                                contract_opportunities.append({
                                    'Vendor Name': vendor,
                                    'Item': item,
                                    'Item Description': item_desc[:50] + "..." if len(str(item_desc)) > 50 else str(item_desc),
                                    'Annual Spend': suitability_analysis['total_spend'],
                                    'Order Frequency': suitability_analysis['order_frequency'],
                                    'Monthly Frequency': suitability_analysis['monthly_frequency'],
                                    'Demand Predictability': suitability_analysis['demand_predictability'],
                                    'Vendor Performance': vendor_performance['overall_score'],
                                    'Suitability Score': suitability_analysis['suitability_score'],
                                    'Contract Priority': suitability_analysis['recommendation'],
                                    'Avg Unit Price': group_data['Unit Price'].mean(),
                                    'Price Stability': vendor_performance['price_stability']
                                })
                        except Exception as e:
                            st.warning(f"Could not analyze {vendor} - {item}: {str(e)}")
                            continue
                
                if contract_opportunities:
                    opportunities_df = pd.DataFrame(contract_opportunities)
                    opportunities_df = opportunities_df.sort_values(['Suitability Score', 'Annual Spend'], ascending=[False, False])
                    
                    # Summary metrics
                    total_contract_spend = opportunities_df['Annual Spend'].sum()
                    high_priority_count = len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])
                    avg_suitability = opportunities_df['Suitability Score'].mean()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Opportunities", len(opportunities_df))
                    with col2:
                        st.metric("High Priority Items", high_priority_count)
                    with col3:
                        st.metric("Total Contract Spend", f"SAR {total_contract_spend:,.0f}")
                    with col4:
                        st.metric("Avg Suitability Score", f"{avg_suitability:.2f}")
                    
                    # Priority distribution
                    priority_counts = opportunities_df['Contract Priority'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            values=priority_counts.values,
                            names=priority_counts.index,
                            title="Contract Priority Distribution",
                            color_discrete_map={
                                'High Priority': '#ff6b6b',
                                'Medium Priority': '#ffd93d',
                                'Low Priority': '#6bcf7f'
                            }
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Top opportunities by spend
                        top_by_spend = opportunities_df.nlargest(10, 'Annual Spend')
                        
                        fig = px.bar(
                            top_by_spend,
                            x='Annual Spend',
                            y='Vendor Name',
                            color='Contract Priority',
                            title="Top 10 Opportunities by Spend",
                            orientation='h',
                            color_discrete_map={
                                'High Priority': '#ff6b6b',
                                'Medium Priority': '#ffd93d',
                                'Low Priority': '#6bcf7f'
                            }
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results table
                    st.subheader("📋 Contract Opportunities Details")
                    
                    st.dataframe(
                        opportunities_df.style.format({
                            'Annual Spend': 'SAR {:,.0f}',
                            'Monthly Frequency': '{:.1f}',
                            'Demand Predictability': '{:.2f}',
                            'Vendor Performance': '{:.2f}',
                            'Suitability Score': '{:.2f}',
                            'Avg Unit Price': 'SAR {:.2f}',
                            'Price Stability': '{:.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Opportunity matrix
                    st.subheader("📊 Contract Opportunity Matrix")
                    
                    fig = px.scatter(
                        opportunities_df,
                        x='Suitability Score',
                        y='Vendor Performance',
                        size='Annual Spend',
                        color='Contract Priority',
                        hover_data=['Vendor Name', 'Item', 'Annual Spend'],
                        title="Contract Opportunity Matrix: Suitability vs Vendor Performance",
                        labels={
                            'Suitability Score': 'Contract Suitability Score',
                            'Vendor Performance': 'Vendor Performance Score'
                        },
                        color_discrete_map={
                            'High Priority': '#ff6b6b',
                            'Medium Priority': '#ffd93d',
                            'Low Priority': '#6bcf7f'
                        }
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export opportunities
                    csv = opportunities_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Contract Opportunities",
                        data=csv,
                        file_name=f"contract_opportunities_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Store results for other tabs
                    st.session_state['contract_opportunities'] = opportunities_df
                
                else:
                    st.info("No contract opportunities found with the current criteria.")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Please check your data and try adjusting the parameters.")
    
    with tab2:
        st.subheader("📊 Vendor Performance Analysis")
        
        # Vendor selection
        vendor_options = sorted(df_clean["Vendor Name"].dropna().unique())
        
        # Set safe defaults - only use first 5 vendors if they exist
        safe_defaults = vendor_options[:min(5, len(vendor_options))] if len(vendor_options) > 0 else []
        
        selected_vendors = st.multiselect(
            "Select Vendors for Performance Analysis",
            options=vendor_options,
            default=safe_defaults,
            key="performance_vendors"
        )
        
        if selected_vendors:
            vendor_performance_results = []
            
            for vendor in selected_vendors:
                vendor_data = df_clean[df_clean["Vendor Name"] == vendor]
                performance_metrics = calculate_vendor_performance_score(vendor_data)
                
                # Additional metrics
                total_spend = vendor_data['Line Total'].sum()
                unique_items = vendor_data['Item'].nunique()
                avg_order_size = vendor_data['Qty Delivered'].mean()
                order_frequency = len(vendor_data)
                
                vendor_performance_results.append({
                    'Vendor Name': vendor,
                    'Overall Score': performance_metrics['overall_score'],
                    'Volume Consistency': performance_metrics['volume_consistency'],
                    'Price Stability': performance_metrics['price_stability'],
                    'Lead Time Consistency': performance_metrics['lead_time_consistency'],
                    'Delivery Performance': performance_metrics['delivery_performance'],
                    'Quality Score': performance_metrics['quality_score'],
                    'Total Spend': total_spend,
                    'Unique Items': unique_items,
                    'Avg Order Size': avg_order_size,
                    'Order Frequency': order_frequency
                })
            
            performance_df = pd.DataFrame(vendor_performance_results)
            performance_df = performance_df.sort_values('Overall Score', ascending=False)
            
            # Performance summary
            col1, col2, col3 = st.columns(3)
            with col1:
                top_performer = performance_df.iloc[0]['Vendor Name']
                top_score = performance_df.iloc[0]['Overall Score']
                st.metric("Top Performer", top_performer, f"{top_score:.2f}")
            with col2:
                avg_performance = performance_df['Overall Score'].mean()
                st.metric("Average Performance", f"{avg_performance:.2f}")
            with col3:
                high_performers = len(performance_df[performance_df['Overall Score'] >= 0.7])
                st.metric("High Performers (≥0.7)", high_performers)
            
            # Performance comparison table
            st.subheader("🏅 Vendor Performance Scorecard")
            
            st.dataframe(
                performance_df.style.format({
                    'Overall Score': '{:.2f}',
                    'Volume Consistency': '{:.2f}',
                    'Price Stability': '{:.2f}',
                    'Lead Time Consistency': '{:.2f}',
                    'Delivery Performance': '{:.2f}',
                    'Quality Score': '{:.2f}',
                    'Total Spend': 'SAR {:,.0f}',
                    'Avg Order Size': '{:.1f}'
                }),
                use_container_width=True
            )
            
            # Performance radar chart
            st.subheader("🎯 Performance Radar Chart")
            
            if len(selected_vendors) <= 5:  # Limit for readability
                categories = ['Volume Consistency', 'Price Stability', 'Lead Time Consistency', 
                             'Delivery Performance', 'Quality Score']
                
                fig = go.Figure()
                
                colors = px.colors.qualitative.Set1
                
                for i, vendor in enumerate(selected_vendors):
                    vendor_data = performance_df[performance_df['Vendor Name'] == vendor].iloc[0]
                    
                    values = [
                        vendor_data['Volume Consistency'],
                        vendor_data['Price Stability'],
                        vendor_data['Lead Time Consistency'],
                        vendor_data['Delivery Performance'],
                        vendor_data['Quality Score']
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=vendor,
                        line=dict(color=colors[i % len(colors)])
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Vendor Performance Comparison",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select 5 or fewer vendors for radar chart visualization.")
            
            # Performance vs spend analysis
            st.subheader("💰 Performance vs Spend Analysis")
            
            fig = px.scatter(
                performance_df,
                x='Overall Score',
                y='Total Spend',
                size='Unique Items',
                color='Overall Score',
                hover_name='Vendor Name',
                title="Vendor Performance vs Total Spend",
                labels={'Overall Score': 'Performance Score', 'Total Spend': 'Total Annual Spend (SAR)'},
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please select vendors for performance analysis.")
    
    with tab3:
        st.subheader("💰 Contract Savings Analysis")
        
        if 'contract_opportunities' in st.session_state:
            opportunities_df = st.session_state['contract_opportunities']
            
            # Select opportunity for detailed analysis
            opportunity_options = [f"{row['Vendor Name']} - Item {row['Item']}" 
                                 for _, row in opportunities_df.iterrows()]
            
            selected_opportunity = st.selectbox(
                "Select Contract Opportunity for Savings Analysis",
                options=opportunity_options,
                key="savings_opportunity"
            )
            
            if selected_opportunity:
                # Parse selection
                vendor_name = selected_opportunity.split(' - Item ')[0]
                item_id = selected_opportunity.split(' - Item ')[1]
                
                # Get historical data
                historical_data = df_clean[
                    (df_clean['Vendor Name'] == vendor_name) & 
                    (df_clean['Item'] == item_id)
                ]
                
                # Contract terms configuration
                st.subheader("⚙️ Contract Terms Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Contract Scenario 1: Short-term**")
                    short_discount = st.slider("Volume Discount (%)", 0, 20, 3, key="short_discount") / 100
                    short_admin_savings = st.number_input("Admin Savings per Order (SAR)", 0, 500, 100, key="short_admin")
                    short_orders_per_year = st.number_input("Contract Orders/Year", 1, 52, 12, key="short_orders")
                
                with col2:
                    st.write("**Contract Scenario 2: Long-term**")
                    long_discount = st.slider("Volume Discount (%)", 0, 30, 8, key="long_discount") / 100
                    long_admin_savings = st.number_input("Admin Savings per Order (SAR)", 0, 500, 200, key="long_admin")
                    long_orders_per_year = st.number_input("Contract Orders/Year", 1, 24, 6, key="long_orders")
                
                # Define contract terms
                contract_terms = [
                    {
                        'name': 'Current (Spot Buy)',
                        'volume_discount': 0,
                        'admin_savings': 0,
                        'orders_per_year': len(historical_data) / ((historical_data['Creation Date'].max() - historical_data['Creation Date'].min()).days / 365)
                    },
                    {
                        'name': 'Short-term Contract',
                        'volume_discount': short_discount,
                        'admin_savings': short_admin_savings,
                        'orders_per_year': short_orders_per_year
                    },
                    {
                        'name': 'Long-term Contract',
                        'volume_discount': long_discount,
                        'admin_savings': long_admin_savings,
                        'orders_per_year': long_orders_per_year
                    }
                ]
                
                # Calculate savings
                savings_analysis = calculate_contract_savings_potential(historical_data, contract_terms)
                savings_df = pd.DataFrame(savings_analysis)
                
                # Display savings comparison
                st.subheader("💵 Savings Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                short_term_savings = savings_df[savings_df['contract_term'] == 'Short-term Contract']['total_savings'].iloc[0]
                long_term_savings = savings_df[savings_df['contract_term'] == 'Long-term Contract']['total_savings'].iloc[0]
                current_annual_cost = historical_data['Line Total'].sum() / ((historical_data['Creation Date'].max() - historical_data['Creation Date'].min()).days / 365)
                
                with col1:
                    st.metric("Current Annual Cost", f"SAR {current_annual_cost:,.0f}")
                with col2:
                    short_percent = (short_term_savings / current_annual_cost) * 100 if current_annual_cost > 0 else 0
                    st.metric("Short-term Savings", f"SAR {short_term_savings:,.0f}", f"{short_percent:.1f}%")
                with col3:
                    long_percent = (long_term_savings / current_annual_cost) * 100 if current_annual_cost > 0 else 0
                    st.metric("Long-term Savings", f"SAR {long_term_savings:,.0f}", f"{long_percent:.1f}%")
                
                # Detailed savings breakdown
                st.dataframe(
                    savings_df.style.format({
                        'contract_price': 'SAR {:.2f}',
                        'price_savings': 'SAR {:,.0f}',
                        'admin_savings': 'SAR {:,.0f}',
                        'total_savings': 'SAR {:,.0f}',
                        'savings_percent': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Savings visualization
                fig = go.Figure()
                
                categories = ['Price Savings', 'Admin Savings']
                
                for _, row in savings_df.iterrows():
                    if row['contract_term'] != 'Current (Spot Buy)':
                        fig.add_trace(go.Bar(
                            name=row['contract_term'],
                            x=categories,
                            y=[row['price_savings'], row['admin_savings']]
                        ))
                
                fig.update_layout(
                    title="Savings Breakdown by Contract Type",
                    xaxis_title="Savings Category",
                    yaxis_title="Annual Savings (SAR)",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ROI analysis
                st.subheader("📈 Contract ROI Analysis")
                
                # Simplified ROI calculation (assuming contract setup costs)
                contract_setup_cost = st.number_input("Estimated Contract Setup Cost (SAR)", min_value=0, value=10000)
                
                roi_data = []
                for _, row in savings_df.iterrows():
                    if row['contract_term'] != 'Current (Spot Buy)':
                        payback_months = (contract_setup_cost / row['total_savings'] * 12) if row['total_savings'] > 0 else float('inf')
                        three_year_roi = ((row['total_savings'] * 3 - contract_setup_cost) / contract_setup_cost * 100) if contract_setup_cost > 0 else 0
                        
                        roi_data.append({
                            'Contract Type': row['contract_term'],
                            'Annual Savings': row['total_savings'],
                            'Setup Cost': contract_setup_cost,
                            'Payback (Months)': payback_months,
                            '3-Year ROI (%)': three_year_roi
                        })
                
                roi_df = pd.DataFrame(roi_data)
                
                st.dataframe(
                    roi_df.style.format({
                        'Annual Savings': 'SAR {:,.0f}',
                        'Setup Cost': 'SAR {:,.0f}',
                        'Payback (Months)': '{:.1f}',
                        '3-Year ROI (%)': '{:.0f}%'
                    }),
                    use_container_width=True
                )
        else:
            st.info("Run contract opportunity identification first to see savings analysis.")
    
    with tab4:
        st.subheader("📋 Contract Portfolio Management")
        
        if 'contract_opportunities' in st.session_state:
            opportunities_df = st.session_state['contract_opportunities']
            
            # Portfolio overview
            st.write("**Contract Portfolio Overview:**")
            
            # Prioritize contracts by value and suitability
            high_priority = opportunities_df[opportunities_df['Contract Priority'] == 'High Priority']
            medium_priority = opportunities_df[opportunities_df['Contract Priority'] == 'Medium Priority']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("High Priority Contracts", len(high_priority))
                st.metric("Total High Priority Spend", f"SAR {high_priority['Annual Spend'].sum():,.0f}")
            
            with col2:
                st.metric("Medium Priority Contracts", len(medium_priority))
                st.metric("Total Medium Priority Spend", f"SAR {medium_priority['Annual Spend'].sum():,.0f}")
            
            # Contract implementation roadmap
            st.subheader("🗺️ Implementation Roadmap")
            
            # Sort by priority and spend for implementation sequence
            implementation_order = opportunities_df.sort_values(
                ['Suitability Score', 'Annual Spend'], 
                ascending=[False, False]
            ).head(20)  # Top 20 for roadmap
            
            # Add implementation phases
            implementation_order['Implementation Phase'] = pd.cut(
                range(len(implementation_order)),
                bins=4,
                labels=['Phase 1 (0-3 months)', 'Phase 2 (3-6 months)', 
                       'Phase 3 (6-9 months)', 'Phase 4 (9-12 months)']
            )
            
            # Implementation timeline
            phase_summary = implementation_order.groupby('Implementation Phase').agg({
                'Annual Spend': 'sum',
                'Vendor Name': 'count'
            }).round(0)
            phase_summary.columns = ['Total Spend (SAR)', 'Number of Contracts']
            
            st.dataframe(phase_summary, use_container_width=True)
            
            # Detailed roadmap
            st.subheader("📅 Detailed Roadmap")
            
            roadmap_display = implementation_order[[
                'Vendor Name', 'Item', 'Annual Spend', 'Contract Priority', 
                'Suitability Score', 'Implementation Phase'
            ]]
            
            st.dataframe(
                roadmap_display.style.format({
                    'Annual Spend': 'SAR {:,.0f}',
                    'Suitability Score': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Resource planning
            st.subheader("📊 Resource Planning")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Contracts by vendor
                vendor_contract_count = opportunities_df['Vendor Name'].value_counts().head(10)
                
                fig = px.bar(
                    x=vendor_contract_count.index,
                    y=vendor_contract_count.values,
                    title="Contracts by Vendor (Top 10)",
                    labels={'x': 'Vendor', 'y': 'Number of Contracts'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Implementation effort estimation
                effort_by_phase = implementation_order.groupby('Implementation Phase').size()
                
                fig = px.pie(
                    values=effort_by_phase.values,
                    names=effort_by_phase.index,
                    title="Contract Distribution by Phase"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            st.subheader("⚠️ Portfolio Risk Assessment")
            
            risk_factors = []
            
            # Vendor concentration risk
            vendor_spend_concentration = opportunities_df.groupby('Vendor Name')['Annual Spend'].sum()
            total_portfolio_spend = vendor_spend_concentration.sum()
            top_vendor_concentration = (vendor_spend_concentration.max() / total_portfolio_spend) * 100
            
            if top_vendor_concentration > 30:
                risk_factors.append({
                    'Risk Type': 'Vendor Concentration',
                    'Risk Level': 'High' if top_vendor_concentration > 50 else 'Medium',
                    'Description': f"Top vendor represents {top_vendor_concentration:.1f}% of contract spend",
                    'Mitigation': 'Diversify supplier base, develop backup suppliers'
                })
            
            # Performance risk
            low_performance_contracts = opportunities_df[opportunities_df['Vendor Performance'] < 0.6]
            if len(low_performance_contracts) > 0:
                risk_factors.append({
                    'Risk Type': 'Vendor Performance',
                    'Risk Level': 'Medium',
                    'Description': f"{len(low_performance_contracts)} contracts with vendors scoring <0.6",
                    'Mitigation': 'Implement performance improvement plans, consider alternative suppliers'
                })
            
            # Market risk for high spend items
            high_spend_items = opportunities_df[opportunities_df['Annual Spend'] > opportunities_df['Annual Spend'].quantile(0.8)]
            if len(high_spend_items) > 0:
                risk_factors.append({
                    'Risk Type': 'Market Risk',
                    'Risk Level': 'Medium',
                    'Description': f"{len(high_spend_items)} high-spend items may be subject to market volatility",
                    'Mitigation': 'Include price adjustment clauses, monitor market conditions'
                })
            
            if risk_factors:
                risk_df = pd.DataFrame(risk_factors)
                st.dataframe(risk_df, use_container_width=True)
            else:
                st.success("No significant portfolio risks identified.")
        
        else:
            st.info("Run contract opportunity identification first to see portfolio management.")
    
    with tab5:
        st.subheader("⚙️ Contract Strategy & Best Practices")
        
        # Strategic guidelines
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 **Contract Selection Criteria**")
            st.write("**High Priority Items:**")
            st.write("• Annual spend > SAR 50,000")
            st.write("• Regular, predictable demand")
            st.write("• Stable supplier relationship")
            st.write("• Limited supplier options")
            st.write("")
            st.write("**Medium Priority Items:**")
            st.write("• Annual spend SAR 10,000 - 50,000")
            st.write("• Moderate demand variability")
            st.write("• Good supplier performance")
            st.write("• Some price volatility")
        
        with col2:
            st.markdown("#### 📋 **Contract Terms Framework**")
            st.write("**Price Terms:**")
            st.write("• Volume-based discounts")
            st.write("• Price adjustment mechanisms")
            st.write("• Market price protections")
            st.write("")
            st.write("**Performance Terms:**")
            st.write("• Delivery requirements")
            st.write("• Quality specifications")
            st.write("• Service level agreements")
            st.write("• Performance penalties/incentives")
        
        # Contract types
        st.subheader("📄 Contract Types & Applications")
        
        contract_types = [
            {
                "Type": "Fixed Price Contract",
                "Best For": "Stable demand, predictable costs",
                "Duration": "6-18 months",
                "Risk Level": "Low",
                "Savings Potential": "3-8%"
            },
            {
                "Type": "Volume Commitment",
                "Best For": "High volume, regular demand",
                "Duration": "12-36 months",
                "Risk Level": "Medium",
                "Savings Potential": "5-15%"
            },
            {
                "Type": "Blanket Purchase Order",
                "Best For": "Multiple items, same supplier",
                "Duration": "12-24 months",
                "Risk Level": "Low",
                "Savings Potential": "2-6%"
            },
            {
                "Type": "Requirements Contract",
                "Best For": "Uncertain volumes, guaranteed supply",
                "Duration": "12-36 months",
                "Risk Level": "Medium",
                "Savings Potential": "4-10%"
            }
        ]
        
        contract_types_df = pd.DataFrame(contract_types)
        st.dataframe(contract_types_df, use_container_width=True)
        
        # Implementation best practices
        st.subheader("💡 Implementation Best Practices")
        
        practices_col1, practices_col2 = st.columns(2)
        
        with practices_col1:
            st.markdown("#### ✅ **Do's:**")
            st.write("• Conduct thorough supplier due diligence")
            st.write("• Define clear performance metrics")
            st.write("• Include termination clauses")
            st.write("• Regular contract reviews")
            st.write("• Document all changes")
            st.write("• Maintain supplier relationships")
        
        with practices_col2:
            st.markdown("#### ❌ **Don'ts:**")
            st.write("• Lock in without market research")
            st.write("• Ignore performance monitoring")
            st.write("• Overlook legal compliance")
            st.write("• Neglect backup suppliers")
            st.write("• Skip regular price benchmarking")
            st.write("• Ignore contract renewal planning")
        
        # KPIs and metrics
        st.subheader("📊 Key Performance Indicators")
        
        kpi_categories = {
            "Financial KPIs": [
                "Cost savings achieved vs target",
                "Price variance from market",
                "Total cost of ownership reduction",
                "Contract compliance rate"
            ],
            "Operational KPIs": [
                "On-time delivery rate",
                "Quality performance",
                "Order cycle time",
                "Supplier responsiveness"
            ],
            "Strategic KPIs": [
                "Supplier relationship score",
                "Innovation contribution",
                "Risk mitigation effectiveness",
                "Market intelligence value"
            ]
        }
        
        for category, kpis in kpi_categories.items():
            st.write(f"**{category}:**")
            for kpi in kpis:
                st.write(f"• {kpi}")
            st.write("")
        
        # Contract lifecycle
        st.subheader("🔄 Contract Lifecycle Management")
        
        lifecycle_stages = [
            {"Stage": "Planning", "Duration": "2-4 weeks", "Key Activities": "Market analysis, supplier evaluation, term negotiation"},
            {"Stage": "Execution", "Duration": "2-6 weeks", "Key Activities": "Legal review, approvals, contract signing"},
            {"Stage": "Management", "Duration": "Contract term", "Key Activities": "Performance monitoring, relationship management"},
            {"Stage": "Renewal/Exit", "Duration": "4-8 weeks", "Key Activities": "Performance review, renegotiation, transition planning"}
        ]
        
        lifecycle_df = pd.DataFrame(lifecycle_stages)
        st.dataframe(lifecycle_df, use_container_width=True)
        
        # Export strategy guide
        if st.button("📥 Export Contract Strategy Guide"):
            strategy_guide = {
                'section': ['Selection Criteria', 'Contract Types', 'Best Practices', 'KPIs', 'Lifecycle'],
                'content': [
                    'High Priority: >SAR 50K annual spend, predictable demand, stable supplier',
                    'Fixed Price (6-18mo), Volume Commitment (12-36mo), BPO (12-24mo)',
                    'Due diligence, clear metrics, termination clauses, regular reviews',
                    'Cost savings, delivery rate, quality performance, supplier score',
                    'Planning→Execution→Management→Renewal (2-52 weeks per stage)'
                ],
                'timeline': ['Pre-contract', 'Contract Design', 'Implementation', 'Ongoing', 'End of Term'],
                'responsibility': ['Procurement', 'Legal/Procurement', 'Operations', 'All Functions', 'Procurement']
            }
            
            guide_df = pd.DataFrame(strategy_guide)
            csv = guide_df.to_csv(index=False)
            
            st.download_button(
                label="Download Strategy Guide",
                data=csv,
                file_name=f"contract_strategy_guide_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
