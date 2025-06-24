import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Currency symbol mapping
CURRENCY_SYMBOLS = {
    "SAR": "﷼",
    "USD": "$", 
    "EUR": "€",
    "AED": "د.إ",
    "INR": "₹",
    "GBP": "£",
    "JPY": "¥",
    "CAD": "C$",
    "AUD": "A$",
    "CHF": "CHF",
    "CNY": "¥",
    "KWD": "د.ك",
    "BHD": "د.ب",
    "EGP": "ج.م",
    "COP": "$",
    "XAF": "₣",
    "IQD": "ع.د",
    "LYD": "ل.د",
    "MXN": "$"
}

def get_currency_symbol(currency_code):
    """Get currency symbol for display"""
    return CURRENCY_SYMBOLS.get(currency_code, currency_code)

def format_currency_value(value, currency_symbol, decimal_places=0):
    """Format currency value with appropriate symbol"""
    if pd.isna(value):
        return "N/A"
    return f"{currency_symbol}{value:,.{decimal_places}f}"

def get_region_currency_mapping(df):
    """Extract region-currency relationships from data"""
    region_currency_map = {}
    
    # Group by region and get unique currencies
    for region in df['Ou'].dropna().unique():
        region_data = df[df['Ou'] == region]
        currencies = region_data['PO Currency'].dropna().unique()
        region_currency_map[region] = sorted(currencies)
    
    return region_currency_map

def calculate_vendor_performance_score(vendor_data):
    """Calculate comprehensive vendor performance score"""
    metrics = {}
    
    # Volume consistency (coefficient of variation of monthly orders)
    monthly_orders = vendor_data.groupby(vendor_data['Creation Date'].dt.to_period('M')).size()
    if len(monthly_orders) > 1:
        metrics['volume_consistency'] = 1 - (monthly_orders.std() / monthly_orders.mean()) if monthly_orders.mean() > 0 else 0
    else:
        metrics['volume_consistency'] = 0
    
    # Price stability (1 - coefficient of variation of unit prices)
    if 'Unit Price' in vendor_data.columns:
        price_cv = vendor_data['Unit Price'].std() / vendor_data['Unit Price'].mean() if vendor_data['Unit Price'].mean() > 0 else 0
        metrics['price_stability'] = max(0, 1 - price_cv)
    else:
        metrics['price_stability'] = 0
    
    # Lead time consistency (if available)
    if 'lead_time_days' in vendor_data.columns:
        lead_times = vendor_data['lead_time_days'].dropna()
        if len(lead_times) > 0:
            lt_cv = lead_times.std() / lead_times.mean() if lead_times.mean() > 0 else 0
            metrics['lead_time_consistency'] = max(0, 1 - lt_cv)
        else:
            metrics['lead_time_consistency'] = 0.5  # Default neutral score
    else:
        metrics['lead_time_consistency'] = 0.5
    
    # Delivery performance (if available)
    if 'delivery_delay_days' in vendor_data.columns:
        delays = vendor_data['delivery_delay_days'].dropna()
        if len(delays) > 0:
            on_time_rate = len(delays[delays <= 0]) / len(delays)
            metrics['delivery_performance'] = on_time_rate
        else:
            metrics['delivery_performance'] = 0.5
    else:
        metrics['delivery_performance'] = 0.5
    
    # Quality score (based on rejection rates if available)
    if 'Qty Rejected' in vendor_data.columns and 'Qty Delivered' in vendor_data.columns:
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
    total_spend = (item_vendor_data['Unit Price'] * item_vendor_data['Qty Delivered']).sum()
    order_frequency = len(item_vendor_data)
    
    # Calculate time span
    date_range = item_vendor_data['Creation Date'].max() - item_vendor_data['Creation Date'].min()
    months_span = date_range.days / 30 if date_range.days > 0 else 1
    monthly_frequency = order_frequency / months_span
    
    # Demand predictability
    monthly_demand = item_vendor_data.groupby(item_vendor_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
    demand_cv = monthly_demand.std() / monthly_demand.mean() if len(monthly_demand) > 1 and monthly_demand.mean() > 0 else 1
    demand_predictability = max(0, 1 - demand_cv)
    
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
            'savings_percent': (total_savings / (current_avg_price * annual_volume)) * 100
        })
    
    return savings_scenarios

def display(df):
    st.header("🤝 Contracting Opportunities")
    st.markdown("Identify optimal contracting opportunities based on spend analysis, vendor performance, and demand predictability.")
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date', 'Ou', 'PO Currency']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Get region-currency mapping
    region_currency_map = get_region_currency_mapping(df)
    
    if not region_currency_map:
        st.error("No valid region-currency data found.")
        return
    
    # Region and Currency Selection
    st.subheader("🌍 Region & Currency Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Region selection
        available_regions = sorted(region_currency_map.keys())
        selected_region = st.selectbox(
            "Select Business Region",
            options=available_regions,
            key="region_selector"
        )
    
    with col2:
        # Currency selection based on selected region
        if selected_region:
            available_currencies = region_currency_map[selected_region]
            selected_currency = st.selectbox(
                "Select Currency",
                options=available_currencies,
                key="currency_selector"
            )
        else:
            st.info("Please select a region first")
            return
    
    # Display region-currency info
    if selected_region and selected_currency:
        currency_symbol = get_currency_symbol(selected_currency)
        
        # Filter data by selected region and currency
        df_filtered = df[
            (df['Ou'] == selected_region) & 
            (df['PO Currency'] == selected_currency)
        ].copy()
        
        if len(df_filtered) == 0:
            st.warning(f"No data found for {selected_region} with {selected_currency} currency.")
            return
        
        # Display summary info
        total_records = len(df_filtered)
        unique_vendors = df_filtered['Vendor Name'].nunique()
        unique_items = df_filtered['Item'].nunique()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Region:** {selected_region}")
        with col2:
            st.info(f"**Currency:** {selected_currency} ({currency_symbol})")
        with col3:
            st.info(f"**Records:** {total_records:,}")
        
        # Clean data
        df_clean = df_filtered.dropna(subset=required_columns[:-2])  # Exclude region and currency from cleaning
        df_clean = df_clean[df_clean['Unit Price'] > 0]
        df_clean = df_clean[df_clean['Qty Delivered'] > 0]
        df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Creation Date'])
        
        if len(df_clean) == 0:
            st.warning("No valid data found after cleaning.")
            return
        
        # Overview metrics
        total_spend = (df_clean['Unit Price'] * df_clean['Qty Delivered']).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Spend", format_currency_value(total_spend, currency_symbol))
        with col2:
            st.metric("Unique Vendors", f"{df_clean['Vendor Name'].nunique():,}")
        with col3:
            st.metric("Unique Items", f"{df_clean['Item'].nunique():,}")
        with col4:
            st.metric("Total Orders", f"{len(df_clean):,}")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Contract Identification", "📊 Vendor Performance", "💰 Savings Analysis", "📋 Contract Portfolio", "⚙️ Contract Strategy"])
        
        with tab1:
            st.subheader("🎯 Contract Opportunity Identification")
            
            # Configuration parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_spend = st.number_input("Min Annual Spend Threshold", min_value=0, value=50000, step=10000, key="min_spend_threshold")
            with col2:
                min_frequency = st.number_input("Min Annual Order Frequency", min_value=1, value=6, step=1, key="min_freq_threshold")
            with col3:
                analysis_period = st.selectbox("Analysis Period", ["All Data", "Last 12 Months", "Last 6 Months"], key="analysis_period")
            
            # Display thresholds in selected currency
            st.info(f"**Threshold:** {format_currency_value(min_spend, currency_symbol)} minimum annual spend • {min_frequency} minimum orders per year")
            
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
                with st.spinner("Analyzing contract opportunities..."):
                    contract_opportunities = []
                    
                    # Analyze vendor-item combinations
                    vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
                    
                    for (vendor, item), group_data in vendor_item_combinations:
                        suitability_analysis = analyze_contract_suitability(
                            group_data, min_spend, min_frequency
                        )
                        
                        if suitability_analysis['recommendation'] != "Not Suitable":
                            vendor_performance = calculate_vendor_performance_score(group_data)
                            
                            item_desc = group_data['Item Description'].iloc[0] if 'Item Description' in group_data.columns else f"Item {item}"
                            
                            contract_opportunities.append({
                                'Vendor Name': vendor,
                                'Item': item,
                                'Item Description': item_desc[:50] + "..." if len(item_desc) > 50 else item_desc,
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
                            st.metric("Total Contract Spend", format_currency_value(total_contract_spend, currency_symbol))
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
                                title=f"Top 10 Opportunities by Spend ({currency_symbol})",
                                orientation='h',
                                color_discrete_map={
                                    'High Priority': '#ff6b6b',
                                    'Medium Priority': '#ffd93d',
                                    'Low Priority': '#6bcf7f'
                                }
                            )
                            fig.update_layout(height=400)
                            fig.update_xaxis(title=f"Annual Spend ({currency_symbol})")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results table
                        st.subheader("📋 Contract Opportunities Details")
                        
                        st.dataframe(
                            opportunities_df.style.format({
                                'Annual Spend': lambda x: format_currency_value(x, currency_symbol),
                                'Monthly Frequency': '{:.1f}',
                                'Demand Predictability': '{:.2f}',
                                'Vendor Performance': '{:.2f}',
                                'Suitability Score': '{:.2f}',
                                'Avg Unit Price': lambda x: format_currency_value(x, currency_symbol, 2),
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
                            hover_name='Vendor Name',
                            hover_data=['Item', 'Annual Spend'],
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
                        
                        # Export opportunities (keep raw numbers for Excel compatibility)
                        csv = opportunities_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Export Contract Opportunities",
                            data=csv,
                            file_name=f"contract_opportunities_{selected_region}_{selected_currency}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        
                        # Store results for other tabs
                        st.session_state['contract_opportunities'] = opportunities_df
                        st.session_state['selected_region'] = selected_region
                        st.session_state['selected_currency'] = selected_currency
                        st.session_state['currency_symbol'] = currency_symbol
                    
                    else:
                        st.info("No contract opportunities found with the current criteria.")
        
        with tab2:
            st.subheader("📊 Vendor Performance Analysis")
            
            # Use stored currency info if available
            display_currency = st.session_state.get('selected_currency', selected_currency)
            display_symbol = st.session_state.get('currency_symbol', currency_symbol)
            
            # Vendor selection
            vendor_options = sorted(df_clean["Vendor Name"].dropna().unique())
            selected_vendors = st.multiselect(
                "Select Vendors for Performance Analysis",
                options=vendor_options,
                default=vendor_options[:5] if len(vendor_options) >= 5 else vendor_options,
                key="performance_vendors"
            )
            
            if selected_vendors:
                vendor_performance_results = []
                
                for vendor in selected_vendors:
                    vendor_data = df_clean[df_clean["Vendor Name"] == vendor]
                    performance_metrics = calculate_vendor_performance_score(vendor_data)
                    
                    # Additional metrics
                    total_spend = (vendor_data['Unit Price'] * vendor_data['Qty Delivered']).sum()
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
                        'Total Spend': lambda x: format_currency_value(x, display_symbol),
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
                    title=f"Vendor Performance vs Total Spend ({display_symbol})",
                    labels={
                        'Overall Score': 'Performance Score', 
                        'Total Spend': f'Total Annual Spend ({display_symbol})'
                    },
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=500,
                    yaxis_title=f"Total Annual Spend ({display_symbol})"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Please select vendors for performance analysis.")
        
        with tab3:
            st.subheader("💰 Contract Savings Analysis")
            
            if 'contract_opportunities' in st.session_state:
                opportunities_df = st.session_state['contract_opportunities']
                display_currency = st.session_state.get('selected_currency', selected_currency)
                display_symbol = st.session_state.get('currency_symbol', currency_symbol)
                
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
                    item_id = int(selected_opportunity.split(' - Item ')[1])
                    
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
                        short_admin_savings = st.number_input(f"Admin Savings per Order ({display_symbol})", 0, 200, 25, key="short_admin")
                        short_orders_per_year = st.number_input("Contract Orders/Year", 1, 52, 12, key="short_orders")
                    
                    with col2:
                        st.write("**Contract Scenario 2: Long-term**")
                        long_discount = st.slider("Volume Discount (%)", 0, 30, 8, key="long_discount") / 100
                        long_admin_savings = st.number_input(f"Admin Savings per Order ({display_symbol})", 0, 200, 50, key="long_admin")
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
                    current_annual_cost = (historical_data['Unit Price'] * historical_data['Qty Delivered']).sum() / ((historical_data['Creation Date'].max() - historical_data['Creation Date'].min()).days / 365)
                    
                    with col1:
                        st.metric("Current Annual Cost", format_currency_value(current_annual_cost, display_symbol))
                    with col2:
                        short_percent = (short_term_savings / current_annual_cost) * 100 if current_annual_cost > 0 else 0
                        st.metric("Short-term Savings", format_currency_value(short_term_savings, display_symbol), f"{short_percent:.1f}%")
                    with col3:
                        long_percent = (long_term_savings / current_annual_cost) * 100 if current_annual_cost > 0 else 0
                        st.metric("Long-term Savings", format_currency_value(long_term_savings, display_symbol), f"{long_percent:.1f}%")
                    
                    # Detailed savings breakdown
                    st.dataframe(
                        savings_df.style.format({
                            'contract_price': lambda x: format_currency_value(x, display_symbol, 2),
                            'price_savings': lambda x: format_currency_value(x, display_symbol),
                            'admin_savings': lambda x: format_currency_value(x, display_symbol),
                            'total_savings': lambda x: format_currency_value(x, display_symbol),
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
                        title=f"Savings Breakdown by Contract Type ({display_symbol})",
                        xaxis_title="Savings Category",
                        yaxis_title=f"Annual Savings ({display_symbol})",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ROI analysis
                    st.subheader("📈 Contract ROI Analysis")
                    
                    # Simplified ROI calculation (assuming contract setup costs)
                    contract_setup_cost = st.number_input(f"Estimated Contract Setup Cost ({display_symbol})", min_value=0, value=5000)
                    
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
                            'Annual Savings': lambda x: format_currency_value(x, display_symbol),
                            'Setup Cost': lambda x: format_currency_value(x, display_symbol),
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
                display_currency = st.session_state.get('selected_currency', selected_currency)
                display_symbol = st.session_state.get('currency_symbol', currency_symbol)
                display_region = st.session_state.get('selected_region', selected_region)
                
                # Portfolio overview
                st.write(f"**Contract Portfolio Overview ({display_region} - {display_currency}):**")
                
                # Prioritize contracts by value and suitability
                high_priority = opportunities_df[opportunities_df['Contract Priority'] == 'High Priority']
                medium_priority = opportunities_df[opportunities_df['Contract Priority'] == 'Medium Priority']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("High Priority Contracts", len(high_priority))
                    st.metric("Total High Priority Spend", format_currency_value(high_priority['Annual Spend'].sum(), display_symbol))
                
                with col2:
                    st.metric("Medium Priority Contracts", len(medium_priority))
                    st.metric("Total Medium Priority Spend", format_currency_value(medium_priority['Annual Spend'].sum(), display_symbol))
                
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
                })
                phase_summary.columns = ['Total Spend', 'Number of Contracts']
                
                # Format phase summary for display
                phase_display = phase_summary.copy()
                phase_display['Total Spend'] = phase_display['Total Spend'].apply(
                    lambda x: format_currency_value(x, display_symbol)
                )
                
                st.dataframe(phase_display, use_container_width=True)
                
                # Detailed roadmap
                st.subheader("📅 Detailed Roadmap")
                
                roadmap_display = implementation_order[[
                    'Vendor Name', 'Item', 'Annual Spend', 'Contract Priority', 
                    'Suitability Score', 'Implementation Phase'
                ]].copy()
                
                st.dataframe(
                    roadmap_display.style.format({
                        'Annual Spend': lambda x: format_currency_value(x, display_symbol),
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
                if total_portfolio_spend > 0:
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
                if len(opportunities_df) > 0:
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
                st.write("• Annual spend > $50,000")
                st.write("• Regular, predictable demand")
                st.write("• Stable supplier relationship")
                st.write("• Limited supplier options")
                st.write("")
                st.write("**Medium Priority Items:**")
                st.write("• Annual spend $10,000 - $50,000")
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
            
            # Regional contracting considerations
            if selected_region and selected_currency:
                st.subheader(f"🌍 Regional Strategy: {selected_region}")
                
                region_col1, region_col2 = st.columns(2)
                
                with region_col1:
                    st.markdown("#### 🎯 **Regional Considerations:**")
                    st.write(f"• Primary currency: {selected_currency}")
                    st.write("• Local supplier preferences")
                    st.write("• Regional compliance requirements")
                    st.write("• Market dynamics and competition")
                    st.write("• Currency exchange risk management")
                
                with region_col2:
                    st.markdown("#### 📊 **Regional Best Practices:**")
                    st.write("• Leverage local market knowledge")
                    st.write("• Consider regional payment terms")
                    st.write("• Include currency hedging clauses")
                    st.write("• Monitor regional regulatory changes")
                    st.write("• Build local supplier relationships")
            
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
                    "Contract compliance rate",
                    "Currency exposure management"
                ],
                "Operational KPIs": [
                    "On-time delivery rate",
                    "Quality performance",
                    "Order cycle time",
                    "Supplier responsiveness",
                    "Regional supplier performance"
                ],
                "Strategic KPIs": [
                    "Supplier relationship score",
                    "Innovation contribution",
                    "Risk mitigation effectiveness",
                    "Market intelligence value",
                    "Regional market penetration"
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
                {"Stage": "Planning", "Duration": "2-4 weeks", "Key Activities": "Market analysis, supplier evaluation, term negotiation, regional considerations"},
                {"Stage": "Execution", "Duration": "2-6 weeks", "Key Activities": "Legal review, approvals, contract signing, currency clause finalization"},
                {"Stage": "Management", "Duration": "Contract term", "Key Activities": "Performance monitoring, relationship management, regional compliance"},
                {"Stage": "Renewal/Exit", "Duration": "4-8 weeks", "Key Activities": "Performance review, renegotiation, transition planning, regional analysis"}
            ]
            
            lifecycle_df = pd.DataFrame(lifecycle_stages)
            st.dataframe(lifecycle_df, use_container_width=True)
            
            # Export strategy guide
            if st.button("📥 Export Contract Strategy Guide"):
                strategy_guide = {
                    'section': ['Selection Criteria', 'Contract Types', 'Best Practices', 'Regional Strategy', 'KPIs', 'Lifecycle'],
                    'content': [
                        'High Priority: >$50K annual spend, predictable demand, stable supplier',
                        'Fixed Price (6-18mo), Volume Commitment (12-36mo), BPO (12-24mo)',
                        'Due diligence, clear metrics, termination clauses, regular reviews',
                        f'Region: {selected_region}, Currency: {selected_currency}, Local compliance',
                        'Cost savings, delivery rate, quality performance, supplier score',
                        'Planning→Execution→Management→Renewal (2-52 weeks per stage)'
                    ],
                    'timeline': ['Pre-contract', 'Contract Design', 'Implementation', 'Regional Focus', 'Ongoing', 'End of Term'],
                    'responsibility': ['Procurement', 'Legal/Procurement', 'Operations', 'Regional Manager', 'All Functions', 'Procurement']
                }
                
                guide_df = pd.DataFrame(strategy_guide)
                csv = guide_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Strategy Guide",
                    data=csv,
                    file_name=f"contract_strategy_guide_{selected_region}_{selected_currency}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("👆 Please select both a region and currency to begin the analysis.")
