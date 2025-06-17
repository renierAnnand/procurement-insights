import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def analyze_cross_regional_patterns(df):
    """Analyze patterns across business units and regions for consolidation opportunities"""
    
    # Ensure we have the required columns
    required_cols = ['Vendor Name', 'Item', 'DEP', 'W/H', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Missing columns for cross-regional analysis: {missing_cols}")
        return None
    
    # Create regional aggregations
    regional_analysis = []
    
    # Group by Product Family, Item, and aggregate across regions
    for product_family in df['Product Family'].unique() if 'Product Family' in df.columns else ['All Products']:
        if product_family == 'All Products':
            product_data = df
        else:
            product_data = df[df['Product Family'] == product_family]
        
        for item in product_data['Item'].unique():
            item_data = product_data[product_data['Item'] == item]
            
            # Cross-regional analysis
            regional_stats = item_data.groupby(['DEP', 'W/H']).agg({
                'Vendor Name': 'nunique',
                'Unit Price': ['mean', 'std', 'min', 'max'],
                'Qty Delivered': 'sum',
                'Line Total': 'sum' if 'Line Total' in item_data.columns else lambda x: (item_data['Unit Price'] * item_data['Qty Delivered']).sum()
            }).round(2)
            
            # Calculate consolidation potential
            unique_vendors_total = item_data['Vendor Name'].nunique()
            total_spend = (item_data['Unit Price'] * item_data['Qty Delivered']).sum()
            price_variance = item_data['Unit Price'].std() / item_data['Unit Price'].mean() if item_data['Unit Price'].mean() > 0 else 0
            
            # Calculate demand correlation across regions
            monthly_demand_by_region = item_data.groupby(['DEP', item_data['Creation Date'].dt.to_period('M')])['Qty Delivered'].sum().unstack(level=0, fill_value=0)
            demand_correlation = monthly_demand_by_region.corr().mean().mean() if len(monthly_demand_by_region.columns) > 1 else 1
            
            regional_analysis.append({
                'Product Family': product_family,
                'Item': item,
                'Total Vendors': unique_vendors_total,
                'Total Regions': item_data[['DEP', 'W/H']].drop_duplicates().shape[0],
                'Total Spend': total_spend,
                'Price Variance': price_variance,
                'Demand Correlation': demand_correlation,
                'Consolidation Potential': calculate_consolidation_score(unique_vendors_total, price_variance, total_spend, demand_correlation),
                'Avg Unit Price': item_data['Unit Price'].mean(),
                'Total Volume': item_data['Qty Delivered'].sum()
            })
    
    return pd.DataFrame(regional_analysis).sort_values('Consolidation Potential', ascending=False)

def calculate_consolidation_score(vendor_count, price_variance, total_spend, demand_correlation):
    """Calculate vendor consolidation opportunity score"""
    
    # Higher score for more vendors (more consolidation opportunity)
    vendor_score = min(vendor_count / 10, 1.0)
    
    # Higher score for higher price variance (standardization opportunity)
    price_score = min(price_variance * 2, 1.0)
    
    # Higher score for higher spend (more impact)
    spend_score = min(total_spend / 100000, 1.0)  # Normalize to $100k
    
    # Higher score for higher demand correlation (easier to consolidate)
    correlation_score = max(demand_correlation, 0)
    
    # Weighted combination
    consolidation_score = (vendor_score * 0.3 + price_score * 0.25 + spend_score * 0.3 + correlation_score * 0.15)
    
    return min(consolidation_score, 1.0)

def advanced_demand_forecasting(historical_data, periods_ahead=12):
    """Use ML for demand forecasting instead of simple statistical methods"""
    
    try:
        # Prepare time series data
        historical_data['Month'] = historical_data['Creation Date'].dt.to_period('M')
        monthly_demand = historical_data.groupby('Month')['Qty Delivered'].sum()
        
        if len(monthly_demand) < 6:  # Need minimum data
            return None, None
        
        # Create features for ML model
        demand_df = monthly_demand.reset_index()
        demand_df['Month_Numeric'] = range(len(demand_df))
        demand_df['Month_of_Year'] = demand_df['Month'].apply(lambda x: x.month)
        demand_df['Trend'] = np.arange(len(demand_df))
        
        # Add rolling statistics
        demand_df['MA_3'] = demand_df['Qty Delivered'].rolling(3, min_periods=1).mean()
        demand_df['MA_6'] = demand_df['Qty Delivered'].rolling(6, min_periods=1).mean()
        demand_df['Std_3'] = demand_df['Qty Delivered'].rolling(3, min_periods=1).std()
        
        # Prepare features and target
        feature_cols = ['Month_Numeric', 'Month_of_Year', 'Trend', 'MA_3', 'MA_6', 'Std_3']
        X = demand_df[feature_cols].fillna(method='bfill').fillna(method='ffill')
        y = demand_df['Qty Delivered']
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Generate forecasts
        forecasts = []
        last_month_numeric = demand_df['Month_Numeric'].max()
        
        for i in range(1, periods_ahead + 1):
            next_features = [
                last_month_numeric + i,
                ((demand_df['Month'].iloc[-1].month + i - 1) % 12) + 1,
                len(demand_df) + i,
                demand_df['Qty Delivered'].tail(3).mean(),
                demand_df['Qty Delivered'].tail(6).mean(),
                demand_df['Qty Delivered'].tail(3).std()
            ]
            
            forecast = model.predict([next_features])[0]
            forecasts.append(max(0, forecast))  # Ensure non-negative
        
        # Calculate prediction confidence
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        confidence_score = 1 - (demand_df['Qty Delivered'].std() / demand_df['Qty Delivered'].mean()) if demand_df['Qty Delivered'].mean() > 0 else 0
        
        return forecasts, {
            'confidence': max(0, min(confidence_score, 1)),
            'feature_importance': feature_importance,
            'historical_trend': 'increasing' if demand_df['Qty Delivered'].iloc[-1] > demand_df['Qty Delivered'].iloc[0] else 'decreasing'
        }
        
    except Exception as e:
        st.warning(f"Advanced forecasting failed, using simple method: {str(e)}")
        return None, None

def calculate_vendor_risk_score(vendor_data, market_data=None):
    """Enhanced vendor risk assessment with multiple factors"""
    
    risk_factors = {}
    
    # Financial concentration risk
    spend_concentration = vendor_data['Line Total'].sum() if 'Line Total' in vendor_data.columns else (vendor_data['Unit Price'] * vendor_data['Qty Delivered']).sum()
    risk_factors['financial_concentration'] = min(spend_concentration / 1000000, 1.0)  # Normalize to $1M
    
    # Price volatility risk
    price_cv = vendor_data['Unit Price'].std() / vendor_data['Unit Price'].mean() if vendor_data['Unit Price'].mean() > 0 else 0
    risk_factors['price_volatility'] = min(price_cv, 1.0)
    
    # Delivery consistency risk
    if 'PO Receipt Date' in vendor_data.columns and 'Requested Delivery Date' in vendor_data.columns:
        delivery_variance = ((vendor_data['PO Receipt Date'] - vendor_data['Requested Delivery Date']).dt.days).std()
        risk_factors['delivery_risk'] = min(delivery_variance / 30, 1.0)  # Normalize to 30 days
    else:
        risk_factors['delivery_risk'] = 0.5  # Neutral when data unavailable
    
    # Geographic risk (single supplier)
    unique_regions = vendor_data[['DEP', 'W/H']].drop_duplicates().shape[0] if all(col in vendor_data.columns for col in ['DEP', 'W/H']) else 1
    risk_factors['geographic_risk'] = 1 - min(unique_regions / 5, 1.0)
    
    # Demand variability risk
    monthly_demand = vendor_data.groupby(vendor_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
    demand_cv = monthly_demand.std() / monthly_demand.mean() if len(monthly_demand) > 1 and monthly_demand.mean() > 0 else 0
    risk_factors['demand_variability'] = min(demand_cv, 1.0)
    
    # Market risk (if market data available)
    if market_data:
        # Placeholder for market risk calculation
        risk_factors['market_risk'] = market_data.get('volatility', 0.5)
    else:
        risk_factors['market_risk'] = 0.5
    
    # Calculate weighted risk score
    weights = {
        'financial_concentration': 0.25,
        'price_volatility': 0.20,
        'delivery_risk': 0.20,
        'geographic_risk': 0.15,
        'demand_variability': 0.15,
        'market_risk': 0.05
    }
    
    overall_risk = sum(risk_factors[factor] * weights[factor] for factor in weights)
    
    return {
        'overall_risk': overall_risk,
        **risk_factors
    }

def optimize_contract_terms(historical_data, risk_profile, market_conditions=None):
    """AI-powered contract term optimization"""
    
    annual_spend = (historical_data['Unit Price'] * historical_data['Qty Delivered']).sum()
    annual_volume = historical_data['Qty Delivered'].sum()
    avg_price = historical_data['Unit Price'].mean()
    
    # Calculate optimal contract length
    demand_stability = 1 - (historical_data.groupby(historical_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum().std() / 
                           historical_data.groupby(historical_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum().mean())
    
    if demand_stability > 0.8 and risk_profile['overall_risk'] < 0.4:
        optimal_length = 36  # Long-term for stable, low-risk
    elif demand_stability > 0.6 and risk_profile['overall_risk'] < 0.6:
        optimal_length = 24  # Medium-term
    else:
        optimal_length = 12  # Short-term for volatile or risky
    
    # Calculate optimal volume commitment
    volume_confidence = min(demand_stability * 1.2, 1.0)
    optimal_volume_commitment = annual_volume * volume_confidence * 0.8  # 80% of confident forecast
    
    # Calculate optimal discount rates
    volume_tier_1 = optimal_volume_commitment * 0.5
    volume_tier_2 = optimal_volume_commitment * 0.8
    volume_tier_3 = optimal_volume_commitment
    
    # Risk-adjusted discount rates
    base_discount = 0.03 + (annual_spend / 1000000) * 0.02  # Higher discount for higher spend
    risk_adjustment = risk_profile['overall_risk'] * 0.02  # Reduce discount for higher risk
    
    discount_tier_1 = max(0, base_discount - risk_adjustment)
    discount_tier_2 = max(0, (base_discount * 1.5) - risk_adjustment)
    discount_tier_3 = max(0, (base_discount * 2) - risk_adjustment)
    
    # Payment terms optimization
    if risk_profile['overall_risk'] < 0.3:
        payment_terms = 45  # Longer terms for low-risk suppliers
    elif risk_profile['overall_risk'] < 0.6:
        payment_terms = 30
    else:
        payment_terms = 15  # Shorter terms for high-risk suppliers
    
    # Performance incentives
    performance_bonus = min(0.01 + (1 - risk_profile['overall_risk']) * 0.02, 0.03)
    performance_penalty = min(0.005 + risk_profile['overall_risk'] * 0.015, 0.02)
    
    return {
        'optimal_length_months': optimal_length,
        'volume_commitment': optimal_volume_commitment,
        'volume_tiers': {
            'tier_1': {'volume': volume_tier_1, 'discount': discount_tier_1},
            'tier_2': {'volume': volume_tier_2, 'discount': discount_tier_2},
            'tier_3': {'volume': volume_tier_3, 'discount': discount_tier_3}
        },
        'payment_terms_days': payment_terms,
        'performance_incentives': {
            'bonus_rate': performance_bonus,
            'penalty_rate': performance_penalty
        },
        'price_adjustment_clause': risk_profile['price_volatility'] > 0.6,
        'termination_flexibility': 'high' if risk_profile['overall_risk'] > 0.7 else 'medium' if risk_profile['overall_risk'] > 0.4 else 'low'
    }

def calculate_total_cost_of_ownership(contract_terms, historical_data, risk_profile):
    """Calculate comprehensive TCO including hidden costs"""
    
    annual_volume = historical_data['Qty Delivered'].sum()
    current_avg_price = historical_data['Unit Price'].mean()
    
    # Base contract cost
    contract_price = current_avg_price * (1 - contract_terms['volume_tiers']['tier_2']['discount'])
    base_cost = contract_price * annual_volume
    
    # Transaction costs
    current_orders_per_year = len(historical_data) / 2  # Assuming 2-year historical data
    contract_orders_per_year = max(4, current_orders_per_year * 0.3)  # Fewer orders with contract
    transaction_cost_savings = (current_orders_per_year - contract_orders_per_year) * 50  # $50 per order
    
    # Inventory carrying costs
    safety_stock_reduction = 0.2 * (1 - risk_profile['delivery_risk'])  # Lower risk = less safety stock
    avg_inventory_value = annual_volume * current_avg_price * 0.25  # 25% of annual volume as inventory
    carrying_cost_savings = avg_inventory_value * safety_stock_reduction * 0.25  # 25% carrying cost rate
    
    # Quality costs
    quality_improvement = 0.1 * (1 - risk_profile['overall_risk'])  # Better suppliers = less quality issues
    quality_cost_savings = base_cost * quality_improvement * 0.02  # 2% of spend in quality costs
    
    # Risk costs
    supply_disruption_cost = base_cost * risk_profile['overall_risk'] * 0.05  # 5% potential disruption cost
    price_volatility_cost = base_cost * risk_profile['price_volatility'] * 0.03  # 3% volatility hedge
    
    # Management and monitoring costs
    contract_management_cost = 5000 + (annual_volume * current_avg_price * 0.001)  # Base + 0.1% of spend
    
    tco_components = {
        'base_contract_cost': base_cost,
        'transaction_savings': -transaction_cost_savings,
        'inventory_savings': -carrying_cost_savings,
        'quality_savings': -quality_cost_savings,
        'risk_costs': supply_disruption_cost + price_volatility_cost,
        'management_costs': contract_management_cost
    }
    
    total_tco = sum(tco_components.values())
    current_tco = current_avg_price * annual_volume + (current_orders_per_year * 50) + avg_inventory_value * 0.25
    
    return {
        'total_tco': total_tco,
        'current_tco': current_tco,
        'net_savings': current_tco - total_tco,
        'savings_percentage': ((current_tco - total_tco) / current_tco) * 100 if current_tco > 0 else 0,
        'components': tco_components
    }

def display_enhanced_contracting(df):
    """Enhanced contracting opportunities display"""
    st.header("🤝 Enhanced Contracting Opportunities")
    st.markdown("AI-powered contract identification with cross-regional analysis and advanced optimization.")
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date'])
    
    # Calculate Line Total if missing
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Enhanced Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌍 Cross-Regional Analysis", 
        "🎯 AI Contract Optimization", 
        "📊 Vendor Consolidation", 
        "🔮 Demand Forecasting",
        "⚖️ Risk Assessment", 
        "💰 TCO Analysis"
    ])
    
    with tab1:
        st.subheader("🌍 Cross-Regional Pattern Analysis")
        st.markdown("Identify consolidation opportunities across business units and regions.")
        
        if st.button("🔍 Analyze Cross-Regional Patterns", type="primary"):
            with st.spinner("Analyzing patterns across regions and business units..."):
                
                regional_patterns = analyze_cross_regional_patterns(df_clean)
                
                if regional_patterns is not None and len(regional_patterns) > 0:
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        high_potential = len(regional_patterns[regional_patterns['Consolidation Potential'] > 0.7])
                        st.metric("High Consolidation Potential", high_potential)
                    with col2:
                        total_consolidation_spend = regional_patterns[regional_patterns['Consolidation Potential'] > 0.5]['Total Spend'].sum()
                        st.metric("Total Consolidation Spend", f"${total_consolidation_spend:,.0f}")
                    with col3:
                        avg_vendors_per_item = regional_patterns['Total Vendors'].mean()
                        st.metric("Avg Vendors per Item", f"{avg_vendors_per_item:.1f}")
                    with col4:
                        avg_price_variance = regional_patterns['Price Variance'].mean()
                        st.metric("Avg Price Variance", f"{avg_price_variance:.2%}")
                    
                    # Consolidation opportunities visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Consolidation potential vs spend
                        fig = px.scatter(
                            regional_patterns,
                            x='Consolidation Potential',
                            y='Total Spend',
                            size='Total Vendors',
                            color='Price Variance',
                            hover_data=['Item', 'Total Regions'],
                            title="Consolidation Potential vs Spend",
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Top consolidation opportunities
                        top_opportunities = regional_patterns.nlargest(10, 'Consolidation Potential')
                        
                        fig = px.bar(
                            top_opportunities,
                            x='Consolidation Potential',
                            y='Item',
                            color='Total Spend',
                            title="Top 10 Consolidation Opportunities",
                            orientation='h'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed analysis table
                    st.subheader("📋 Cross-Regional Analysis Details")
                    
                    display_df = regional_patterns.head(20)
                    st.dataframe(
                        display_df.style.format({
                            'Total Spend': '${:,.0f}',
                            'Consolidation Potential': '{:.2f}',
                            'Price Variance': '{:.2%}',
                            'Demand Correlation': '{:.2f}',
                            'Avg Unit Price': '${:.2f}',
                            'Total Volume': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Store results for other tabs
                    st.session_state['regional_analysis'] = regional_patterns
                    
                else:
                    st.info("No cross-regional patterns detected with available data.")
    
    with tab2:
        st.subheader("🎯 AI-Powered Contract Optimization")
        st.markdown("Machine learning-based contract term optimization and risk assessment.")
        
        if 'regional_analysis' in st.session_state:
            regional_data = st.session_state['regional_analysis']
            
            # Select opportunity for detailed optimization
            opportunity_options = [f"Item {row['Item']} - {row['Product Family']}" for _, row in regional_data.head(20).iterrows()]
            
            selected_opportunity = st.selectbox(
                "Select Opportunity for AI Optimization",
                options=opportunity_options,
                key="ai_optimization"
            )
            
            if selected_opportunity and st.button("🤖 Run AI Optimization", type="primary"):
                # Parse selection
                item_id = int(selected_opportunity.split(' - ')[0].replace('Item ', ''))
                
                # Get historical data for the item
                item_data = df_clean[df_clean['Item'] == item_id]
                
                with st.spinner("Running AI optimization algorithms..."):
                    
                    # Calculate risk profile
                    risk_profile = calculate_vendor_risk_score(item_data)
                    
                    # Generate optimal contract terms
                    optimal_terms = optimize_contract_terms(item_data, risk_profile)
                    
                    # Calculate TCO
                    tco_analysis = calculate_total_cost_of_ownership(optimal_terms, item_data, risk_profile)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 Optimized Contract Terms")
                        
                        st.metric("Optimal Contract Length", f"{optimal_terms['optimal_length_months']} months")
                        st.metric("Volume Commitment", f"{optimal_terms['volume_commitment']:,.0f} units")
                        st.metric("Payment Terms", f"{optimal_terms['payment_terms_days']} days")
                        
                        # Volume tiers
                        st.write("**Volume Discount Tiers:**")
                        for tier, terms in optimal_terms['volume_tiers'].items():
                            st.write(f"• {tier.title()}: {terms['volume']:,.0f} units @ {terms['discount']:.1%} discount")
                        
                        # Performance incentives
                        st.write("**Performance Incentives:**")
                        st.write(f"• Bonus Rate: {optimal_terms['performance_incentives']['bonus_rate']:.1%}")
                        st.write(f"• Penalty Rate: {optimal_terms['performance_incentives']['penalty_rate']:.1%}")
                        
                        if optimal_terms['price_adjustment_clause']:
                            st.write("• ⚠️ Price adjustment clause recommended")
                        
                        st.write(f"**Termination Flexibility:** {optimal_terms['termination_flexibility'].title()}")
                    
                    with col2:
                        st.subheader("⚖️ Risk Assessment")
                        
                        # Risk radar chart
                        risk_categories = ['Financial Concentration', 'Price Volatility', 'Delivery Risk', 
                                         'Geographic Risk', 'Demand Variability', 'Market Risk']
                        risk_values = [
                            risk_profile['financial_concentration'],
                            risk_profile['price_volatility'], 
                            risk_profile['delivery_risk'],
                            risk_profile['geographic_risk'],
                            risk_profile['demand_variability'],
                            risk_profile['market_risk']
                        ]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=risk_values,
                            theta=risk_categories,
                            fill='toself',
                            name='Risk Profile',
                            line=dict(color='red')
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1])
                            ),
                            showlegend=True,
                            title="Risk Profile Analysis",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Overall risk rating
                        overall_risk = risk_profile['overall_risk']
                        if overall_risk < 0.3:
                            risk_rating = "🟢 Low Risk"
                        elif overall_risk < 0.6:
                            risk_rating = "🟡 Medium Risk"
                        else:
                            risk_rating = "🔴 High Risk"
                        
                        st.metric("Overall Risk Rating", risk_rating, f"{overall_risk:.2f}")
                    
                    # TCO Analysis
                    st.subheader("💰 Total Cost of Ownership Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current TCO", f"${tco_analysis['current_tco']:,.0f}")
                    with col2:
                        st.metric("Contract TCO", f"${tco_analysis['total_tco']:,.0f}")
                    with col3:
                        st.metric("Net Savings", f"${tco_analysis['net_savings']:,.0f}", 
                                f"{tco_analysis['savings_percentage']:.1f}%")
                    
                    # TCO breakdown
                    tco_components = pd.DataFrame({
                        'Component': ['Contract Cost', 'Transaction Savings', 'Inventory Savings', 
                                    'Quality Savings', 'Risk Costs', 'Management Costs'],
                        'Value': [
                            tco_analysis['components']['base_contract_cost'],
                            tco_analysis['components']['transaction_savings'],
                            tco_analysis['components']['inventory_savings'],
                            tco_analysis['components']['quality_savings'],
                            tco_analysis['components']['risk_costs'],
                            tco_analysis['components']['management_costs']
                        ]
                    })
                    
                    # Waterfall chart
                    fig = go.Figure(go.Waterfall(
                        name="TCO Components",
                        orientation="v",
                        measure=["absolute"] + ["relative"] * 5,
                        x=tco_components['Component'],
                        y=tco_components['Value'],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    
                    fig.update_layout(
                        title="Total Cost of Ownership Breakdown",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please run Cross-Regional Analysis first to enable AI optimization.")
    
    with tab3:
        st.subheader("📊 Vendor Consolidation Analysis")
        
        # Vendor consolidation opportunities
        if st.button("🔍 Analyze Vendor Consolidation", type="primary"):
            with st.spinner("Analyzing vendor consolidation opportunities..."):
                
                # Group by product family and analyze vendor overlap
                consolidation_opportunities = []
                
                for product_family in df_clean['Product Family'].unique() if 'Product Family' in df_clean.columns else ['All Products']:
                    if product_family == 'All Products':
                        family_data = df_clean
                    else:
                        family_data = df_clean[df_clean['Product Family'] == product_family]
                    
                    # Analyze vendor overlap
                    vendor_stats = family_data.groupby('Vendor Name').agg({
                        'Item': 'nunique',
                        'Line Total': 'sum',
                        'Unit Price': 'mean',
                        'DEP': 'nunique',
                        'W/H': 'nunique'
                    }).round(2)
                    
                    vendor_stats.columns = ['Unique Items', 'Total Spend', 'Avg Price', 'Departments', 'Warehouses']
                    vendor_stats['Product Family'] = product_family
                    vendor_stats['Vendor'] = vendor_stats.index
                    
                    # Calculate consolidation score
                    vendor_stats['Consolidation Score'] = (
                        (vendor_stats['Unique Items'] / vendor_stats['Unique Items'].max()) * 0.3 +
                        (vendor_stats['Total Spend'] / vendor_stats['Total Spend'].max()) * 0.4 +
                        (vendor_stats['Departments'] / vendor_stats['Departments'].max()) * 0.3
                    )
                    
                    consolidation_opportunities.extend(vendor_stats.to_dict('records'))
                
                consolidation_df = pd.DataFrame(consolidation_opportunities)
                consolidation_df = consolidation_df.sort_values('Consolidation Score', ascending=False)
                
                # Display consolidation opportunities
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top consolidation candidates
                    top_consolidation = consolidation_df.head(10)
                    
                    fig = px.bar(
                        top_consolidation,
                        x='Consolidation Score',
                        y='Vendor',
                        color='Total Spend',
                        title="Top Vendor Consolidation Candidates",
                        orientation='h'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Vendor portfolio distribution
                    portfolio_summary = consolidation_df.groupby('Product Family').agg({
                        'Total Spend': 'sum',
                        'Vendor': 'count'
                    }).reset_index()
                    
                    fig = px.scatter(
                        portfolio_summary,
                        x='Vendor',
                        y='Total Spend',
                        size='Total Spend',
                        color='Product Family',
                        title="Vendor Portfolio by Product Family"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed consolidation table
                st.subheader("📋 Vendor Consolidation Opportunities")
                
                st.dataframe(
                    consolidation_df.head(20).style.format({
                        'Total Spend': '${:,.0f}',
                        'Avg Price': '${:.2f}',
                        'Consolidation Score': '{:.2f}'
                    }),
                    use_container_width=True
                )
    
    with tab4:
        st.subheader("🔮 Advanced Demand Forecasting")
        
        # Item selection for forecasting
        items = sorted(df_clean['Item'].unique())
        selected_item = st.selectbox("Select Item for Demand Forecasting", items)
        
        if selected_item and st.button("🔮 Generate ML Forecast", type="primary"):
            item_data = df_clean[df_clean['Item'] == selected_item]
            
            with st.spinner("Running machine learning demand forecasting..."):
                
                forecasts, forecast_info = advanced_demand_forecasting(item_data)
                
                if forecasts is not None:
                    # Historical demand
                    monthly_demand = item_data.groupby(item_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
                    
                    # Create forecast periods
                    last_month = monthly_demand.index[-1]
                    forecast_periods = [last_month + i for i in range(1, 13)]
                    
                    # Combine historical and forecast data
                    historical_df = pd.DataFrame({
                        'Period': monthly_demand.index.astype(str),
                        'Demand': monthly_demand.values,
                        'Type': 'Historical'
                    })
                    
                    forecast_df = pd.DataFrame({
                        'Period': [str(p) for p in forecast_periods],
                        'Demand': forecasts,
                        'Type': 'Forecast'
                    })
                    
                    combined_df = pd.concat([historical_df, forecast_df])
                    
                    # Forecast visualization
                    fig = px.line(
                        combined_df,
                        x='Period',
                        y='Demand',
                        color='Type',
                        title=f"Demand Forecast for Item {selected_item}",
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Forecast Confidence", f"{forecast_info['confidence']:.1%}")
                    with col2:
                        avg_forecast = np.mean(forecasts)
                        st.metric("Avg Monthly Forecast", f"{avg_forecast:.0f}")
                    with col3:
                        st.metric("Trend Direction", forecast_info['historical_trend'].title())
                    
                    # Feature importance
                    st.subheader("📊 Forecast Model Insights")
                    
                    importance_df = pd.DataFrame({
                        'Feature': list(forecast_info['feature_importance'].keys()),
                        'Importance': list(forecast_info['feature_importance'].values())
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        title="Feature Importance in Demand Forecasting",
                        orientation='h'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("Unable to generate ML forecast. Insufficient historical data.")
    
    with tab5:
        st.subheader("⚖️ Comprehensive Risk Assessment")
        
        # Risk assessment for all vendors
        if st.button("⚖️ Assess All Vendor Risks", type="primary"):
            with st.spinner("Calculating comprehensive risk scores..."):
                
                vendor_risks = []
                
                for vendor in df_clean['Vendor Name'].unique():
                    vendor_data = df_clean[df_clean['Vendor Name'] == vendor]
                    risk_profile = calculate_vendor_risk_score(vendor_data)
                    
                    total_spend = (vendor_data['Unit Price'] * vendor_data['Qty Delivered']).sum()
                    
                    vendor_risks.append({
                        'Vendor': vendor,
                        'Overall Risk': risk_profile['overall_risk'],
                        'Financial Risk': risk_profile['financial_concentration'],
                        'Price Risk': risk_profile['price_volatility'],
                        'Delivery Risk': risk_profile['delivery_risk'],
                        'Geographic Risk': risk_profile['geographic_risk'],
                        'Demand Risk': risk_profile['demand_variability'],
                        'Total Spend': total_spend,
                        'Risk Category': 'High' if risk_profile['overall_risk'] > 0.6 else 'Medium' if risk_profile['overall_risk'] > 0.3 else 'Low'
                    })
                
                risk_df = pd.DataFrame(vendor_risks).sort_values('Overall Risk', ascending=False)
                
                # Risk summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    high_risk_count = len(risk_df[risk_df['Risk Category'] == 'High'])
                    st.metric("High Risk Vendors", high_risk_count)
                with col2:
                    high_risk_spend = risk_df[risk_df['Risk Category'] == 'High']['Total Spend'].sum()
                    st.metric("High Risk Spend", f"${high_risk_spend:,.0f}")
                with col3:
                    avg_risk = risk_df['Overall Risk'].mean()
                    st.metric("Average Risk Score", f"{avg_risk:.2f}")
                
                # Risk visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk vs spend bubble chart
                    fig = px.scatter(
                        risk_df,
                        x='Overall Risk',
                        y='Total Spend',
                        size='Total Spend',
                        color='Risk Category',
                        hover_name='Vendor',
                        title="Vendor Risk vs Spend Analysis"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk category distribution
                    risk_counts = risk_df['Risk Category'].value_counts()
                    
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Vendor Risk Distribution",
                        color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed risk table
                st.subheader("📋 Vendor Risk Scorecard")
                
                st.dataframe(
                    risk_df.head(20).style.format({
                        'Overall Risk': '{:.2f}',
                        'Financial Risk': '{:.2f}',
                        'Price Risk': '{:.2f}',
                        'Delivery Risk': '{:.2f}',
                        'Geographic Risk': '{:.2f}',
                        'Demand Risk': '{:.2f}',
                        'Total Spend': '${:,.0f}'
                    }),
                    use_container_width=True
                )
    
    with tab6:
        st.subheader("💰 Total Cost of Ownership Analysis")
        st.markdown("Comprehensive TCO analysis including hidden costs and risk adjustments.")
        
        # TCO analysis for selected contracts
        if 'regional_analysis' in st.session_state:
            st.write("Select items for detailed TCO analysis:")
            
            regional_data = st.session_state['regional_analysis']
            selected_items = st.multiselect(
                "Items for TCO Analysis",
                options=[f"Item {row['Item']}" for _, row in regional_data.head(10).iterrows()],
                default=[f"Item {row['Item']}" for _, row in regional_data.head(3).iterrows()]
            )
            
            if selected_items and st.button("💰 Calculate TCO", type="primary"):
                with st.spinner("Calculating comprehensive Total Cost of Ownership..."):
                    
                    tco_results = []
                    
                    for item_str in selected_items:
                        item_id = int(item_str.replace('Item ', ''))
                        item_data = df_clean[df_clean['Item'] == item_id]
                        
                        # Calculate risk and optimization
                        risk_profile = calculate_vendor_risk_score(item_data)
                        optimal_terms = optimize_contract_terms(item_data, risk_profile)
                        tco_analysis = calculate_total_cost_of_ownership(optimal_terms, item_data, risk_profile)
                        
                        tco_results.append({
                            'Item': item_id,
                            'Current TCO': tco_analysis['current_tco'],
                            'Contract TCO': tco_analysis['total_tco'],
                            'Savings': tco_analysis['net_savings'],
                            'Savings %': tco_analysis['savings_percentage'],
                            'Risk Score': risk_profile['overall_risk']
                        })
                    
                    tco_df = pd.DataFrame(tco_results)
                    
                    # TCO summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_savings = tco_df['Savings'].sum()
                        st.metric("Total TCO Savings", f"${total_savings:,.0f}")
                    with col2:
                        avg_savings_pct = tco_df['Savings %'].mean()
                        st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
                    with col3:
                        best_opportunity = tco_df.loc[tco_df['Savings'].idxmax(), 'Item']
                        st.metric("Best Opportunity", f"Item {best_opportunity}")
                    
                    # TCO comparison chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Current TCO',
                        x=[f"Item {item}" for item in tco_df['Item']],
                        y=tco_df['Current TCO'],
                        marker_color='lightcoral'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Contract TCO',
                        x=[f"Item {item}" for item in tco_df['Item']],
                        y=tco_df['Contract TCO'],
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title="TCO Comparison: Current vs Contract",
                        xaxis_title="Items",
                        yaxis_title="Total Cost of Ownership ($)",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed TCO table
                    st.subheader("📋 TCO Analysis Summary")
                    
                    st.dataframe(
                        tco_df.style.format({
                            'Current TCO': '${:,.0f}',
                            'Contract TCO': '${:,.0f}',
                            'Savings': '${:,.0f}',
                            'Savings %': '{:.1f}%',
                            'Risk Score': '{:.2f}'
                        }),
                        use_container_width=True
                    )
        
        else:
            st.info("Please run Cross-Regional Analysis first to enable TCO analysis.")

# Main function to integrate with the app
def display(df):
    display_enhanced_contracting(df)
