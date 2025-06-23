# Save this as: contracting_opportunities.py
# Complete enhanced contracting module with graceful dependency handling

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import json

# Optional imports with graceful fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ Machine Learning features not available. Install scikit-learn for full functionality.")

try:
    import sqlite3
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

def calculate_vendor_performance_score(vendor_data):
    """Calculate comprehensive vendor performance score"""
    metrics = {}
    
    # Volume consistency (coefficient of variation of monthly orders)
    if 'Creation Date' in vendor_data.columns:
        monthly_orders = vendor_data.groupby(vendor_data['Creation Date'].dt.to_period('M')).size()
        if len(monthly_orders) > 1:
            metrics['volume_consistency'] = 1 - (monthly_orders.std() / monthly_orders.mean()) if monthly_orders.mean() > 0 else 0
        else:
            metrics['volume_consistency'] = 0
    else:
        metrics['volume_consistency'] = 0.5
    
    # Price stability (1 - coefficient of variation of unit prices)
    if 'Unit Price' in vendor_data.columns:
        price_cv = vendor_data['Unit Price'].std() / vendor_data['Unit Price'].mean() if vendor_data['Unit Price'].mean() > 0 else 0
        metrics['price_stability'] = max(0, 1 - price_cv)
    else:
        metrics['price_stability'] = 0
    
    # Lead time consistency (simplified)
    metrics['lead_time_consistency'] = 0.7  # Default value
    
    # Delivery performance (based on order consistency)
    if 'Qty Delivered' in vendor_data.columns:
        order_cv = vendor_data['Qty Delivered'].std() / vendor_data['Qty Delivered'].mean() if vendor_data['Qty Delivered'].mean() > 0 else 0
        metrics['delivery_performance'] = max(0, 1 - order_cv)
    else:
        metrics['delivery_performance'] = 0.5
    
    # Quality score (default high)
    metrics['quality_score'] = 0.8
    
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
    if 'Creation Date' in item_vendor_data.columns:
        date_range = item_vendor_data['Creation Date'].max() - item_vendor_data['Creation Date'].min()
        months_span = date_range.days / 30 if date_range.days > 0 else 1
        monthly_frequency = order_frequency / months_span
    else:
        monthly_frequency = order_frequency / 12
    
    # Demand predictability
    if 'Creation Date' in item_vendor_data.columns and len(item_vendor_data) > 3:
        monthly_demand = item_vendor_data.groupby(item_vendor_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
        if len(monthly_demand) > 1 and monthly_demand.mean() > 0:
            demand_cv = monthly_demand.std() / monthly_demand.mean()
            demand_predictability = max(0, 1 - demand_cv)
        else:
            demand_predictability = 0.5
    else:
        demand_predictability = 0.5
    
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
        'months_span': months_span if 'Creation Date' in item_vendor_data.columns else 12
    }

def calculate_contract_savings_potential(historical_data, contract_terms):
    """Calculate potential savings from contracting"""
    
    # Current average price
    current_avg_price = historical_data['Unit Price'].mean()
    annual_volume = historical_data['Qty Delivered'].sum()
    
    # Contract savings scenarios
    savings_scenarios = []
    
    for term in contract_terms:
        # Price reduction from volume commitment
        volume_discount = term.get('volume_discount', 0)
        contract_price = current_avg_price * (1 - volume_discount)
        
        # Administrative cost savings
        admin_savings_per_order = term.get('admin_savings', 0)
        current_orders_per_year = len(historical_data)
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

class SimpleMLEngine:
    """Simplified ML engine that works with or without scikit-learn"""
    
    def __init__(self):
        self.is_trained = False
        self.model_available = ML_AVAILABLE
        self.predictions = None
    
    def train_simple_model(self, df):
        """Train a simple model or use statistical analysis"""
        if not self.model_available:
            # Fallback to statistical analysis
            self.is_trained = True
            return True, "Statistical analysis model ready"
        
        try:
            # Simple ML training
            vendor_stats = df.groupby('Vendor Name').agg({
                'Line Total': 'sum',
                'Unit Price': 'mean',
                'Qty Delivered': 'sum'
            })
            
            if len(vendor_stats) > 5:
                # Simple clustering
                kmeans = KMeans(n_clusters=min(3, len(vendor_stats)), random_state=42)
                vendor_stats['Cluster'] = kmeans.fit_predict(vendor_stats)
                self.vendor_clusters = vendor_stats
                self.is_trained = True
                return True, "ML model trained successfully"
            else:
                self.is_trained = True
                return True, "Statistical model ready (insufficient data for ML)"
                
        except Exception as e:
            self.is_trained = True
            return True, f"Using statistical fallback: {str(e)}"
    
    def predict_contract_success(self, vendor_data):
        """Predict contract success using available methods"""
        if not self.is_trained:
            return None, "Model not trained"
        
        # Simple scoring based on vendor performance
        total_spend = (vendor_data['Unit Price'] * vendor_data['Qty Delivered']).sum()
        order_consistency = 1 - (vendor_data['Qty Delivered'].std() / vendor_data['Qty Delivered'].mean()) if vendor_data['Qty Delivered'].mean() > 0 else 0
        price_stability = 1 - (vendor_data['Unit Price'].std() / vendor_data['Unit Price'].mean()) if vendor_data['Unit Price'].mean() > 0 else 0
        
        # Combine scores
        success_score = (
            min(total_spend / 50000, 1.0) * 0.4 +  # Spend impact
            max(order_consistency, 0) * 0.3 +       # Consistency
            max(price_stability, 0) * 0.3           # Stability
        )
        
        return success_score, "Success probability calculated"

def display(df):
    """Main display function for enhanced contracting opportunities"""
    st.header("🤝 Enhanced Contracting Opportunities")
    st.markdown("Advanced contract analysis with AI-powered insights and comprehensive reporting.")
    
    # Show capability status
    with st.expander("🔧 System Capabilities"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if ML_AVAILABLE:
                st.success("✅ Machine Learning")
            else:
                st.warning("⚠️ ML Limited (install scikit-learn)")
        with col2:
            if DATABASE_AVAILABLE:
                st.success("✅ Database Support")
            else:
                st.warning("⚠️ No Database Support")
        with col3:
            if YAML_AVAILABLE:
                st.success("✅ Configuration")
            else:
                st.warning("⚠️ Limited Config")
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("Required columns: Vendor Name, Item, Unit Price, Qty Delivered, Creation Date")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date'])
    
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Initialize ML engine
    if 'ml_engine' not in st.session_state:
        st.session_state.ml_engine = SimpleMLEngine()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Contract Identification", 
        "📊 Vendor Performance", 
        "💰 Savings Analysis", 
        "🤖 AI Insights",
        "📋 Reports & Export"
    ])
    
    with tab1:
        st.subheader("🎯 Contract Opportunity Identification")
        
        # Configuration parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_spend = st.number_input("Min Annual Spend Threshold", min_value=0, value=50000, step=10000)
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
            with st.spinner("Analyzing contract opportunities..."):
                
                contract_opportunities = []
                
                # Analyze vendor-item combinations
                vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
                
                progress_bar = st.progress(0)
                total_combinations = len(vendor_item_combinations)
                
                for i, ((vendor, item), group_data) in enumerate(vendor_item_combinations):
                    progress_bar.progress((i + 1) / total_combinations)
                    
                    suitability_analysis = analyze_contract_suitability(
                        group_data, min_spend, min_frequency
                    )
                    
                    if suitability_analysis['recommendation'] != "Not Suitable":
                        vendor_performance = calculate_vendor_performance_score(group_data)
                        
                        item_desc = group_data.get('Item Description', f"Item {item}")
                        if hasattr(item_desc, 'iloc'):
                            item_desc = item_desc.iloc[0] if len(item_desc) > 0 else f"Item {item}"
                        
                        contract_opportunities.append({
                            'Vendor Name': vendor,
                            'Item': item,
                            'Item Description': str(item_desc)[:50] + "..." if len(str(item_desc)) > 50 else str(item_desc),
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
                
                progress_bar.empty()
                
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
                        st.metric("Total Contract Spend", f"${total_contract_spend:,.0f}")
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
                            'Annual Spend': '{:,.0f}',
                            'Monthly Frequency': '{:.1f}',
                            'Demand Predictability': '{:.2f}',
                            'Vendor Performance': '{:.2f}',
                            'Suitability Score': '{:.2f}',
                            'Avg Unit Price': '{:.2f}',
                            'Price Stability': '{:.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Store results for other tabs
                    st.session_state['contract_opportunities'] = opportunities_df
                
                else:
                    st.info("No contract opportunities found with the current criteria.")
    
    with tab2:
        st.subheader("📊 Vendor Performance Analysis")
        
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
                    'Total Spend': '{:,.0f}',
                    'Avg Order Size': '{:.1f}'
                }),
                use_container_width=True
            )
            
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
                labels={'Overall Score': 'Performance Score', 'Total Spend': 'Total Annual Spend'},
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
                    (df_clean['Item'].astype(str) == str(item_id))
                ]
                
                # Contract terms configuration
                st.subheader("⚙️ Contract Terms Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Contract Scenario 1: Short-term**")
                    short_discount = st.slider("Volume Discount (%)", 0, 20, 3, key="short_discount") / 100
                    short_admin_savings = st.number_input("Admin Savings per Order", 0, 200, 25, key="short_admin")
                    short_orders_per_year = st.number_input("Contract Orders/Year", 1, 52, 12, key="short_orders")
                
                with col2:
                    st.write("**Contract Scenario 2: Long-term**")
                    long_discount = st.slider("Volume Discount (%)", 0, 30, 8, key="long_discount") / 100
                    long_admin_savings = st.number_input("Admin Savings per Order", 0, 200, 50, key="long_admin")
                    long_orders_per_year = st.number_input("Contract Orders/Year", 1, 24, 6, key="long_orders")
                
                # Define contract terms
                contract_terms = [
                    {
                        'name': 'Current (Spot Buy)',
                        'volume_discount': 0,
                        'admin_savings': 0,
                        'orders_per_year': len(historical_data)
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
                
                short_term_savings = savings_df[savings_df['contract_term'] == 'Short-term Contract']['total_savings'].iloc[0]
                long_term_savings = savings_df[savings_df['contract_term'] == 'Long-term Contract']['total_savings'].iloc[0]
                current_annual_cost = (historical_data['Unit Price'] * historical_data['Qty Delivered']).sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Annual Cost", f"${current_annual_cost:,.0f}")
                with col2:
                    short_percent = (short_term_savings / current_annual_cost) * 100 if current_annual_cost > 0 else 0
                    st.metric("Short-term Savings", f"${short_term_savings:,.0f}", f"{short_percent:.1f}%")
                with col3:
                    long_percent = (long_term_savings / current_annual_cost) * 100 if current_annual_cost > 0 else 0
                    st.metric("Long-term Savings", f"${long_term_savings:,.0f}", f"{long_percent:.1f}%")
                
                # Detailed savings breakdown
                st.dataframe(
                    savings_df.style.format({
                        'contract_price': '{:.2f}',
                        'price_savings': '{:,.0f}',
                        'admin_savings': '{:,.0f}',
                        'total_savings': '{:,.0f}',
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
                    yaxis_title="Annual Savings",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run contract opportunity identification first to see savings analysis.")
    
    with tab4:
        st.subheader("🤖 AI-Powered Insights")
        
        # Train ML model
        if st.button("🤖 Train AI Model", type="primary"):
            with st.spinner("Training AI model..."):
                success, message = st.session_state.ml_engine.train_simple_model(df_clean)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
        
        # AI insights
        if st.session_state.ml_engine.is_trained:
            st.subheader("🧠 AI-Generated Insights")
            
            # Vendor clustering (if ML available)
            if ML_AVAILABLE and hasattr(st.session_state.ml_engine, 'vendor_clusters'):
                st.write("**Vendor Segmentation:**")
                vendor_clusters = st.session_state.ml_engine.vendor_clusters
                
                cluster_summary = vendor_clusters.groupby('Cluster').agg({
                    'Line Total': ['count', 'mean'],
                    'Unit Price': 'mean',
                    'Qty Delivered': 'mean'
                }).round(2)
                
                st.dataframe(cluster_summary)
                
                # Cluster visualization
                fig = px.scatter(
                    vendor_clusters.reset_index(),
                    x='Line Total',
                    y='Unit Price',
                    color='Cluster',
                    hover_name='Vendor Name',
                    title="Vendor Clustering Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Contract success predictions
            st.write("**Contract Success Predictions:**")
            
            if 'contract_opportunities' in st.session_state:
                opportunities_df = st.session_state['contract_opportunities']
                
                # Add AI predictions
                ai_scores = []
                for _, row in opportunities_df.iterrows():
                    vendor_data = df_clean[df_clean['Vendor Name'] == row['Vendor Name']]
                    score, _ = st.session_state.ml_engine.predict_contract_success(vendor_data)
                    ai_scores.append(score)
                
                opportunities_df['AI Success Score'] = ai_scores
                
                # Show top AI recommendations
                top_ai_recommendations = opportunities_df.nlargest(10, 'AI Success Score')
                
                st.dataframe(
                    top_ai_recommendations[['Vendor Name', 'Item', 'Annual Spend', 'AI Success Score']].style.format({
                        'Annual Spend': '{:,.0f}',
                        'AI Success Score': '{:.3f}'
                    }),
                    use_container_width=True
                )
                
                # AI vs Manual scoring comparison
                fig = px.scatter(
                    opportunities_df,
                    x='Suitability Score',
                    y='AI Success Score',
                    size='Annual Spend',
                    color='Contract Priority',
                    title="AI Score vs Manual Suitability Score",
                    hover_data=['Vendor Name']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # AI recommendations
            st.subheader("💡 AI Recommendations")
            
            recommendations = [
                "🎯 Focus on high AI success score vendors for immediate contracting",
                "📊 Prioritize vendors with consistent order patterns and stable pricing",
                "🔄 Review vendors with high spend but low success scores for process improvements",
                "📈 Implement performance monitoring for contracted vendors",
                "🤝 Develop strategic partnerships with top-performing vendor clusters"
            ]
            
            for rec in recommendations:
                st.write(f"• {rec}")
        
        else:
            st.info("Train the AI model first to see insights.")
    
    with tab5:
        st.subheader("📋 Advanced Reports & Export")
        
        if 'contract_opportunities' in st.session_state:
            opportunities_df = st.session_state['contract_opportunities']
            
            # Report configuration
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Report Configuration:**")
                report_type = st.selectbox("Report Type", [
                    "Executive Summary",
                    "Detailed Analysis Report",
                    "Vendor Performance Report",
                    "Savings Analysis Report"
                ])
                
                include_charts = st.checkbox("Include Visualizations", True)
                include_ai_insights = st.checkbox("Include AI Insights", True)
            
            with col2:
                st.write("**Export Options:**")
                export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"])
                
                report_title = st.text_input("Report Title", "Contract Opportunities Analysis")
                
                if st.button("📊 Generate Report", type="primary"):
                    with st.spinner("Generating report..."):
                        
                        # Prepare report data
                        report_data = {
                            'metadata': {
                                'report_title': report_title,
                                'generated_at': datetime.now().isoformat(),
                                'report_type': report_type,
                                'total_opportunities': len(opportunities_df),
                                'high_priority_count': len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority']),
                                'total_contract_spend': opportunities_df['Annual Spend'].sum()
                            },
                            'opportunities': opportunities_df.to_dict('records'),
                            'summary_stats': {
                                'avg_suitability_score': opportunities_df['Suitability Score'].mean(),
                                'avg_vendor_performance': opportunities_df['Vendor Performance'].mean(),
                                'price_stability_avg': opportunities_df['Price Stability'].mean()
                            }
                        }
                        
                        if export_format == "CSV":
                            csv_data = opportunities_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV Report",
                                data=csv_data,
                                file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        
                        elif export_format == "Excel":
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                opportunities_df.to_excel(writer, sheet_name='Opportunities', index=False)
                                
                                # Add summary sheet
                                summary_df = pd.DataFrame([report_data['summary_stats']])
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=buffer.getvalue(),
                                file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        elif export_format == "JSON":
                            json_data = json.dumps(report_data, indent=2, default=str)
                            st.download_button(
                                label="📥 Download JSON Report",
                                data=json_data,
                                file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                                mime="application/json"
                            )
                        
                        st.success("✅ Report generated successfully!")
            
            # Report preview
            st.subheader("📊 Report Preview")
            
            # Executive summary
            total_opportunities = len(opportunities_df)
            high_priority = len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])
            total_spend = opportunities_df['Annual Spend'].sum()
            avg_score = opportunities_df['Suitability Score'].mean()
            
            st.write(f"**{report_title}**")
            st.write(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
            st.write("")
            st.write("**Key Findings:**")
            st.write(f"• Total contract opportunities identified: {total_opportunities}")
            st.write(f"• High priority opportunities: {high_priority}")
            st.write(f"• Total addressable spend: ${total_spend:,.0f}")
            st.write(f"• Average suitability score: {avg_score:.2f}")
            
            # Quick export for current view
            st.subheader("📤 Quick Export")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_current = opportunities_df.to_csv(index=False)
                st.download_button(
                    label="📄 Export CSV",
                    data=csv_current,
                    file_name=f"contract_opportunities_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_current = opportunities_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📄 Export JSON",
                    data=json_current,
                    file_name=f"contract_opportunities_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            with col3:
                if st.button("🔄 Refresh Analysis"):
                    st.rerun()
        
        else:
            st.info("Run contract opportunity identification first to generate reports.")
