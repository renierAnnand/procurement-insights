import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedInventoryOptimizer:
    """Advanced inventory optimization with multiple models and constraints"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def calculate_basic_eoq(self, annual_demand, ordering_cost, holding_cost_rate, unit_cost):
        """Traditional EOQ calculation"""
        holding_cost = unit_cost * holding_cost_rate
        if holding_cost <= 0 or annual_demand <= 0:
            return 0
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return eoq
    
    def calculate_eoq_with_quantity_discounts(self, annual_demand, ordering_cost, holding_cost_rate, discount_structure):
        """EOQ with quantity discount breaks"""
        best_option = None
        min_total_cost = float('inf')
        
        for break_qty, discount_rate, unit_cost in discount_structure:
            # Calculate holding cost with discounted price
            discounted_price = unit_cost * (1 - discount_rate)
            holding_cost = discounted_price * holding_cost_rate
            
            # Calculate EOQ for this price level
            if holding_cost > 0:
                eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                # Use the larger of EOQ or minimum quantity for this discount
                order_quantity = max(eoq, break_qty)
                
                # Calculate total annual cost
                ordering_cost_annual = (annual_demand / order_quantity) * ordering_cost
                holding_cost_annual = (order_quantity / 2) * holding_cost
                purchase_cost_annual = annual_demand * discounted_price
                
                total_cost = ordering_cost_annual + holding_cost_annual + purchase_cost_annual
                
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_option = {
                        'order_quantity': order_quantity,
                        'unit_cost': discounted_price,
                        'total_cost': total_cost,
                        'discount_rate': discount_rate,
                        'break_quantity': break_qty
                    }
        
        return best_option
    
    def calculate_safety_stock(self, lead_time_demand_mean, lead_time_demand_std, service_level):
        """Calculate safety stock for given service level"""
        if lead_time_demand_std <= 0:
            return 0
        
        z_score = stats.norm.ppf(service_level)
        safety_stock = z_score * lead_time_demand_std
        return max(0, safety_stock)
    
    def calculate_reorder_point(self, daily_demand_mean, lead_time_days, lead_time_std, demand_std, service_level):
        """Calculate reorder point with variability in both demand and lead time"""
        
        # Mean lead time demand
        mean_ltd = daily_demand_mean * lead_time_days
        
        # Variance of lead time demand (includes both demand and lead time variability)
        variance_ltd = (lead_time_days * demand_std**2) + (daily_demand_mean**2 * lead_time_std**2)
        std_ltd = np.sqrt(variance_ltd)
        
        # Safety stock
        safety_stock = self.calculate_safety_stock(mean_ltd, std_ltd, service_level)
        
        # Reorder point
        reorder_point = mean_ltd + safety_stock
        
        return {
            'reorder_point': reorder_point,
            'safety_stock': safety_stock,
            'mean_lead_time_demand': mean_ltd,
            'std_lead_time_demand': std_ltd
        }
    
    def abc_xyz_classification(self, items_df):
        """Perform ABC (value) and XYZ (variability) analysis"""
        
        # ABC Analysis (based on annual value)
        items_df['Annual_Value'] = items_df['Annual_Demand'] * items_df['Unit_Cost']
        items_df['Value_Cumsum'] = items_df.sort_values('Annual_Value', ascending=False)['Annual_Value'].cumsum()
        total_value = items_df['Annual_Value'].sum()
        items_df['Value_Cumsum_Pct'] = items_df['Value_Cumsum'] / total_value
        
        # ABC classification
        items_df['ABC_Class'] = 'C'
        items_df.loc[items_df['Value_Cumsum_Pct'] <= 0.8, 'ABC_Class'] = 'A'
        items_df.loc[items_df['Value_Cumsum_Pct'] <= 0.95, 'ABC_Class'] = 'B'
        
        # XYZ Analysis (based on demand variability)
        items_df['Demand_CV'] = items_df['Demand_Std'] / items_df['Annual_Demand']
        items_df['XYZ_Class'] = 'Z'
        items_df.loc[items_df['Demand_CV'] <= 0.5, 'XYZ_Class'] = 'X'
        items_df.loc[items_df['Demand_CV'] <= 1.0, 'XYZ_Class'] = 'Y'
        
        # Combined classification
        items_df['Combined_Class'] = items_df['ABC_Class'] + items_df['XYZ_Class']
        
        return items_df
    
    def optimize_service_levels(self, items_df, target_fill_rate=0.95):
        """Optimize service levels across item portfolio"""
        
        optimization_results = []
        
        for _, item in items_df.iterrows():
            # Different service level strategies based on ABC/XYZ class
            if item['Combined_Class'] in ['AX', 'AY']:
                target_service_level = 0.98  # High service for high-value, predictable items
            elif item['Combined_Class'] in ['AZ', 'BX']:
                target_service_level = 0.95  # Medium-high service
            elif item['Combined_Class'] in ['BY', 'CX']:
                target_service_level = 0.90  # Medium service
            else:
                target_service_level = 0.85  # Lower service for low-value, unpredictable items
            
            # Calculate optimal inventory parameters
            eoq = self.calculate_basic_eoq(
                item['Annual_Demand'], 
                item['Ordering_Cost'], 
                item['Holding_Cost_Rate'], 
                item['Unit_Cost']
            )
            
            reorder_data = self.calculate_reorder_point(
                item['Daily_Demand_Mean'],
                item['Lead_Time_Days'],
                item['Lead_Time_Std'],
                item['Daily_Demand_Std'],
                target_service_level
            )
            
            optimization_results.append({
                'Item': item['Item'],
                'ABC_XYZ_Class': item['Combined_Class'],
                'Target_Service_Level': target_service_level,
                'EOQ': eoq,
                'Reorder_Point': reorder_data['reorder_point'],
                'Safety_Stock': reorder_data['safety_stock'],
                'Annual_Holding_Cost': (eoq / 2 + reorder_data['safety_stock']) * item['Unit_Cost'] * item['Holding_Cost_Rate'],
                'Annual_Ordering_Cost': (item['Annual_Demand'] / eoq) * item['Ordering_Cost'] if eoq > 0 else 0,
                'Annual_Value': item['Annual_Value']
            })
        
        return pd.DataFrame(optimization_results)
    
    def multi_location_analysis(self, locations_df):
        """Analyze inventory pooling effects across multiple locations"""
        
        pooling_analysis = []
        
        # Group by item across locations
        for item in locations_df['Item'].unique():
            item_locations = locations_df[locations_df['Item'] == item]
            
            if len(item_locations) > 1:
                # Individual location parameters
                total_demand = item_locations['Demand'].sum()
                individual_safety_stock = item_locations['Safety_Stock'].sum()
                
                # Pooled parameters (assuming demand correlation < 1)
                pooled_demand_std = np.sqrt(item_locations['Demand_Variance'].sum())
                pooled_safety_stock = self.calculate_safety_stock(
                    total_demand * item_locations['Lead_Time_Days'].mean() / 365,
                    pooled_demand_std * np.sqrt(item_locations['Lead_Time_Days'].mean() / 365),
                    0.95
                )
                
                # Inventory reduction potential
                inventory_reduction = individual_safety_stock - pooled_safety_stock
                cost_savings = inventory_reduction * item_locations['Unit_Cost'].mean() * item_locations['Holding_Cost_Rate'].mean()
                
                pooling_analysis.append({
                    'Item': item,
                    'Locations': len(item_locations),
                    'Total_Demand': total_demand,
                    'Individual_Safety_Stock': individual_safety_stock,
                    'Pooled_Safety_Stock': pooled_safety_stock,
                    'Inventory_Reduction': inventory_reduction,
                    'Annual_Cost_Savings': cost_savings,
                    'Reduction_Percentage': (inventory_reduction / individual_safety_stock * 100) if individual_safety_stock > 0 else 0
                })
        
        return pd.DataFrame(pooling_analysis)
    
    def monte_carlo_simulation(self, item_params, num_simulations=1000):
        """Monte Carlo simulation for inventory optimization under uncertainty"""
        
        results = []
        
        for _ in range(num_simulations):
            # Sample random parameters
            demand = np.random.normal(item_params['demand_mean'], item_params['demand_std'])
            lead_time = np.random.normal(item_params['lead_time_mean'], item_params['lead_time_std'])
            unit_cost = np.random.normal(item_params['unit_cost_mean'], item_params['unit_cost_std'])
            
            # Ensure positive values
            demand = max(0, demand)
            lead_time = max(1, lead_time)
            unit_cost = max(0.01, unit_cost)
            
            # Calculate EOQ for this scenario
            eoq = self.calculate_basic_eoq(
                demand * 365,  # Annual demand
                item_params['ordering_cost'],
                item_params['holding_cost_rate'],
                unit_cost
            )
            
            # Calculate total cost
            if eoq > 0:
                annual_ordering_cost = (demand * 365 / eoq) * item_params['ordering_cost']
                annual_holding_cost = (eoq / 2) * unit_cost * item_params['holding_cost_rate']
                total_cost = annual_ordering_cost + annual_holding_cost
            else:
                total_cost = float('inf')
            
            results.append({
                'EOQ': eoq,
                'Total_Cost': total_cost,
                'Demand': demand * 365,
                'Lead_Time': lead_time,
                'Unit_Cost': unit_cost
            })
        
        simulation_df = pd.DataFrame(results)
        
        return {
            'mean_eoq': simulation_df['EOQ'].mean(),
            'std_eoq': simulation_df['EOQ'].std(),
            'percentile_5': simulation_df['EOQ'].quantile(0.05),
            'percentile_95': simulation_df['EOQ'].quantile(0.95),
            'mean_cost': simulation_df['Total_Cost'].mean(),
            'std_cost': simulation_df['Total_Cost'].std(),
            'simulation_data': simulation_df
        }

def prepare_item_data(df):
    """Prepare and enrich item data for advanced optimization"""
    
    # Calculate basic statistics for each item
    item_stats = []
    
    for item in df['Item'].unique():
        item_data = df[df['Item'] == item].copy()
        
        if len(item_data) < 3:  # Need minimum data points
            continue
        
        # Calculate demand statistics
        item_data['Month'] = item_data['Creation Date'].dt.to_period('M')
        monthly_demand = item_data.groupby('Month')['Qty Delivered'].sum()
        
        annual_demand = monthly_demand.sum() * (12 / len(monthly_demand)) if len(monthly_demand) > 0 else 0
        demand_mean = monthly_demand.mean()
        demand_std = monthly_demand.std() if len(monthly_demand) > 1 else demand_mean * 0.3
        
        # Calculate lead time statistics (if available)
        if 'PO Receipt Date' in item_data.columns and 'Creation Date' in item_data.columns:
            lead_times = (item_data['PO Receipt Date'] - item_data['Creation Date']).dt.days
            lead_time_mean = lead_times.mean()
            lead_time_std = lead_times.std() if len(lead_times) > 1 else lead_time_mean * 0.2
        else:
            lead_time_mean = 14  # Default 2 weeks
            lead_time_std = 3    # Default variability
        
        # Calculate costs
        unit_cost = item_data['Unit Price'].mean()
        
        item_stats.append({
            'Item': item,
            'Annual_Demand': annual_demand,
            'Monthly_Demand_Mean': demand_mean,
            'Monthly_Demand_Std': demand_std,
            'Daily_Demand_Mean': annual_demand / 365,
            'Daily_Demand_Std': demand_std * np.sqrt(12) / np.sqrt(365),
            'Demand_Std': demand_std,
            'Lead_Time_Days': lead_time_mean,
            'Lead_Time_Std': lead_time_std,
            'Unit_Cost': unit_cost,
            'Ordering_Cost': 100,  # Default ordering cost
            'Holding_Cost_Rate': 0.25,  # Default 25% holding cost
            'Current_Order_Qty': item_data['Qty Delivered'].mean(),
            'Order_Frequency': len(item_data),
            'Price_Volatility': item_data['Unit Price'].std() / item_data['Unit Price'].mean() if item_data['Unit Price'].mean() > 0 else 0
        })
    
    return pd.DataFrame(item_stats)

def display_enhanced_lot_optimization(df):
    """Enhanced LOT Size Optimization with advanced inventory models"""
    
    st.header("📦 Advanced LOT Size & Inventory Optimization")
    st.markdown("Comprehensive inventory optimization with stochastic models, ABC/XYZ analysis, and multi-location strategies.")
    
    # Data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean and prepare data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date'])
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Initialize optimizer
    optimizer = AdvancedInventoryOptimizer()
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 ABC/XYZ Analysis",
        "🎯 Advanced EOQ Models", 
        "🔒 Safety Stock Optimization",
        "🌐 Multi-Location Analysis",
        "🎲 Monte Carlo Simulation",
        "📈 Portfolio Optimization"
    ])
    
    with tab1:
        st.subheader("📊 ABC/XYZ Inventory Classification")
        st.markdown("Classify items by value (ABC) and demand variability (XYZ) for differentiated inventory strategies.")
        
        if st.button("🔍 Perform ABC/XYZ Analysis", type="primary"):
            with st.spinner("Performing advanced inventory classification..."):
                
                # Prepare item data
                item_data = prepare_item_data(df_clean)
                
                if len(item_data) > 0:
                    # Perform ABC/XYZ classification
                    classified_items = optimizer.abc_xyz_classification(item_data)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        a_items = len(classified_items[classified_items['ABC_Class'] == 'A'])
                        st.metric("A Items (High Value)", a_items, f"{a_items/len(classified_items)*100:.1f}%")
                    with col2:
                        x_items = len(classified_items[classified_items['XYZ_Class'] == 'X'])
                        st.metric("X Items (Low Variability)", x_items, f"{x_items/len(classified_items)*100:.1f}%")
                    with col3:
                        ax_items = len(classified_items[classified_items['Combined_Class'] == 'AX'])
                        st.metric("AX Items (High Value, Stable)", ax_items)
                    with col4:
                        total_value = classified_items['Annual_Value'].sum()
                        st.metric("Total Annual Value", f"${total_value:,.0f}")
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # ABC/XYZ Matrix
                        matrix_data = classified_items.groupby(['ABC_Class', 'XYZ_Class']).size().unstack(fill_value=0)
                        
                        fig = px.imshow(
                            matrix_data.values,
                            x=matrix_data.columns,
                            y=matrix_data.index,
                            color_continuous_scale='Blues',
                            title="ABC/XYZ Classification Matrix",
                            text_auto=True
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Value vs Variability scatter
                        fig = px.scatter(
                            classified_items,
                            x='Demand_CV',
                            y='Annual_Value',
                            color='Combined_Class',
                            size='Annual_Demand',
                            hover_data=['Item'],
                            title="Value vs Demand Variability",
                            labels={'Demand_CV': 'Demand Coefficient of Variation', 'Annual_Value': 'Annual Value ($)'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification strategies
                    st.subheader("📋 Inventory Strategy by Classification")
                    
                    strategy_data = {
                        'Class': ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ'],
                        'Description': [
                            'High Value, Stable Demand', 'High Value, Medium Variability', 'High Value, High Variability',
                            'Medium Value, Stable Demand', 'Medium Value, Medium Variability', 'Medium Value, High Variability',
                            'Low Value, Stable Demand', 'Low Value, Medium Variability', 'Low Value, High Variability'
                        ],
                        'Strategy': [
                            'EOQ with high service level (98%)', 'EOQ with safety stock (95%)', 'Continuous review with buffer',
                            'Periodic review (95%)', 'EOQ with moderate safety stock (90%)', 'Min-Max system',
                            'Two-bin system (90%)', 'Simple periodic review (85%)', 'Basic stock policies (80%)'
                        ],
                        'Review_Frequency': [
                            'Weekly', 'Weekly', 'Daily',
                            'Bi-weekly', 'Bi-weekly', 'Monthly',
                            'Monthly', 'Monthly', 'Quarterly'
                        ]
                    }
                    
                    strategy_df = pd.DataFrame(strategy_data)
                    st.dataframe(strategy_df, use_container_width=True)
                    
                    # Detailed classification results
                    st.subheader("📊 Detailed Classification Results")
                    
                    display_columns = ['Item', 'Combined_Class', 'Annual_Value', 'Annual_Demand', 'Demand_CV', 'Value_Cumsum_Pct']
                    
                    st.dataframe(
                        classified_items[display_columns].style.format({
                            'Annual_Value': '${:,.0f}',
                            'Annual_Demand': '{:,.0f}',
                            'Demand_CV': '{:.2f}',
                            'Value_Cumsum_Pct': '{:.1%}'
                        }),
                        use_container_width=True
                    )
                    
                    # Store results for other tabs
                    st.session_state['classified_items'] = classified_items
                
                else:
                    st.warning("Insufficient data for ABC/XYZ analysis.")
    
    with tab2:
        st.subheader("🎯 Advanced EOQ Models")
        st.markdown("Multiple EOQ models including quantity discounts, storage constraints, and backorders.")
        
        # Model selection
        model_type = st.selectbox(
            "Select EOQ Model Type",
            ["Basic EOQ", "EOQ with Quantity Discounts", "EOQ with Storage Constraints", "EOQ with Backorders"]
        )
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            ordering_cost = st.number_input("Ordering Cost ($)", min_value=1.0, value=100.0)
        with col2:
            holding_cost_rate = st.slider("Holding Cost Rate (%)", 5, 50, 25) / 100
        with col3:
            service_level = st.slider("Target Service Level (%)", 80, 99, 95) / 100
        
        # Item selection
        if st.button("🚀 Run Advanced EOQ Analysis", type="primary"):
            with st.spinner("Running advanced EOQ calculations..."):
                
                item_data = prepare_item_data(df_clean)
                eoq_results = []
                
                for _, item in item_data.iterrows():
                    if model_type == "Basic EOQ":
                        eoq = optimizer.calculate_basic_eoq(
                            item['Annual_Demand'], ordering_cost, holding_cost_rate, item['Unit_Cost']
                        )
                        model_result = {
                            'EOQ': eoq,
                            'Model_Type': 'Basic',
                            'Special_Info': 'Standard EOQ formula'
                        }
                    
                    elif model_type == "EOQ with Quantity Discounts":
                        # Generate sample discount structure
                        discount_structure = [
                            (1, 0.00, item['Unit_Cost']),      # No discount for small quantities
                            (100, 0.03, item['Unit_Cost']),    # 3% discount for 100+ units
                            (500, 0.05, item['Unit_Cost']),    # 5% discount for 500+ units
                            (1000, 0.08, item['Unit_Cost'])    # 8% discount for 1000+ units
                        ]
                        
                        result = optimizer.calculate_eoq_with_quantity_discounts(
                            item['Annual_Demand'], ordering_cost, holding_cost_rate, discount_structure
                        )
                        
                        if result:
                            model_result = {
                                'EOQ': result['order_quantity'],
                                'Model_Type': 'Quantity Discount',
                                'Special_Info': f"Discount: {result['discount_rate']:.1%}, Break: {result['break_quantity']}"
                            }
                        else:
                            model_result = {'EOQ': 0, 'Model_Type': 'Quantity Discount', 'Special_Info': 'No viable solution'}
                    
                    elif model_type == "EOQ with Storage Constraints":
                        # Assume storage constraint
                        max_storage = item['Annual_Demand'] * 0.3  # 30% of annual demand
                        basic_eoq = optimizer.calculate_basic_eoq(
                            item['Annual_Demand'], ordering_cost, holding_cost_rate, item['Unit_Cost']
                        )
                        constrained_eoq = min(basic_eoq, max_storage)
                        
                        model_result = {
                            'EOQ': constrained_eoq,
                            'Model_Type': 'Storage Constrained',
                            'Special_Info': f"Constraint: {max_storage:.0f} units, Unconstrained: {basic_eoq:.0f}"
                        }
                    
                    else:  # EOQ with Backorders
                        # Simplified backorder model
                        shortage_cost = item['Unit_Cost'] * 0.1  # 10% of unit cost per unit short per year
                        holding_cost = item['Unit_Cost'] * holding_cost_rate
                        
                        if shortage_cost > 0:
                            # EOQ with planned backorders
                            basic_eoq = optimizer.calculate_basic_eoq(
                                item['Annual_Demand'], ordering_cost, holding_cost_rate, item['Unit_Cost']
                            )
                            backorder_factor = np.sqrt((holding_cost + shortage_cost) / shortage_cost)
                            eoq_backorder = basic_eoq * backorder_factor
                            
                            model_result = {
                                'EOQ': eoq_backorder,
                                'Model_Type': 'With Backorders',
                                'Special_Info': f"Backorder factor: {backorder_factor:.2f}"
                            }
                        else:
                            model_result = {'EOQ': 0, 'Model_Type': 'With Backorders', 'Special_Info': 'Invalid parameters'}
                    
                    # Calculate costs and metrics
                    if model_result['EOQ'] > 0:
                        annual_holding_cost = (model_result['EOQ'] / 2) * item['Unit_Cost'] * holding_cost_rate
                        annual_ordering_cost = (item['Annual_Demand'] / model_result['EOQ']) * ordering_cost
                        total_cost = annual_holding_cost + annual_ordering_cost
                        current_cost = (item['Current_Order_Qty'] / 2) * item['Unit_Cost'] * holding_cost_rate + \
                                      (item['Annual_Demand'] / item['Current_Order_Qty']) * ordering_cost if item['Current_Order_Qty'] > 0 else 0
                        savings = current_cost - total_cost if current_cost > 0 else 0
                    else:
                        annual_holding_cost = annual_ordering_cost = total_cost = savings = 0
                    
                    eoq_results.append({
                        'Item': item['Item'],
                        'Current_Order_Qty': item['Current_Order_Qty'],
                        'Optimized_EOQ': model_result['EOQ'],
                        'Model_Type': model_result['Model_Type'],
                        'Annual_Demand': item['Annual_Demand'],
                        'Annual_Holding_Cost': annual_holding_cost,
                        'Annual_Ordering_Cost': annual_ordering_cost,
                        'Total_Annual_Cost': total_cost,
                        'Annual_Savings': savings,
                        'Special_Info': model_result['Special_Info']
                    })
                
                eoq_df = pd.DataFrame(eoq_results)
                eoq_df = eoq_df.sort_values('Annual_Savings', ascending=False)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_savings = eoq_df['Annual_Savings'].sum()
                    st.metric("Total Annual Savings", f"${total_savings:,.0f}")
                with col2:
                    avg_eoq = eoq_df['Optimized_EOQ'].mean()
                    st.metric("Average EOQ", f"{avg_eoq:.0f}")
                with col3:
                    avg_reduction = ((eoq_df['Current_Order_Qty'] - eoq_df['Optimized_EOQ']) / eoq_df['Current_Order_Qty']).mean() * 100
                    st.metric("Avg Order Size Change", f"{avg_reduction:.1f}%")
                with col4:
                    items_optimized = len(eoq_df[eoq_df['Annual_Savings'] > 0])
                    st.metric("Items with Savings", items_optimized)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Current vs Optimized order quantities
                    fig = px.scatter(
                        eoq_df,
                        x='Current_Order_Qty',
                        y='Optimized_EOQ',
                        size='Annual_Savings',
                        color='Model_Type',
                        hover_data=['Item', 'Annual_Demand'],
                        title="Current vs Optimized Order Quantities"
                    )
                    # Add diagonal line
                    max_qty = max(eoq_df['Current_Order_Qty'].max(), eoq_df['Optimized_EOQ'].max())
                    fig.add_shape(type='line', x0=0, y0=0, x1=max_qty, y1=max_qty, 
                                line=dict(dash='dash', color='red'))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Savings by item
                    top_savings = eoq_df.nlargest(10, 'Annual_Savings')
                    
                    fig = px.bar(
                        top_savings,
                        x='Annual_Savings',
                        y='Item',
                        color='Model_Type',
                        title="Top 10 Items by Annual Savings",
                        orientation='h'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                st.subheader("📊 Advanced EOQ Analysis Results")
                
                display_columns = ['Item', 'Current_Order_Qty', 'Optimized_EOQ', 'Model_Type', 
                                 'Total_Annual_Cost', 'Annual_Savings', 'Special_Info']
                
                st.dataframe(
                    eoq_df[display_columns].style.format({
                        'Current_Order_Qty': '{:.0f}',
                        'Optimized_EOQ': '{:.0f}',
                        'Total_Annual_Cost': '${:,.0f}',
                        'Annual_Savings': '${:,.0f}'
                    }),
                    use_container_width=True
                )
    
    with tab3:
        st.subheader("🔒 Safety Stock & Reorder Point Optimization")
        st.markdown("Optimize safety stock levels and reorder points based on service level targets and demand/lead time variability.")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            target_service_level = st.slider("Target Service Level (%)", 85, 99, 95) / 100
        with col2:
            lead_time_variability = st.slider("Lead Time Variability (%)", 10, 50, 20) / 100
        
        if st.button("🔒 Optimize Safety Stock", type="primary"):
            with st.spinner("Optimizing safety stock and reorder points..."):
                
                item_data = prepare_item_data(df_clean)
                safety_stock_results = []
                
                for _, item in item_data.iterrows():
                    # Calculate reorder point with variability
                    reorder_data = optimizer.calculate_reorder_point(
                        item['Daily_Demand_Mean'],
                        item['Lead_Time_Days'],
                        item['Lead_Time_Days'] * lead_time_variability,  # Lead time std
                        item['Daily_Demand_Std'],
                        target_service_level
                    )
                    
                    # Calculate costs
                    safety_stock_cost = reorder_data['safety_stock'] * item['Unit_Cost'] * item['Holding_Cost_Rate']
                    
                    # Calculate stockout risk
                    stockout_probability = 1 - target_service_level
                    expected_stockout_cost = stockout_probability * item['Daily_Demand_Mean'] * item['Unit_Cost'] * 0.1  # 10% penalty
                    
                    safety_stock_results.append({
                        'Item': item['Item'],
                        'Daily_Demand_Mean': item['Daily_Demand_Mean'],
                        'Daily_Demand_Std': item['Daily_Demand_Std'],
                        'Lead_Time_Days': item['Lead_Time_Days'],
                        'Reorder_Point': reorder_data['reorder_point'],
                        'Safety_Stock': reorder_data['safety_stock'],
                        'Mean_Lead_Time_Demand': reorder_data['mean_lead_time_demand'],
                        'Safety_Stock_Cost': safety_stock_cost,
                        'Stockout_Probability': stockout_probability,
                        'Expected_Stockout_Cost': expected_stockout_cost,
                        'Service_Level': target_service_level
                    })
                
                safety_df = pd.DataFrame(safety_stock_results)
                safety_df = safety_df.sort_values('Safety_Stock_Cost', ascending=False)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_safety_stock = safety_df['Safety_Stock'].sum()
                    st.metric("Total Safety Stock", f"{total_safety_stock:,.0f} units")
                with col2:
                    total_safety_cost = safety_df['Safety_Stock_Cost'].sum()
                    st.metric("Total Safety Stock Cost", f"${total_safety_cost:,.0f}")
                with col3:
                    avg_service_level = safety_df['Service_Level'].mean()
                    st.metric("Target Service Level", f"{avg_service_level:.1%}")
                with col4:
                    total_stockout_risk = safety_df['Expected_Stockout_Cost'].sum()
                    st.metric("Expected Stockout Cost", f"${total_stockout_risk:,.0f}")
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Safety stock vs demand variability
                    fig = px.scatter(
                        safety_df,
                        x='Daily_Demand_Std',
                        y='Safety_Stock',
                        size='Safety_Stock_Cost',
                        color='Lead_Time_Days',
                        hover_data=['Item'],
                        title="Safety Stock vs Demand Variability"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Service level trade-off analysis
                    service_levels = np.arange(0.85, 0.995, 0.01)
                    sample_item = safety_df.iloc[0]
                    
                    trade_off_data = []
                    for sl in service_levels:
                        ss = optimizer.calculate_safety_stock(
                            sample_item['Mean_Lead_Time_Demand'],
                            sample_item['Daily_Demand_Std'] * np.sqrt(sample_item['Lead_Time_Days']),
                            sl
                        )
                        cost = ss * item_data.iloc[0]['Unit_Cost'] * item_data.iloc[0]['Holding_Cost_Rate']
                        trade_off_data.append({'Service_Level': sl, 'Safety_Stock_Cost': cost})
                    
                    trade_off_df = pd.DataFrame(trade_off_data)
                    
                    fig = px.line(
                        trade_off_df,
                        x='Service_Level',
                        y='Safety_Stock_Cost',
                        title="Service Level vs Safety Stock Cost Trade-off"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                st.subheader("📊 Safety Stock Optimization Results")
                
                display_columns = ['Item', 'Daily_Demand_Mean', 'Lead_Time_Days', 'Reorder_Point', 
                                 'Safety_Stock', 'Safety_Stock_Cost', 'Stockout_Probability']
                
                st.dataframe(
                    safety_df[display_columns].style.format({
                        'Daily_Demand_Mean': '{:.1f}',
                        'Reorder_Point': '{:.0f}',
                        'Safety_Stock': '{:.0f}',
                        'Safety_Stock_Cost': '${:,.0f}',
                        'Stockout_Probability': '{:.1%}'
                    }),
                    use_container_width=True
                )
    
    with tab4:
        st.subheader("🌐 Multi-Location Inventory Analysis")
        st.markdown("Analyze inventory pooling effects and optimize across multiple warehouses/locations.")
        
        # Check if location data is available
        if 'W/H' not in df_clean.columns and 'DEP' not in df_clean.columns:
            st.warning("Location data (W/H or DEP columns) not available for multi-location analysis.")
        else:
            if st.button("🌐 Analyze Multi-Location Opportunities", type="primary"):
                with st.spinner("Analyzing multi-location inventory optimization..."):
                    
                    # Prepare location-level data
                    location_data = []
                    location_col = 'W/H' if 'W/H' in df_clean.columns else 'DEP'
                    
                    for item in df_clean['Item'].unique():
                        item_locations = df_clean[df_clean['Item'] == item]
                        
                        for location in item_locations[location_col].unique():
                            location_item_data = item_locations[item_locations[location_col] == location]
                            
                            if len(location_item_data) > 0:
                                monthly_demand = location_item_data.groupby(
                                    location_item_data['Creation Date'].dt.to_period('M')
                                )['Qty Delivered'].sum()
                                
                                demand_mean = monthly_demand.mean() if len(monthly_demand) > 0 else 0
                                demand_std = monthly_demand.std() if len(monthly_demand) > 1 else demand_mean * 0.3
                                
                                # Calculate safety stock for this location
                                safety_stock = optimizer.calculate_safety_stock(
                                    demand_mean, demand_std, 0.95
                                )
                                
                                location_data.append({
                                    'Item': item,
                                    'Location': location,
                                    'Demand': demand_mean * 12,  # Annual demand
                                    'Demand_Variance': (demand_std * np.sqrt(12))**2,
                                    'Safety_Stock': safety_stock,
                                    'Unit_Cost': location_item_data['Unit Price'].mean(),
                                    'Holding_Cost_Rate': 0.25,
                                    'Lead_Time_Days': 14
                                })
                    
                    if location_data:
                        locations_df = pd.DataFrame(location_data)
                        
                        # Perform pooling analysis
                        pooling_results = optimizer.multi_location_analysis(locations_df)
                        
                        if len(pooling_results) > 0:
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                total_inventory_reduction = pooling_results['Inventory_Reduction'].sum()
                                st.metric("Total Inventory Reduction", f"{total_inventory_reduction:,.0f} units")
                            with col2:
                                total_cost_savings = pooling_results['Annual_Cost_Savings'].sum()
                                st.metric("Annual Cost Savings", f"${total_cost_savings:,.0f}")
                            with col3:
                                avg_reduction_pct = pooling_results['Reduction_Percentage'].mean()
                                st.metric("Avg Reduction %", f"{avg_reduction_pct:.1f}%")
                            with col4:
                                items_with_pooling = len(pooling_results)
                                st.metric("Items for Pooling", items_with_pooling)
                            
                            # Visualization
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Inventory reduction by item
                                fig = px.bar(
                                    pooling_results.nlargest(10, 'Annual_Cost_Savings'),
                                    x='Annual_Cost_Savings',
                                    y='Item',
                                    color='Locations',
                                    title="Top 10 Items by Pooling Savings",
                                    orientation='h'
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Reduction percentage vs locations
                                fig = px.scatter(
                                    pooling_results,
                                    x='Locations',
                                    y='Reduction_Percentage',
                                    size='Annual_Cost_Savings',
                                    color='Total_Demand',
                                    hover_data=['Item'],
                                    title="Inventory Reduction vs Number of Locations"
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Pooling strategy recommendations
                            st.subheader("📋 Inventory Pooling Recommendations")
                            
                            # Classify pooling opportunities
                            pooling_results['Priority'] = 'Low'
                            pooling_results.loc[
                                (pooling_results['Annual_Cost_Savings'] > pooling_results['Annual_Cost_Savings'].quantile(0.7)) &
                                (pooling_results['Reduction_Percentage'] > 15), 'Priority'
                            ] = 'High'
                            pooling_results.loc[
                                (pooling_results['Annual_Cost_Savings'] > pooling_results['Annual_Cost_Savings'].quantile(0.4)) &
                                (pooling_results['Reduction_Percentage'] > 10) &
                                (pooling_results['Priority'] == 'Low'), 'Priority'
                            ] = 'Medium'
                            
                            # Display results
                            display_columns = ['Item', 'Locations', 'Individual_Safety_Stock', 'Pooled_Safety_Stock',
                                             'Inventory_Reduction', 'Annual_Cost_Savings', 'Reduction_Percentage', 'Priority']
                            
                            st.dataframe(
                                pooling_results[display_columns].style.format({
                                    'Individual_Safety_Stock': '{:.0f}',
                                    'Pooled_Safety_Stock': '{:.0f}',
                                    'Inventory_Reduction': '{:.0f}',
                                    'Annual_Cost_Savings': '${:,.0f}',
                                    'Reduction_Percentage': '{:.1f}%'
                                }),
                                use_container_width=True
                            )
                        else:
                            st.info("No significant inventory pooling opportunities found.")
                    else:
                        st.warning("Insufficient location-specific data for analysis.")
    
    with tab5:
        st.subheader("🎲 Monte Carlo Simulation")
        st.markdown("Use Monte Carlo simulation to handle uncertainty in demand, lead times, and costs.")
        
        # Item selection for simulation
        item_data = prepare_item_data(df_clean)
        
        if len(item_data) > 0:
            selected_item = st.selectbox(
                "Select Item for Monte Carlo Analysis",
                options=item_data['Item'].tolist()
            )
            
            # Simulation parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                num_simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000)
            with col2:
                demand_uncertainty = st.slider("Demand Uncertainty (%)", 10, 50, 25) / 100
            with col3:
                cost_uncertainty = st.slider("Cost Uncertainty (%)", 5, 30, 15) / 100
            
            if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
                with st.spinner("Running Monte Carlo simulation..."):
                    
                    item_info = item_data[item_data['Item'] == selected_item].iloc[0]
                    
                    # Prepare simulation parameters
                    sim_params = {
                        'demand_mean': item_info['Annual_Demand'] / 365,  # Daily demand
                        'demand_std': item_info['Annual_Demand'] / 365 * demand_uncertainty,
                        'lead_time_mean': item_info['Lead_Time_Days'],
                        'lead_time_std': item_info['Lead_Time_Days'] * 0.2,
                        'unit_cost_mean': item_info['Unit_Cost'],
                        'unit_cost_std': item_info['Unit_Cost'] * cost_uncertainty,
                        'ordering_cost': item_info['Ordering_Cost'],
                        'holding_cost_rate': item_info['Holding_Cost_Rate']
                    }
                    
                    # Run simulation
                    simulation_results = optimizer.monte_carlo_simulation(sim_params, num_simulations)
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean EOQ", f"{simulation_results['mean_eoq']:.0f}")
                    with col2:
                        st.metric("EOQ Std Dev", f"{simulation_results['std_eoq']:.0f}")
                    with col3:
                        st.metric("5th Percentile", f"{simulation_results['percentile_5']:.0f}")
                    with col4:
                        st.metric("95th Percentile", f"{simulation_results['percentile_95']:.0f}")
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # EOQ distribution
                        fig = px.histogram(
                            simulation_results['simulation_data'],
                            x='EOQ',
                            nbins=50,
                            title="EOQ Distribution from Monte Carlo Simulation"
                        )
                        fig.add_vline(x=simulation_results['mean_eoq'], line_dash="dash", 
                                     line_color="red", annotation_text="Mean")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cost distribution
                        fig = px.histogram(
                            simulation_results['simulation_data'],
                            x='Total_Cost',
                            nbins=50,
                            title="Total Cost Distribution"
                        )
                        fig.add_vline(x=simulation_results['mean_cost'], line_dash="dash", 
                                     line_color="red", annotation_text="Mean")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk analysis
                    st.subheader("📊 Risk Analysis")
                    
                    sim_data = simulation_results['simulation_data']
                    
                    # Calculate Value at Risk (VaR)
                    cost_var_95 = sim_data['Total_Cost'].quantile(0.95)
                    cost_var_99 = sim_data['Total_Cost'].quantile(0.99)
                    
                    risk_metrics = {
                        'Metric': ['Value at Risk (95%)', 'Value at Risk (99%)', 'Expected Cost', 'Cost Volatility'],
                        'Value': [
                            f"${cost_var_95:,.0f}",
                            f"${cost_var_99:,.0f}",
                            f"${simulation_results['mean_cost']:,.0f}",
                            f"${simulation_results['std_cost']:,.0f}"
                        ]
                    }
                    
                    risk_df = pd.DataFrame(risk_metrics)
                    st.dataframe(risk_df, use_container_width=True)
                    
                    # Sensitivity analysis
                    st.subheader("📈 Sensitivity Analysis")
                    
                    # Correlation with input parameters
                    correlations = {
                        'Parameter': ['Demand', 'Lead Time', 'Unit Cost'],
                        'Correlation with EOQ': [
                            sim_data[['Demand', 'EOQ']].corr().iloc[0, 1],
                            sim_data[['Lead_Time', 'EOQ']].corr().iloc[0, 1],
                            sim_data[['Unit_Cost', 'EOQ']].corr().iloc[0, 1]
                        ]
                    }
                    
                    corr_df = pd.DataFrame(correlations)
                    
                    fig = px.bar(
                        corr_df,
                        x='Parameter',
                        y='Correlation with EOQ',
                        title="Parameter Sensitivity Analysis"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No item data available for Monte Carlo simulation.")
    
    with tab6:
        st.subheader("📈 Portfolio Optimization")
        st.markdown("Optimize inventory across the entire item portfolio with budget and service level constraints.")
        
        if 'classified_items' in st.session_state:
            classified_items = st.session_state['classified_items']
            
            # Portfolio constraints
            col1, col2, col3 = st.columns(3)
            with col1:
                inventory_budget = st.number_input("Inventory Budget ($)", min_value=0, value=1000000, step=50000)
            with col2:
                min_service_level = st.slider("Minimum Service Level (%)", 80, 95, 85) / 100
            with col3:
                max_service_level = st.slider("Maximum Service Level (%)", 95, 99, 98) / 100
            
            if st.button("🎯 Optimize Portfolio", type="primary"):
                with st.spinner("Optimizing inventory portfolio..."):
                    
                    # Perform service level optimization
                    portfolio_results = optimizer.optimize_service_levels(classified_items)
                    
                    # Calculate total portfolio metrics
                    total_inventory_value = portfolio_results['Annual_Holding_Cost'].sum() / 0.25  # Assuming 25% holding cost
                    total_holding_cost = portfolio_results['Annual_Holding_Cost'].sum()
                    total_ordering_cost = portfolio_results['Annual_Ordering_Cost'].sum()
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Inventory Value", f"${total_inventory_value:,.0f}")
                    with col2:
                        st.metric("Total Holding Cost", f"${total_holding_cost:,.0f}")
                    with col3:
                        st.metric("Total Ordering Cost", f"${total_ordering_cost:,.0f}")
                    with col4:
                        weighted_service_level = (portfolio_results['Target_Service_Level'] * portfolio_results['Annual_Value']).sum() / portfolio_results['Annual_Value'].sum()
                        st.metric("Weighted Service Level", f"{weighted_service_level:.1%}")
                    
                    # Portfolio allocation
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Service level by class
                        class_summary = portfolio_results.groupby('ABC_XYZ_Class').agg({
                            'Target_Service_Level': 'mean',
                            'Annual_Holding_Cost': 'sum',
                            'Annual_Value': 'sum'
                        }).reset_index()
                        
                        fig = px.bar(
                            class_summary,
                            x='ABC_XYZ_Class',
                            y='Target_Service_Level',
                            color='Annual_Value',
                            title="Service Levels by Item Class"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Inventory allocation
                        fig = px.pie(
                            class_summary,
                            values='Annual_Holding_Cost',
                            names='ABC_XYZ_Class',
                            title="Inventory Investment by Class"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Portfolio efficiency analysis
                    st.subheader("📊 Portfolio Efficiency Analysis")
                    
                    # Calculate efficiency metrics
                    portfolio_results['Inventory_Turns'] = portfolio_results['Annual_Value'] / (portfolio_results['EOQ'] * classified_items['Unit_Cost'])
                    portfolio_results['Service_Cost_Ratio'] = portfolio_results['Target_Service_Level'] / portfolio_results['Annual_Holding_Cost']
                    
                    # Efficiency frontier
                    fig = px.scatter(
                        portfolio_results,
                        x='Annual_Holding_Cost',
                        y='Target_Service_Level',
                        size='Annual_Value',
                        color='ABC_XYZ_Class',
                        hover_data=['Item'],
                        title="Inventory Efficiency Frontier: Service Level vs Cost"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed portfolio results
                    st.subheader("📋 Portfolio Optimization Results")
                    
                    display_columns = ['Item', 'ABC_XYZ_Class', 'Target_Service_Level', 'EOQ', 
                                     'Safety_Stock', 'Annual_Holding_Cost', 'Inventory_Turns']
                    
                    st.dataframe(
                        portfolio_results[display_columns].style.format({
                            'Target_Service_Level': '{:.1%}',
                            'EOQ': '{:.0f}',
                            'Safety_Stock': '{:.0f}',
                            'Annual_Holding_Cost': '${:,.0f}',
                            'Inventory_Turns': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Optimization recommendations
                    st.subheader("💡 Portfolio Optimization Recommendations")
                    
                    recommendations = []
                    
                    # Budget utilization
                    if total_inventory_value > inventory_budget * 1.1:
                        recommendations.append("🔴 **Over Budget**: Consider reducing service levels for C-class items")
                    elif total_inventory_value < inventory_budget * 0.9:
                        recommendations.append("🟢 **Under Budget**: Opportunity to increase service levels for A-class items")
                    
                    # Service level optimization
                    low_service_high_value = portfolio_results[
                        (portfolio_results['Target_Service_Level'] < 0.90) & 
                        (portfolio_results['Annual_Value'] > portfolio_results['Annual_Value'].quantile(0.8))
                    ]
                    if len(low_service_high_value) > 0:
                        recommendations.append(f"⚠️ **Service Gap**: {len(low_service_high_value)} high-value items have low service levels")
                    
                    # Inventory turns
                    low_turns = portfolio_results[portfolio_results['Inventory_Turns'] < 2]
                    if len(low_turns) > 0:
                        recommendations.append(f"📦 **Slow Movers**: {len(low_turns)} items have low inventory turns (<2x)")
                    
                    for rec in recommendations:
                        st.write(rec)
                    
                    if not recommendations:
                        st.success("✅ Portfolio appears well optimized within constraints!")
        
        else:
            st.info("Please run ABC/XYZ Analysis first to enable portfolio optimization.")

# Main function to integrate with the app
def display(df):
    display_enhanced_lot_optimization(df)
