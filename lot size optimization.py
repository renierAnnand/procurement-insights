import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def calculate_eoq(annual_demand, ordering_cost, holding_cost_rate, unit_cost):
    """Calculate Economic Order Quantity"""
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_rate <= 0 or unit_cost <= 0:
        return 0
    
    holding_cost_per_unit = holding_cost_rate * unit_cost
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
    return eoq

def calculate_total_cost(annual_demand, order_quantity, ordering_cost, holding_cost_rate, unit_cost):
    """Calculate total inventory cost"""
    if order_quantity <= 0:
        return float('inf')
    
    # Purchase cost
    purchase_cost = annual_demand * unit_cost
    
    # Ordering cost
    ordering_cost_total = (annual_demand / order_quantity) * ordering_cost
    
    # Holding cost
    holding_cost_per_unit = holding_cost_rate * unit_cost
    holding_cost_total = (order_quantity / 2) * holding_cost_per_unit
    
    total_cost = purchase_cost + ordering_cost_total + holding_cost_total
    
    return {
        'total_cost': total_cost,
        'purchase_cost': purchase_cost,
        'ordering_cost': ordering_cost_total,
        'holding_cost': holding_cost_total
    }

def analyze_quantity_discounts(annual_demand, ordering_cost, holding_cost_rate, unit_cost, discount_tiers):
    """Analyze quantity discount scenarios"""
    results = []
    
    for tier in discount_tiers:
        min_qty = tier['min_quantity']
        discounted_price = tier['unit_price']
        
        # Calculate EOQ for this price tier
        eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost_rate, discounted_price)
        
        # Determine actual order quantity (must meet minimum)
        order_qty = max(eoq, min_qty)
        
        # Calculate total cost
        cost_breakdown = calculate_total_cost(annual_demand, order_qty, ordering_cost, holding_cost_rate, discounted_price)
        
        results.append({
            'tier': tier['name'],
            'min_quantity': min_qty,
            'unit_price': discounted_price,
            'eoq': eoq,
            'recommended_qty': order_qty,
            'total_cost': cost_breakdown['total_cost'],
            'purchase_cost': cost_breakdown['purchase_cost'],
            'ordering_cost': cost_breakdown['ordering_cost'],
            'holding_cost': cost_breakdown['holding_cost'],
            'discount_percent': ((unit_cost - discounted_price) / unit_cost) * 100
        })
    
    return results

def calculate_reorder_metrics(eoq, annual_demand, lead_time_days=30):
    """Calculate reorder point and cycle metrics"""
    # Daily demand
    daily_demand = annual_demand / 365
    
    # Reorder point (simplified - no safety stock)
    reorder_point = daily_demand * lead_time_days
    
    # Order frequency
    orders_per_year = annual_demand / eoq if eoq > 0 else 0
    days_between_orders = 365 / orders_per_year if orders_per_year > 0 else 0
    
    # Average inventory
    average_inventory = eoq / 2
    
    # Inventory turnover
    inventory_turnover = annual_demand / average_inventory if average_inventory > 0 else 0
    
    return {
        'reorder_point': reorder_point,
        'orders_per_year': orders_per_year,
        'days_between_orders': days_between_orders,
        'average_inventory': average_inventory,
        'inventory_turnover': inventory_turnover,
        'daily_demand': daily_demand
    }

def display(df):
    st.header("📦 LOT Size Optimization")
    st.markdown("Optimize order quantities to minimize total inventory costs using Economic Order Quantity (EOQ) and advanced analytics.")
    
    # Data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
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
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 EOQ Analysis", "💰 Cost Optimization", "🎯 Quantity Discounts", "📈 Bulk Analysis", "⚙️ Settings"])
    
    with tab1:
        st.subheader("📊 Economic Order Quantity Analysis")
        
        # Item selection
        item_options = sorted(df_clean["Item"].dropna().unique())
        selected_item = st.selectbox("Select Item for EOQ Analysis", item_options, key="eoq_item")
        
        if selected_item:
            item_df = df_clean[df_clean["Item"] == selected_item].copy()
            
            # Calculate annual demand
            data_period_days = (item_df['Creation Date'].max() - item_df['Creation Date'].min()).days
            total_demand = item_df['Qty Delivered'].sum()
            annual_demand = (total_demand / data_period_days * 365) if data_period_days > 0 else total_demand
            
            # Item information
            item_desc = item_df['Item Description'].iloc[0] if 'Item Description' in item_df.columns else "N/A"
            avg_unit_price = item_df['Unit Price'].mean()
            current_avg_order_size = item_df['Qty Delivered'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annual Demand", f"{annual_demand:,.1f}")
            with col2:
                st.metric("Average Unit Price", f"{avg_unit_price:.2f}")
            with col3:
                st.metric("Current Avg Order", f"{current_avg_order_size:.1f}")
            
            st.info(f"**Item Description:** {item_desc}")
            
            # User inputs for EOQ calculation
            st.subheader("⚙️ EOQ Parameters")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                ordering_cost = st.number_input(
                    "Ordering Cost per Order",
                    min_value=0.0,
                    value=100.0,
                    step=10.0,
                    help="Cost to place one order (processing, shipping, handling)"
                )
            with col2:
                holding_cost_rate = st.number_input(
                    "Holding Cost Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    help="Annual holding cost as percentage of item value"
                ) / 100
            with col3:
                lead_time_days = st.number_input(
                    "Lead Time (days)",
                    min_value=1,
                    max_value=365,
                    value=30,
                    help="Time from order placement to receipt"
                )
            
            # Calculate EOQ
            eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost_rate, avg_unit_price)
            
            if eoq > 0:
                # Calculate costs for current vs optimal
                current_cost = calculate_total_cost(annual_demand, current_avg_order_size, ordering_cost, holding_cost_rate, avg_unit_price)
                optimal_cost = calculate_total_cost(annual_demand, eoq, ordering_cost, holding_cost_rate, avg_unit_price)
                
                cost_savings = current_cost['total_cost'] - optimal_cost['total_cost']
                savings_percent = (cost_savings / current_cost['total_cost']) * 100
                
                # Display EOQ results
                st.subheader("🎯 EOQ Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Optimal Order Qty", f"{eoq:.0f}")
                with col2:
                    st.metric("Current Total Cost", f"{current_cost['total_cost']:,.0f}")
                with col3:
                    st.metric("Optimal Total Cost", f"{optimal_cost['total_cost']:,.0f}")
                with col4:
                    st.metric("Annual Savings", f"{cost_savings:,.0f}", delta=f"{savings_percent:.1f}%")
                
                # Cost breakdown comparison
                st.subheader("💰 Cost Breakdown Comparison")
                
                comparison_data = pd.DataFrame({
                    'Cost Component': ['Purchase Cost', 'Ordering Cost', 'Holding Cost', 'Total Cost'],
                    'Current Approach': [
                        current_cost['purchase_cost'],
                        current_cost['ordering_cost'],
                        current_cost['holding_cost'],
                        current_cost['total_cost']
                    ],
                    'EOQ Approach': [
                        optimal_cost['purchase_cost'],
                        optimal_cost['ordering_cost'],
                        optimal_cost['holding_cost'],
                        optimal_cost['total_cost']
                    ]
                })
                
                comparison_data['Savings'] = comparison_data['Current Approach'] - comparison_data['EOQ Approach']
                
                st.dataframe(
                    comparison_data.style.format({
                        'Current Approach': '{:,.0f}',
                        'EOQ Approach': '{:,.0f}',
                        'Savings': '{:,.0f}'
                    }),
                    use_container_width=True
                )
                
                # Cost visualization
                fig = go.Figure()
                
                x_categories = ['Purchase Cost', 'Ordering Cost', 'Holding Cost']
                
                fig.add_trace(go.Bar(
                    name='Current Approach',
                    x=x_categories,
                    y=[current_cost['purchase_cost'], current_cost['ordering_cost'], current_cost['holding_cost']],
                    marker_color='lightcoral'
                ))
                
                fig.add_trace(go.Bar(
                    name='EOQ Approach',
                    x=x_categories,
                    y=[optimal_cost['purchase_cost'], optimal_cost['ordering_cost'], optimal_cost['holding_cost']],
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title=f"Cost Comparison - Item {selected_item}",
                    xaxis_title="Cost Component",
                    yaxis_title="Annual Cost",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Reorder metrics
                metrics = calculate_reorder_metrics(eoq, annual_demand, lead_time_days)
                
                st.subheader("📊 Inventory Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Reorder Point", f"{metrics['reorder_point']:.0f}")
                with col2:
                    st.metric("Orders per Year", f"{metrics['orders_per_year']:.1f}")
                with col3:
                    st.metric("Days Between Orders", f"{metrics['days_between_orders']:.0f}")
                with col4:
                    st.metric("Inventory Turnover", f"{metrics['inventory_turnover']:.1f}x")
            else:
                st.error("Unable to calculate EOQ with current parameters.")
    
    with tab2:
        st.subheader("💰 Total Cost Optimization")
        
        # Item selection for cost analysis
        cost_item = st.selectbox("Select Item for Cost Analysis", item_options, key="cost_item")
        
        if cost_item:
            item_df = df_clean[df_clean["Item"] == cost_item].copy()
            
            # Calculate parameters
            data_period_days = (item_df['Creation Date'].max() - item_df['Creation Date'].min()).days
            total_demand = item_df['Qty Delivered'].sum()
            annual_demand = (total_demand / data_period_days * 365) if data_period_days > 0 else total_demand
            avg_unit_price = item_df['Unit Price'].mean()
            
            # Cost parameters
            col1, col2 = st.columns(2)
            with col1:
                ordering_cost = st.number_input("Ordering Cost", min_value=0.0, value=100.0, key="cost_ordering")
            with col2:
                holding_rate = st.number_input("Holding Rate (%)", min_value=0.0, value=20.0, key="cost_holding") / 100
            
            # Generate cost curve
            quantity_range = np.linspace(1, annual_demand, 100)
            cost_data = []
            
            for qty in quantity_range:
                cost_breakdown = calculate_total_cost(annual_demand, qty, ordering_cost, holding_rate, avg_unit_price)
                cost_data.append({
                    'quantity': qty,
                    'total_cost': cost_breakdown['total_cost'],
                    'ordering_cost': cost_breakdown['ordering_cost'],
                    'holding_cost': cost_breakdown['holding_cost']
                })
            
            cost_df = pd.DataFrame(cost_data)
            
            # Find optimal point
            optimal_idx = cost_df['total_cost'].idxmin()
            optimal_qty = cost_df.loc[optimal_idx, 'quantity']
            optimal_cost = cost_df.loc[optimal_idx, 'total_cost']
            
            # Create cost curve visualization
            fig = go.Figure()
            
            # Total cost curve
            fig.add_trace(go.Scatter(
                x=cost_df['quantity'],
                y=cost_df['total_cost'],
                mode='lines',
                name='Total Cost',
                line=dict(color='red', width=3)
            ))
            
            # Ordering cost curve
            fig.add_trace(go.Scatter(
                x=cost_df['quantity'],
                y=cost_df['ordering_cost'],
                mode='lines',
                name='Ordering Cost',
                line=dict(color='blue', dash='dash')
            ))
            
            # Holding cost curve
            fig.add_trace(go.Scatter(
                x=cost_df['quantity'],
                y=cost_df['holding_cost'],
                mode='lines',
                name='Holding Cost',
                line=dict(color='green', dash='dot')
            ))
            
            # Optimal point
            fig.add_trace(go.Scatter(
                x=[optimal_qty],
                y=[optimal_cost],
                mode='markers',
                name='Optimal Point',
                marker=dict(color='red', size=12, symbol='star')
            ))
            
            fig.update_layout(
                title=f"Total Cost Optimization Curve - Item {cost_item}",
                xaxis_title="Order Quantity",
                yaxis_title="Annual Cost",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sensitivity analysis
            st.subheader("🔍 Sensitivity Analysis")
            
            # Parameter sensitivity
            base_eoq = calculate_eoq(annual_demand, ordering_cost, holding_rate, avg_unit_price)
            
            sensitivity_results = []
            parameter_ranges = {
                'Ordering Cost': [ordering_cost * 0.5, ordering_cost, ordering_cost * 1.5, ordering_cost * 2],
                'Holding Rate': [holding_rate * 0.5, holding_rate, holding_rate * 1.5, holding_rate * 2]
            }
            
            for param, values in parameter_ranges.items():
                for value in values:
                    if param == 'Ordering Cost':
                        test_eoq = calculate_eoq(annual_demand, value, holding_rate, avg_unit_price)
                        test_cost = calculate_total_cost(annual_demand, test_eoq, value, holding_rate, avg_unit_price)
                    else:  # Holding Rate
                        test_eoq = calculate_eoq(annual_demand, ordering_cost, value, avg_unit_price)
                        test_cost = calculate_total_cost(annual_demand, test_eoq, ordering_cost, value, avg_unit_price)
                    
                    sensitivity_results.append({
                        'Parameter': param,
                        'Value': value if param == 'Ordering Cost' else value * 100,
                        'Unit': '' if param == 'Ordering Cost' else '%',
                        'EOQ': test_eoq,
                        'Total Cost': test_cost['total_cost']
                    })
            
            sensitivity_df = pd.DataFrame(sensitivity_results)
            
            # Sensitivity visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('EOQ Sensitivity', 'Cost Sensitivity'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            for param in parameter_ranges.keys():
                param_data = sensitivity_df[sensitivity_df['Parameter'] == param]
                
                fig.add_trace(
                    go.Scatter(
                        x=param_data['Value'],
                        y=param_data['EOQ'],
                        mode='lines+markers',
                        name=f'{param} (EOQ)',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=param_data['Value'],
                        y=param_data['Total Cost'],
                        mode='lines+markers',
                        name=f'{param} (Cost)',
                        line=dict(width=2)
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, title_text="Parameter Sensitivity Analysis")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Quantity Discount Analysis")
        
        # Item selection
        discount_item = st.selectbox("Select Item for Discount Analysis", item_options, key="discount_item")
        
        if discount_item:
            item_df = df_clean[df_clean["Item"] == discount_item].copy()
            
            # Calculate base parameters
            data_period_days = (item_df['Creation Date'].max() - item_df['Creation Date'].min()).days
            total_demand = item_df['Qty Delivered'].sum()
            annual_demand = (total_demand / data_period_days * 365) if data_period_days > 0 else total_demand
            base_unit_price = item_df['Unit Price'].mean()
            
            st.write(f"**Base Unit Price:** {base_unit_price:.2f}")
            st.write(f"**Annual Demand:** {annual_demand:,.1f}")
            
            # Define discount tiers
            st.subheader("💰 Define Discount Tiers")
            
            col1, col2 = st.columns(2)
            with col1:
                ordering_cost = st.number_input("Ordering Cost", min_value=0.0, value=100.0, key="discount_ordering")
            with col2:
                holding_rate = st.number_input("Holding Rate (%)", min_value=0.0, value=20.0, key="discount_holding") / 100
            
            # Discount tier inputs
            discount_tiers = []
            
            st.write("**Discount Tiers:**")
            
            # Base tier (no discount)
            discount_tiers.append({
                'name': 'Base Price',
                'min_quantity': 1,
                'unit_price': base_unit_price
            })
            
            # Additional tiers
            num_tiers = st.number_input("Number of Discount Tiers", min_value=1, max_value=5, value=3)
            
            for i in range(num_tiers):
                col1, col2, col3 = st.columns(3)
                with col1:
                    tier_name = st.text_input(f"Tier {i+1} Name", value=f"Tier {i+1}", key=f"tier_name_{i}")
                with col2:
                    min_qty = st.number_input(f"Min Quantity", min_value=1, value=int(annual_demand * (0.1 + i * 0.2)), key=f"tier_qty_{i}")
                with col3:
                    discount_pct = st.number_input(f"Discount %", min_value=0.0, max_value=50.0, value=5.0 + i * 5, key=f"tier_discount_{i}")
                
                discounted_price = base_unit_price * (1 - discount_pct / 100)
                
                discount_tiers.append({
                    'name': tier_name,
                    'min_quantity': min_qty,
                    'unit_price': discounted_price
                })
            
            # Analyze discount scenarios
            if st.button("🔍 Analyze Discount Scenarios", type="primary"):
                discount_results = analyze_quantity_discounts(
                    annual_demand, ordering_cost, holding_rate, base_unit_price, discount_tiers
                )
                
                results_df = pd.DataFrame(discount_results)
                
                # Find optimal tier
                optimal_tier_idx = results_df['total_cost'].idxmin()
                optimal_tier = results_df.loc[optimal_tier_idx]
                
                st.subheader("📊 Discount Analysis Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Optimal Tier", optimal_tier['tier'])
                with col2:
                    st.metric("Optimal Order Qty", f"{optimal_tier['recommended_qty']:.0f}")
                with col3:
                    st.metric("Total Annual Cost", f"{optimal_tier['total_cost']:,.0f}")
                with col4:
                    base_cost = results_df[results_df['tier'] == 'Base Price']['total_cost'].iloc[0]
                    savings = base_cost - optimal_tier['total_cost']
                    st.metric("Annual Savings", f"{savings:,.0f}")
                
                # Results table
                st.dataframe(
                    results_df.style.format({
                        'unit_price': '{:.2f}',
                        'eoq': '{:.0f}',
                        'recommended_qty': '{:.0f}',
                        'total_cost': '{:,.0f}',
                        'purchase_cost': '{:,.0f}',
                        'ordering_cost': '{:,.0f}',
                        'holding_cost': '{:,.0f}',
                        'discount_percent': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Visualization
                fig = go.Figure()
                
                # Total cost by tier
                fig.add_trace(go.Scatter(
                    x=results_df['recommended_qty'],
                    y=results_df['total_cost'],
                    mode='markers+lines',
                    name='Total Cost',
                    text=results_df['tier'],
                    textposition='top center',
                    marker=dict(size=12),
                    line=dict(width=3)
                ))
                
                # Highlight optimal point
                fig.add_trace(go.Scatter(
                    x=[optimal_tier['recommended_qty']],
                    y=[optimal_tier['total_cost']],
                    mode='markers',
                    name='Optimal Choice',
                    marker=dict(color='red', size=15, symbol='star')
                ))
                
                fig.update_layout(
                    title="Total Cost vs Order Quantity (Discount Tiers)",
                    xaxis_title="Order Quantity",
                    yaxis_title="Total Annual Cost",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Break-even analysis
                st.subheader("⚖️ Break-even Analysis")
                
                breakeven_data = []
                for i, result in enumerate(results_df.iterrows()):
                    if i > 0:  # Skip base price
                        tier_data = result[1]
                        base_total_cost = results_df[results_df['tier'] == 'Base Price']['total_cost'].iloc[0]
                        
                        # Calculate break-even quantity
                        cost_per_unit_base = base_total_cost / annual_demand
                        cost_per_unit_tier = tier_data['total_cost'] / annual_demand
                        
                        if cost_per_unit_tier < cost_per_unit_base:
                            breakeven_qty = tier_data['min_quantity']
                        else:
                            breakeven_qty = "Not beneficial"
                        
                        breakeven_data.append({
                            'Tier': tier_data['tier'],
                            'Min Quantity': tier_data['min_quantity'],
                            'Break-even Quantity': breakeven_qty,
                            'Annual Savings': base_total_cost - tier_data['total_cost'],
                            'Cost per Unit': cost_per_unit_tier
                        })
                
                if breakeven_data:
                    breakeven_df = pd.DataFrame(breakeven_data)
                    st.dataframe(
                        breakeven_df.style.format({
                            'Annual Savings': '{:+,.0f}',
                            'Cost per Unit': '{:.2f}'
                        }),
                        use_container_width=True
                    )
    
    with tab4:
        st.subheader("📈 Bulk LOT Size Analysis")
        
        # Parameters for bulk analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            bulk_ordering_cost = st.number_input("Default Ordering Cost", min_value=0.0, value=100.0, key="bulk_ordering")
        with col2:
            bulk_holding_rate = st.number_input("Default Holding Rate (%)", min_value=0.0, value=20.0, key="bulk_holding") / 100
        with col3:
            min_annual_spend = st.number_input("Min Annual Spend Filter", min_value=0, value=1000, key="min_spend")
        
        if st.button("🔄 Analyze All Items", type="primary"):
            with st.spinner("Calculating optimal lot sizes for all items..."):
                bulk_results = []
                
                for item in df_clean['Item'].unique():
                    item_df = df_clean[df_clean["Item"] == item].copy()
                    
                    if len(item_df) < 2:
                        continue
                    
                    # Calculate annual demand
                    data_period_days = (item_df['Creation Date'].max() - item_df['Creation Date'].min()).days
                    total_demand = item_df['Qty Delivered'].sum()
                    annual_demand = (total_demand / data_period_days * 365) if data_period_days > 0 else total_demand
                    
                    avg_unit_price = item_df['Unit Price'].mean()
                    annual_spend = annual_demand * avg_unit_price
                    
                    if annual_spend < min_annual_spend:
                        continue
                    
                    current_avg_order = item_df['Qty Delivered'].mean()
                    
                    # Calculate EOQ
                    eoq = calculate_eoq(annual_demand, bulk_ordering_cost, bulk_holding_rate, avg_unit_price)
                    
                    if eoq > 0:
                        # Calculate costs
                        current_cost = calculate_total_cost(annual_demand, current_avg_order, bulk_ordering_cost, bulk_holding_rate, avg_unit_price)
                        optimal_cost = calculate_total_cost(annual_demand, eoq, bulk_ordering_cost, bulk_holding_rate, avg_unit_price)
                        
                        cost_savings = current_cost['total_cost'] - optimal_cost['total_cost']
                        savings_percent = (cost_savings / current_cost['total_cost']) * 100
                        
                        # Calculate metrics
                        metrics = calculate_reorder_metrics(eoq, annual_demand)
                        
                        item_desc = item_df['Item Description'].iloc[0] if 'Item Description' in item_df.columns else f"Item {item}"
                        
                        bulk_results.append({
                            'Item': item,
                            'Description': item_desc[:40] + "..." if len(item_desc) > 40 else item_desc,
                            'Annual Demand': annual_demand,
                            'Unit Price': avg_unit_price,
                            'Annual Spend': annual_spend,
                            'Current Avg Order': current_avg_order,
                            'Optimal EOQ': eoq,
                            'Current Total Cost': current_cost['total_cost'],
                            'Optimal Total Cost': optimal_cost['total_cost'],
                            'Annual Savings': cost_savings,
                            'Savings %': savings_percent,
                            'Orders per Year': metrics['orders_per_year'],
                            'Inventory Turnover': metrics['inventory_turnover'],
                            'Reorder Point': metrics['reorder_point']
                        })
                
                if bulk_results:
                    bulk_df = pd.DataFrame(bulk_results)
                    bulk_df = bulk_df.sort_values('Annual Savings', ascending=False)
                    
                    # Summary metrics
                    total_savings = bulk_df['Annual Savings'].sum()
                    items_with_savings = len(bulk_df[bulk_df['Annual Savings'] > 0])
                    avg_savings_percent = bulk_df[bulk_df['Annual Savings'] > 0]['Savings %'].mean() if items_with_savings > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Items Analyzed", len(bulk_df))
                    with col2:
                        st.metric("Total Potential Savings", f"{total_savings:,.0f}")
                    with col3:
                        st.metric("Items with Savings", items_with_savings)
                    with col4:
                        st.metric("Avg Savings %", f"{avg_savings_percent:.1f}%")
                    
                    # Top opportunities
                    st.subheader("🎯 Top LOT Size Optimization Opportunities")
                    
                    top_opportunities = bulk_df.head(20)
                    st.dataframe(
                        top_opportunities.style.format({
                            'Annual Demand': '{:,.1f}',
                            'Unit Price': '{:.2f}',
                            'Annual Spend': '{:,.0f}',
                            'Current Avg Order': '{:.1f}',
                            'Optimal EOQ': '{:.0f}',
                            'Current Total Cost': '{:,.0f}',
                            'Optimal Total Cost': '{:,.0f}',
                            'Annual Savings': '{:,.0f}',
                            'Savings %': '{:.1f}%',
                            'Orders per Year': '{:.1f}',
                            'Inventory Turnover': '{:.1f}x',
                            'Reorder Point': '{:.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Visualization
                    fig = px.scatter(
                        bulk_df,
                        x='Annual Spend',
                        y='Savings %',
                        size='Annual Savings',
                        color='Inventory Turnover',
                        hover_data=['Item', 'Optimal EOQ', 'Orders per Year'],
                        title="LOT Size Optimization: Savings vs Annual Spend",
                        labels={'Savings %': 'Savings Percentage (%)', 'Annual Spend': 'Annual Spend'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export option
                    csv = bulk_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export LOT Size Analysis",
                        data=csv,
                        file_name=f"lot_size_optimization_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.warning("No items found meeting the criteria for analysis.")
    
    with tab5:
        st.subheader("⚙️ Advanced Settings & Guidelines")
        
        # Cost parameter guidelines
        st.write("**Parameter Guidelines:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📦 **Ordering Costs Include:**")
            st.write("• Purchase order processing")
            st.write("• Vendor communication")
            st.write("• Receiving and inspection")
            st.write("• Invoice processing")
            st.write("• Shipping and handling")
            st.write("")
            st.markdown("**Typical Range:** $50 - $500 per order")
        
        with col2:
            st.markdown("#### 🏪 **Holding Costs Include:**")
            st.write("• Storage space and utilities")
            st.write("• Insurance and security")
            st.write("• Obsolescence and spoilage")
            st.write("• Capital cost (interest)")
            st.write("• Handling and management")
            st.write("")
            st.markdown("**Typical Range:** 15% - 35% annually")
        
        # Industry benchmarks
        st.subheader("📊 Industry Benchmarks")
        
        benchmark_data = [
            {"Industry": "Manufacturing", "Ordering Cost": "$100-300", "Holding Rate": "20-25%", "Turnover": "6-12x"},
            {"Industry": "Retail", "Ordering Cost": "$50-150", "Holding Rate": "25-35%", "Turnover": "8-15x"},
            {"Industry": "Healthcare", "Ordering Cost": "$75-200", "Holding Rate": "15-25%", "Turnover": "10-20x"},
            {"Industry": "Food & Beverage", "Ordering Cost": "$60-180", "Holding Rate": "30-40%", "Turnover": "15-25x"}
        ]
        
        benchmark_df = pd.DataFrame(benchmark_data)
        st.dataframe(benchmark_df, use_container_width=True)
        
        # EOQ limitations
        st.subheader("⚠️ EOQ Model Limitations")
        
        st.markdown("""
        **Key Assumptions & Limitations:**
        
        • **Constant Demand:** Assumes steady, predictable demand
        • **Fixed Costs:** Ordering and holding costs remain constant
        • **No Stockouts:** Doesn't account for stockout costs
        • **Single Item:** Doesn't consider item interactions
        • **Perfect Information:** Assumes perfect knowledge of parameters
        
        **When to Use Alternatives:**
        
        • **High Demand Variability:** Use stochastic models
        • **Quantity Discounts:** Use price-break analysis
        • **Multiple Items:** Use joint replenishment models
        • **Capacity Constraints:** Use constrained optimization
        """)
        
        # Best practices
        st.subheader("💡 Implementation Best Practices")
        
        practices_col1, practices_col2 = st.columns(2)
        
        with practices_col1:
            st.markdown("#### ✅ **Do's:**")
            st.write("• Regular parameter updates")
            st.write("• ABC analysis integration")
            st.write("• Supplier relationship consideration")
            st.write("• Demand forecast validation")
            st.write("• Storage capacity checks")
        
        with practices_col2:
            st.markdown("#### ❌ **Don'ts:**")
            st.write("• Ignore demand seasonality")
            st.write("• Use outdated cost parameters")
            st.write("• Apply to all items uniformly")
            st.write("• Forget storage constraints")
            st.write("• Neglect supplier minimums")
