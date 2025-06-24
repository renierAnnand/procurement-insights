import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from math import sqrt

# Regional currency mapping
REGION_CURRENCIES = {
    'Saudi Arabia': {'symbol': 'SR', 'name': 'SAR'},
    'UAE': {'symbol': 'AED', 'name': 'AED'},
    'Kuwait': {'symbol': 'KD', 'name': 'KWD'},
    'Qatar': {'symbol': 'QR', 'name': 'QAR'},
    'Bahrain': {'symbol': 'BD', 'name': 'BHD'},
    'Oman': {'symbol': 'OMR', 'name': 'OMR'},
    'Egypt': {'symbol': 'EGP', 'name': 'EGP'},
    'Jordan': {'symbol': 'JD', 'name': 'JOD'},
    'Lebanon': {'symbol': 'LBP', 'name': 'LBP'},
    'USA': {'symbol': '$', 'name': 'USD'},
    'Europe': {'symbol': '€', 'name': 'EUR'},
    'UK': {'symbol': '£', 'name': 'GBP'}
}

def get_currency_info(region):
    """Get currency symbol and name for a region"""
    return REGION_CURRENCIES.get(region, {'symbol': '$', 'name': 'USD'})

def display(df):
    """Enhanced LOT Size Optimization Module"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management with regional support.")
    
    # Add methodology expander (enhances original's educational value)
    with st.expander("📚 EOQ Methodology & Assumptions", expanded=False):
        st.markdown("""
        **Economic Order Quantity (EOQ) Formula:** EOQ = √((2 × Annual Demand × Ordering Cost) / Holding Cost per Unit)
        
        **Key Assumptions:**
        - Constant demand rate throughout the year
        - Fixed ordering cost per order
        - Fixed holding cost per unit per year
        - No stockouts or backorders
        - Instant replenishment (zero lead time)
        
        **Cost Components:**
        - **Ordering Cost:** Cost incurred each time an order is placed
        - **Holding Cost:** Cost of storing one unit for one year (storage, insurance, obsolescence, etc.)
        - **Total Cost:** Sum of annual ordering and holding costs
        """)
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("This module requires: Item, Unit Price, Qty Delivered, and optionally Region columns")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Region filter (if Region column exists)
    if 'Region' in df_clean.columns:
        st.sidebar.header("🌍 Regional Settings")
        available_regions = sorted(df_clean['Region'].unique())
        selected_region = st.sidebar.selectbox("Select Region", available_regions)
        df_filtered = df_clean[df_clean['Region'] == selected_region]
        
        # Currency information
        currency_info = get_currency_info(selected_region)
        currency_symbol = currency_info['symbol']
        currency_name = currency_info['name']
        
        st.sidebar.info(f"**Currency:** {currency_name} ({currency_symbol})")
    else:
        # Default to showing all data and USD
        df_filtered = df_clean
        selected_region = "Global"
        currency_symbol = "$"
        currency_name = "USD"
        st.sidebar.info("**Note:** Add 'Region' column to enable regional filtering")
    
    if len(df_filtered) == 0:
        st.warning(f"No data found for region: {selected_region}")
        return
    
    # Display current region info
    st.info(f"📍 **Region:** {selected_region} | **Currency:** {currency_name} ({currency_symbol}) | **Items:** {len(df_filtered):,}")
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 EOQ Analysis", "💰 Cost Optimization"])
    
    with tab1:
        st.subheader("📊 Economic Order Quantity Analysis")
        
        # Enhanced Parameters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            holding_cost_type = st.radio("Holding Cost Type", ["Percentage (%)", "Fixed Amount"])
        
        with col2:
            if holding_cost_type == "Percentage (%)":
                holding_cost_rate = st.slider("Holding Cost Rate (%)", 5, 30, 15) / 100
                holding_cost_fixed = None
            else:
                holding_cost_fixed = st.number_input(f"Holding Cost ({currency_symbol})", 0.1, 50.0, 2.5, step=0.1)
                holding_cost_rate = None
        
        with col3:
            ordering_cost = st.number_input(f"Ordering Cost ({currency_symbol})", 50, 500, 100)
        
        with col4:
            working_days = st.number_input("Working Days/Year", 200, 365, 250)
            st.caption("ℹ️ For future lead time calculations")
        
        # Item selection
        items = sorted(df_filtered['Item'].unique())
        selected_item = st.selectbox("Select Item for EOQ Analysis", items)
        
        if selected_item:
            item_data = df_filtered[df_filtered['Item'] == selected_item]
            
            # Calculate demand and costs
            annual_demand = item_data['Qty Delivered'].sum()
            avg_unit_cost = item_data['Unit Price'].mean()
            
            # Calculate holding cost
            if holding_cost_type == "Percentage (%)":
                holding_cost = avg_unit_cost * holding_cost_rate
                holding_cost_display = f"{holding_cost_rate*100:.1f}% of unit cost"
            else:
                holding_cost = holding_cost_fixed
                holding_cost_display = f"{currency_symbol}{holding_cost_fixed:.2f} per unit"
            
            # EOQ calculation
            if annual_demand > 0 and holding_cost > 0:
                eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                # Current average order size
                current_avg_order = item_data['Qty Delivered'].mean()
                
                # Cost calculation functions
                def ordering_cost_func(order_qty):
                    if order_qty <= 0:
                        return float('inf')
                    return (annual_demand / order_qty) * ordering_cost
                
                def holding_cost_func(order_qty):
                    return (order_qty / 2) * holding_cost
                
                def total_cost(order_qty):
                    if order_qty <= 0:
                        return float('inf')
                    return ordering_cost_func(order_qty) + holding_cost_func(order_qty)
                
                eoq_cost = total_cost(eoq)
                current_cost = total_cost(current_avg_order)
                potential_savings = current_cost - eoq_cost
                
                # Display results
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Annual Demand", f"{annual_demand:,.0f}")
                with col2:
                    st.metric("Optimal Order Qty (EOQ)", f"{eoq:.0f}")
                with col3:
                    st.metric("Current Avg Order", f"{current_avg_order:.0f}")
                with col4:
                    st.metric("Potential Savings", f"{currency_symbol}{potential_savings:,.0f}")
                with col5:
                    st.metric("Holding Cost", holding_cost_display)
                
                # Additional insights (similar to original's comprehensive approach)
                col1, col2, col3 = st.columns(3)
                with col1:
                    orders_per_year_eoq = annual_demand / eoq if eoq > 0 else 0
                    st.metric("Orders/Year (EOQ)", f"{orders_per_year_eoq:.1f}")
                with col2:
                    orders_per_year_current = annual_demand / current_avg_order if current_avg_order > 0 else 0
                    st.metric("Orders/Year (Current)", f"{orders_per_year_current:.1f}")
                with col3:
                    cycle_time_eoq = working_days / orders_per_year_eoq if orders_per_year_eoq > 0 else 0
                    st.metric("Days Between Orders (EOQ)", f"{cycle_time_eoq:.0f}")
                
                # Enhanced EOQ curve with cost breakdown
                order_sizes = np.arange(max(10, eoq * 0.1), eoq * 3, max(1, int(eoq * 0.1)))
                total_costs = [total_cost(q) for q in order_sizes]
                ordering_costs = [ordering_cost_func(q) for q in order_sizes]
                holding_costs = [holding_cost_func(q) for q in order_sizes]
                
                fig = go.Figure()
                
                # Add cost breakdown lines
                fig.add_trace(go.Scatter(
                    x=order_sizes, y=ordering_costs, 
                    name='Ordering Costs', 
                    line=dict(width=2, color='blue', dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=order_sizes, y=holding_costs, 
                    name='Holding Costs', 
                    line=dict(width=2, color='green', dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=order_sizes, y=total_costs, 
                    name='Total Cost', 
                    line=dict(width=3, color='red')
                ))
                
                # Add vertical lines for EOQ and current order size
                fig.add_vline(
                    x=eoq, line_dash="dash", line_color="red", line_width=2,
                    annotation_text=f"EOQ: {eoq:.0f}"
                )
                fig.add_vline(
                    x=current_avg_order, line_dash="dash", line_color="orange", line_width=2,
                    annotation_text=f"Current: {current_avg_order:.0f}"
                )
                
                # Add cost points
                fig.add_trace(go.Scatter(
                    x=[eoq], y=[eoq_cost],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name=f'EOQ Cost: {currency_symbol}{eoq_cost:,.0f}'
                ))
                fig.add_trace(go.Scatter(
                    x=[current_avg_order], y=[current_cost],
                    mode='markers',
                    marker=dict(size=10, color='orange'),
                    name=f'Current Cost: {currency_symbol}{current_cost:,.0f}'
                ))
                
                fig.update_layout(
                    title=f"EOQ Cost Analysis - {selected_item} ({selected_region})",
                    xaxis_title="Order Quantity",
                    yaxis_title=f"Annual Cost ({currency_symbol})",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cost breakdown table
                st.subheader("💡 Cost Breakdown Comparison")
                breakdown_data = {
                    'Scenario': ['Current Practice', 'Optimal EOQ'],
                    'Order Quantity': [current_avg_order, eoq],
                    f'Ordering Cost ({currency_symbol})': [ordering_cost_func(current_avg_order), ordering_cost_func(eoq)],
                    f'Holding Cost ({currency_symbol})': [holding_cost_func(current_avg_order), holding_cost_func(eoq)],
                    f'Total Cost ({currency_symbol})': [current_cost, eoq_cost]
                }
                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(
                    breakdown_df.style.format({
                        'Order Quantity': '{:.0f}',
                        f'Ordering Cost ({currency_symbol})': '{:,.0f}',
                        f'Holding Cost ({currency_symbol})': '{:,.0f}',
                        f'Total Cost ({currency_symbol})': '{:,.0f}'
                    }),
                    use_container_width=True
                )
    
    with tab2:
        st.subheader(f"💰 Portfolio Cost Optimization - {selected_region}")
        
        # Parameters from Tab 1 are automatically used here
        st.info(f"📋 Using parameters from EOQ Analysis tab: " +
                f"Holding Cost: {holding_cost_display if 'holding_cost_display' in locals() else 'Not set'}, " +
                f"Ordering Cost: {currency_symbol}{ordering_cost}, " +
                f"Working Days: {working_days}")
        
        # Use the same parameters from tab1 - maintain original behavior
        if 'holding_cost_type' not in locals():
            holding_cost_type = "Percentage (%)"
            holding_cost_rate = 0.15
            holding_cost_fixed = None
            ordering_cost = 100
        
        # Calculate EOQ for all items in the filtered region
        optimization_results = []
        
        for item in df_filtered['Item'].unique():
            item_data = df_filtered[df_filtered['Item'] == item]
            
            if len(item_data) >= 3:  # Need minimum data points for statistical reliability
                annual_demand = item_data['Qty Delivered'].sum()
                avg_unit_cost = item_data['Unit Price'].mean()
                current_avg_order = item_data['Qty Delivered'].mean()
                
                # Calculate holding cost based on type
                if holding_cost_type == "Percentage (%)":
                    holding_cost = avg_unit_cost * holding_cost_rate
                else:
                    holding_cost = holding_cost_fixed
                
                if annual_demand > 0 and holding_cost > 0:
                    eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                    
                    def total_cost(order_qty):
                        if order_qty <= 0:
                            return float('inf')
                        ordering_cost_total = (annual_demand / order_qty) * ordering_cost
                        holding_cost_total = (order_qty / 2) * holding_cost
                        return ordering_cost_total + holding_cost_total
                    
                    eoq_cost = total_cost(eoq)
                    current_cost = total_cost(current_avg_order)
                    potential_savings = current_cost - eoq_cost
                    
                    optimization_results.append({
                        'Item': item,
                        'Annual Demand': annual_demand,
                        'Current Avg Order': current_avg_order,
                        'Optimal EOQ': eoq,
                        'Current Cost': current_cost,
                        'EOQ Cost': eoq_cost,
                        'Potential Savings': potential_savings,
                        'Savings %': (potential_savings / current_cost * 100) if current_cost > 0 else 0
                    })
        
        if optimization_results:
            results_df = pd.DataFrame(optimization_results)
            results_df = results_df.sort_values('Potential Savings', ascending=False)
            
            # Summary metrics
            total_savings = results_df['Potential Savings'].sum()
            avg_savings_pct = results_df['Savings %'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Potential Savings", f"{currency_symbol}{total_savings:,.0f}")
            with col2:
                st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
            with col3:
                st.metric("Items Analyzed", len(results_df))
            with col4:
                st.metric("Region", selected_region)
            
            # Top opportunities
            st.subheader("🎯 Top Optimization Opportunities")
            
            display_df = results_df.head(15)[['Item', 'Current Avg Order', 'Optimal EOQ', 'Potential Savings', 'Savings %']]
            
            st.dataframe(
                display_df.style.format({
                    'Current Avg Order': '{:.0f}',
                    'Optimal EOQ': '{:.0f}',
                    'Potential Savings': f'{currency_symbol}{{:,.0f}}',
                    'Savings %': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            # Visualization
            fig = px.bar(
                results_df.head(10), 
                x='Potential Savings', 
                y='Item',
                orientation='h',
                title=f"Top 10 Items by Savings Potential - {selected_region}",
                labels={'Potential Savings': f'Potential Savings ({currency_symbol})'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export option
            if st.button("📊 Download Optimization Report"):
                export_df = results_df.copy()
                export_df['Region'] = selected_region
                export_df['Currency'] = currency_name
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"eoq_optimization_{selected_region.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        else:
            st.info(f"Need more data to perform EOQ optimization for {selected_region}.")

if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced LOT Size Optimization", layout="wide")
    
    # Enhanced sample data with regions
    regions = ['Saudi Arabia', 'UAE', 'Kuwait', 'Qatar', 'Bahrain', 'Egypt']
    sample_data = {
        'Item': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D', 'Product E'], 150),
        'Unit Price': np.random.uniform(5, 50, 150),
        'Qty Delivered': np.random.randint(10, 200, 150),
        'Region': np.random.choice(regions, 150)
    }
    df = pd.DataFrame(sample_data)
    display(df)
