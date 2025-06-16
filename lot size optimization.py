import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from math import sqrt

def display(df):
    """LOT Size Optimization Module - Simplified Version"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management.")
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("This module requires: Item, Unit Price, and Qty Delivered columns")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 EOQ Analysis", "💰 Cost Optimization"])
    
    with tab1:
        st.subheader("📊 Economic Order Quantity Analysis")
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            holding_cost_rate = st.slider("Holding Cost Rate (%)", 5, 30, 15) / 100
        with col2:
            ordering_cost = st.number_input("Ordering Cost ($)", 50, 500, 100)
        with col3:
            working_days = st.number_input("Working Days/Year", 200, 365, 250)
        
        # Item selection
        items = sorted(df_clean['Item'].unique())
        selected_item = st.selectbox("Select Item for EOQ Analysis", items)
        
        if selected_item:
            item_data = df_clean[df_clean['Item'] == selected_item]
            
            # Calculate demand and costs
            annual_demand = item_data['Qty Delivered'].sum()
            avg_unit_cost = item_data['Unit Price'].mean()
            holding_cost = avg_unit_cost * holding_cost_rate
            
            # EOQ calculation
            if annual_demand > 0 and holding_cost > 0:
                eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                # Current average order size
                current_avg_order = item_data['Qty Delivered'].mean()
                
                # Total costs
                def total_cost(order_qty):
                    if order_qty <= 0:
                        return float('inf')
                    ordering_cost_total = (annual_demand / order_qty) * ordering_cost
                    holding_cost_total = (order_qty / 2) * holding_cost
                    return ordering_cost_total + holding_cost_total
                
                eoq_cost = total_cost(eoq)
                current_cost = total_cost(current_avg_order)
                potential_savings = current_cost - eoq_cost
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Annual Demand", f"{annual_demand:,.0f}")
                with col2:
                    st.metric("Optimal Order Qty (EOQ)", f"{eoq:.0f}")
                with col3:
                    st.metric("Current Avg Order", f"{current_avg_order:.0f}")
                with col4:
                    st.metric("Potential Savings", f"${potential_savings:,.0f}")
                
                # EOQ curve
                order_sizes = np.arange(10, eoq * 3, 10)
                costs = [total_cost(q) for q in order_sizes]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=order_sizes, y=costs, name='Total Cost', line=dict(width=3)))
                fig.add_vline(x=eoq, line_dash="dash", line_color="red", 
                             annotation_text=f"EOQ: {eoq:.0f}")
                fig.add_vline(x=current_avg_order, line_dash="dash", line_color="blue",
                             annotation_text=f"Current: {current_avg_order:.0f}")
                
                fig.update_layout(
                    title=f"EOQ Cost Curve - {selected_item}",
                    xaxis_title="Order Quantity",
                    yaxis_title="Total Annual Cost ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Portfolio Cost Optimization")
        
        # Calculate EOQ for all items
        optimization_results = []
        
        for item in df_clean['Item'].unique():
            item_data = df_clean[df_clean['Item'] == item]
            
            if len(item_data) >= 3:  # Need minimum data points
                annual_demand = item_data['Qty Delivered'].sum()
                avg_unit_cost = item_data['Unit Price'].mean()
                current_avg_order = item_data['Qty Delivered'].mean()
                
                holding_cost = avg_unit_cost * holding_cost_rate
                
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
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Potential Savings", f"${total_savings:,.0f}")
            with col2:
                st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
            with col3:
                st.metric("Items Analyzed", len(results_df))
            
            # Top opportunities
            st.subheader("🎯 Top Optimization Opportunities")
            
            display_df = results_df.head(15)[['Item', 'Current Avg Order', 'Optimal EOQ', 'Potential Savings', 'Savings %']]
            
            st.dataframe(
                display_df.style.format({
                    'Current Avg Order': '{:.0f}',
                    'Optimal EOQ': '{:.0f}',
                    'Potential Savings': '${:,.0f}',
                    'Savings %': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            # Visualization
            fig = px.bar(results_df.head(10), 
                        x='Potential Savings', 
                        y='Item',
                        orientation='h',
                        title="Top 10 Items by Savings Potential")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more data to perform EOQ optimization.")

if __name__ == "__main__":
    st.set_page_config(page_title="LOT Size Optimization", layout="wide")
    
    # Sample data
    sample_data = {
        'Item': np.random.choice(['Product A', 'Product B', 'Product C'], 100),
        'Unit Price': np.random.uniform(5, 50, 100),
        'Qty Delivered': np.random.randint(10, 200, 100)
    }
    df = pd.DataFrame(sample_data)
    display(df)
