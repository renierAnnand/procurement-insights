import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from math import sqrt

def display(df):
    """LOT Size Optimization Module - Fixed Version"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management.")
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("This module requires: Item, Unit Price, and Qty Delivered columns")
        return
    
    # Clean data with proper error handling
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    
    # Convert to numeric and handle mixed data types
    df_clean['Unit Price'] = pd.to_numeric(df_clean['Unit Price'], errors='coerce')
    df_clean['Qty Delivered'] = pd.to_numeric(df_clean['Qty Delivered'], errors='coerce')
    
    # Remove rows with invalid numeric data
    df_clean = df_clean.dropna(subset=['Unit Price', 'Qty Delivered'])
    
    # Now safely filter for positive values
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        st.info("Please ensure Unit Price and Qty Delivered contain positive numeric values")
        return
    
    st.success(f"✅ Successfully processed {len(df_clean)} records")
    
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
            annual_demand = float(item_data['Qty Delivered'].sum())
            avg_unit_cost = float(item_data['Unit Price'].mean())
            holding_cost = avg_unit_cost * holding_cost_rate
            
            # EOQ calculation
            if annual_demand > 0 and holding_cost > 0:
                eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                # Current average order size
                current_avg_order = float(item_data['Qty Delivered'].mean())
                
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
                
                # Additional insights
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
                
                # EOQ curve
                try:
                    order_sizes = np.arange(10, eoq * 3, max(1, int(eoq * 0.1)))
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
                except Exception as e:
                    st.warning("Chart generation failed, but calculations above are still valid.")
    
    with tab2:
        st.subheader("💰 Portfolio Cost Optimization")
        
        # Calculate EOQ for all items
        optimization_results = []
        
        for item in df_clean['Item'].unique():
            item_data = df_clean[df_clean['Item'] == item]
            
            if len(item_data) >= 3:  # Need minimum data points
                try:
                    annual_demand = float(item_data['Qty Delivered'].sum())
                    avg_unit_cost = float(item_data['Unit Price'].mean())
                    current_avg_order = float(item_data['Qty Delivered'].mean())
                    
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
                except:
                    # Skip items with calculation errors
                    continue
        
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
            
            # Format the dataframe for display
            formatted_df = display_df.copy()
            formatted_df['Current Avg Order'] = formatted_df['Current Avg Order'].apply(lambda x: f"{x:.0f}")
            formatted_df['Optimal EOQ'] = formatted_df['Optimal EOQ'].apply(lambda x: f"{x:.0f}")
            formatted_df['Potential Savings'] = formatted_df['Potential Savings'].apply(lambda x: f"${x:,.0f}")
            formatted_df['Savings %'] = formatted_df['Savings %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Visualization
            try:
                fig = px.bar(results_df.head(10), 
                            x='Potential Savings', 
                            y='Item',
                            orientation='h',
                            title="Top 10 Items by Savings Potential")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Visualization failed, but data table above shows the results.")
                
            # Export functionality
            if st.button("📊 Download Results"):
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="eoq_optimization_results.csv",
                    mime="text/csv"
                )
        else:
            st.info("Need more data to perform EOQ optimization.")

if __name__ == "__main__":
    st.set_page_config(page_title="LOT Size Optimization", layout="wide")
    
    # Sample data with proper numeric types
    np.random.seed(42)  # For reproducible results
    sample_data = {
        'Item': np.random.choice(['Product A', 'Product B', 'Product C'], 100),
        'Unit Price': np.round(np.random.uniform(5.0, 50.0, 100), 2),
        'Qty Delivered': np.random.randint(10, 200, 100).astype(float)
    }
    df = pd.DataFrame(sample_data)
    
    # Ensure proper data types
    df['Unit Price'] = df['Unit Price'].astype(float)
    df['Qty Delivered'] = df['Qty Delivered'].astype(float)
    
    display(df)
