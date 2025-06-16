import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def calculate_savings_opportunity(df):
    """Calculate potential savings by switching to cheapest vendor"""
    savings_data = []
    
    for item in df['Item'].unique():
        item_data = df[df['Item'] == item]
        
        if len(item_data) > 1:  # Only items with multiple vendors
            min_price = item_data['Unit Price'].min()
            
            for _, row in item_data.iterrows():
                if row['Unit Price'] > min_price:
                    potential_savings = (row['Unit Price'] - min_price) * row['Qty Delivered']
                    savings_data.append({
                        'Item': item,
                        'Item Description': row['Item Description'],
                        'Current Vendor': row['Vendor Name'],
                        'Current Price': row['Unit Price'],
                        'Best Price': min_price,
                        'Price Difference': row['Unit Price'] - min_price,
                        'Qty Delivered': row['Qty Delivered'],
                        'Potential Savings': potential_savings,
                        'W/H': row['W/H']
                    })
    
    return pd.DataFrame(savings_data)

def display(df):
    st.header("🔄 Cross-Region Vendor Optimization")
    st.markdown("Identify price discrepancies and optimization opportunities across vendors and regions.")
    
    # Data validation
    required_columns = ['Item', 'Vendor Name', 'W/H', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=['Item', 'Vendor Name', 'Unit Price'])
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Item Analysis", "💰 Savings Opportunities", "📈 Price Trends", "🏢 Vendor Comparison"])
    
    with tab1:
        st.subheader("Item-Level Price Analysis")
        
        # Select item to analyze
        item_options = sorted(df_clean["Item"].dropna().unique())
        selected_item = st.selectbox("Select Item to Analyze", item_options, key="item_analysis")
        
        if selected_item:
            filtered = df_clean[df_clean["Item"] == selected_item]
            
            # Show item description
            item_desc = filtered['Item Description'].iloc[0] if 'Item Description' in filtered.columns else "N/A"
            st.info(f"**Item Description:** {item_desc}")
            
            # Group by vendor and warehouse
            result = (
                filtered.groupby(["Vendor Name", "W/H"])
                .agg({
                    'Unit Price': ['mean', 'min', 'max', 'count'],
                    'Qty Delivered': 'sum',
                    'Line Total': 'sum'
                })
                .round(2)
            )
            
            # Flatten column names
            result.columns = ['Avg_Price', 'Min_Price', 'Max_Price', 'PO_Count', 'Total_Qty', 'Total_Value']
            result = result.reset_index().sort_values(by="Avg_Price")
            
            # Add price difference from cheapest
            min_avg_price = result['Avg_Price'].min()
            result['Price_Difference'] = result['Avg_Price'] - min_avg_price
            result['Price_Premium_%'] = ((result['Avg_Price'] - min_avg_price) / min_avg_price * 100).round(2)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Price Comparison by Vendor and Warehouse:**")
                st.dataframe(
                    result.style.format({
                        'Avg_Price': '{:.2f}',
                        'Min_Price': '{:.2f}',
                        'Max_Price': '{:.2f}',
                        'Total_Value': '{:,.2f}',
                        'Price_Difference': '{:.2f}',
                        'Price_Premium_%': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            with col2:
                # Key metrics
                st.metric("Vendors Compared", len(result))
                st.metric("Price Range", f"{result['Min_Price'].min():.2f} - {result['Max_Price'].max():.2f}")
                st.metric("Max Premium", f"{result['Price_Premium_%'].max():.1f}%")
            
            # Visualization
            if len(result) > 1:
                fig = px.bar(
                    result, 
                    x='Avg_Price', 
                    y='Vendor Name',
                    color='W/H',
                    title=f"Average Unit Price Comparison for Item {selected_item}",
                    labels={'Avg_Price': 'Average Unit Price', 'Vendor Name': 'Vendor'},
                    text='Avg_Price'
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Savings Opportunities")
        
        # Calculate savings
        with st.spinner("Calculating savings opportunities..."):
            savings_df = calculate_savings_opportunity(df_clean)
        
        if len(savings_df) > 0:
            # Summary metrics
            total_savings = savings_df['Potential Savings'].sum()
            avg_price_diff = savings_df['Price Difference'].mean()
            items_with_savings = savings_df['Item'].nunique()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Potential Savings", f"{total_savings:,.2f}")
            with col2:
                st.metric("Avg Price Difference", f"{avg_price_diff:.2f}")
            with col3:
                st.metric("Items with Opportunities", items_with_savings)
            
            # Top savings opportunities
            st.subheader("🎯 Top Savings Opportunities")
            top_savings = savings_df.nlargest(20, 'Potential Savings')[
                ['Item', 'Item Description', 'Current Vendor', 'Current Price', 
                 'Best Price', 'Price Difference', 'Potential Savings', 'W/H']
            ]
            
            st.dataframe(
                top_savings.style.format({
                    'Current Price': '{:.2f}',
                    'Best Price': '{:.2f}',
                    'Price Difference': '{:.2f}',
                    'Potential Savings': '{:,.2f}'
                }),
                use_container_width=True
            )
            
            # Visualization
            fig = px.scatter(
                savings_df.head(50),
                x='Price Difference',
                y='Potential Savings',
                color='W/H',
                size='Qty Delivered',
                hover_data=['Item', 'Current Vendor', 'Current Price'],
                title="Savings Opportunities: Price Difference vs Potential Savings"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export option
            if st.button("📥 Export Savings Report"):
                csv = savings_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"savings_opportunities_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No savings opportunities found. All items may be optimally priced.")
    
    with tab3:
        st.subheader("📈 Price Trends Analysis")
        
        # Select multiple items for trend comparison
        selected_items = st.multiselect(
            "Select Items for Trend Analysis",
            options=sorted(df_clean["Item"].dropna().unique()),
            max_selections=5
        )
        
        if selected_items and 'Creation Date' in df_clean.columns:
            trend_data = df_clean[df_clean['Item'].isin(selected_items)]
            
            # Monthly price trends
            trend_data['Month'] = pd.to_datetime(trend_data['Creation Date']).dt.to_period('M')
            monthly_prices = trend_data.groupby(['Month', 'Item', 'Vendor Name'])['Unit Price'].mean().reset_index()
            monthly_prices['Month'] = monthly_prices['Month'].astype(str)
            
            # Create trend chart
            fig = px.line(
                monthly_prices,
                x='Month',
                y='Unit Price',
                color='Item',
                line_dash='Vendor Name',
                title="Price Trends by Item and Vendor",
                labels={'Unit Price': 'Average Unit Price', 'Month': 'Month'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Price volatility analysis
            st.subheader("📊 Price Volatility")
            volatility = trend_data.groupby('Item')['Unit Price'].agg(['std', 'mean']).reset_index()
            volatility['CV'] = (volatility['std'] / volatility['mean'] * 100).round(2)
            volatility = volatility.sort_values('CV', ascending=False)
            
            fig = px.bar(
                volatility.head(10),
                x='Item',
                y='CV',
                title="Price Volatility (Coefficient of Variation) by Item",
                labels={'CV': 'Coefficient of Variation (%)', 'Item': 'Item ID'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🏢 Vendor Performance Comparison")
        
        # Vendor performance metrics
        vendor_metrics = df_clean.groupby('Vendor Name').agg({
            'Unit Price': ['mean', 'std'],
            'Item': 'nunique',
            'Line Total': 'sum',
            'Qty Delivered': 'sum'
        }).round(2)
        
        vendor_metrics.columns = ['Avg_Price', 'Price_Std', 'Unique_Items', 'Total_Value', 'Total_Qty']
        vendor_metrics = vendor_metrics.reset_index()
        vendor_metrics['Price_CV'] = (vendor_metrics['Price_Std'] / vendor_metrics['Avg_Price'] * 100).round(2)
        
        # Sort by total value
        vendor_metrics = vendor_metrics.sort_values('Total_Value', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Vendors by Value:**")
            st.dataframe(
                vendor_metrics.head(15).style.format({
                    'Avg_Price': '{:.2f}',
                    'Price_Std': '{:.2f}',
                    'Total_Value': '{:,.2f}',
                    'Price_CV': '{:.1f}%'
                }),
                use_container_width=True
            )
        
        with col2:
            # Vendor diversification chart
            fig = px.scatter(
                vendor_metrics,
                x='Unique_Items',
                y='Total_Value',
                size='Total_Qty',
                color='Price_CV',
                hover_name='Vendor Name',
                title="Vendor Portfolio: Diversification vs Value",
                labels={
                    'Unique_Items': 'Number of Unique Items',
                    'Total_Value': 'Total Purchase Value',
                    'Price_CV': 'Price Volatility (%)'
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional vendor distribution
        if 'W/H' in df_clean.columns:
            st.subheader("🌍 Regional Vendor Distribution")
            regional_dist = df_clean.groupby(['W/H', 'Vendor Name']).size().reset_index(name='PO_Count')
            
            fig = px.sunburst(
                regional_dist,
                path=['W/H', 'Vendor Name'],
                values='PO_Count',
                title="Purchase Orders by Region and Vendor"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
