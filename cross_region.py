import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def display(df):
    """Cross-Region Vendor Optimization Module - Simplified Version"""
    st.header("🔄 Cross-Region Vendor Optimization")
    st.markdown("Identify price discrepancies and vendor consolidation opportunities across regions.")
    
    # Check for region/location data
    location_columns = ['Region', 'Location', 'State', 'Country', 'Site']
    region_col = None
    
    for col in location_columns:
        if col in df.columns:
            region_col = col
            break
    
    if region_col is None:
        st.warning("⚠️ No region/location data found. Creating mock regions for demonstration.")
        # Create mock regions based on vendor names
        df['Region'] = np.random.choice(['North', 'South', 'East', 'West', 'Central'], len(df))
        region_col = 'Region'
    
    # Clean data
    required_columns = ['Vendor Name', 'Unit Price', 'Item']
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Calculate line total if missing
    if 'Line Total' not in df_clean.columns and 'Qty Delivered' in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Regional Analysis", "💰 Price Optimization", "🤝 Vendor Consolidation"])
    
    with tab1:
        st.subheader("📊 Regional Spending Analysis")
        
        # Regional summary
        regional_summary = df_clean.groupby(region_col).agg({
            'Line Total': 'sum' if 'Line Total' in df_clean.columns else 'count',
            'Vendor Name': 'nunique',
            'Item': 'nunique',
            'Unit Price': 'mean'
        }).round(2)
        
        if 'Line Total' in df_clean.columns:
            regional_summary.columns = ['Total Spend', 'Unique Vendors', 'Unique Items', 'Avg Unit Price']
        else:
            regional_summary.columns = ['Transaction Count', 'Unique Vendors', 'Unique Items', 'Avg Unit Price']
        
        # Display regional metrics
        regions = df_clean[region_col].unique()
        cols = st.columns(len(regions))
        
        for i, region in enumerate(regions):
            with cols[i % len(cols)]:
                region_data = regional_summary.loc[region]
                if 'Total Spend' in regional_summary.columns:
                    st.metric(f"{region} Region", f"${region_data['Total Spend']:,.0f}")
                else:
                    st.metric(f"{region} Region", f"{region_data['Transaction Count']:,.0f} orders")
        
        # Regional comparison table
        st.subheader("🏢 Regional Comparison")
        st.dataframe(
            regional_summary.style.format({
                'Total Spend': '${:,.0f}' if 'Total Spend' in regional_summary.columns else '{:,.0f}',
                'Avg Unit Price': '${:.2f}'
            }),
            use_container_width=True
        )
        
        # Regional spend visualization
        if 'Total Spend' in regional_summary.columns:
            fig = px.bar(regional_summary.reset_index(), 
                        x=region_col, 
                        y='Total Spend',
                        title="Total Spend by Region")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Regional Price Optimization")
        
        # Price analysis by item across regions
        item_region_analysis = df_clean.groupby(['Item', region_col])['Unit Price'].agg(['mean', 'std', 'count']).reset_index()
        item_region_analysis.columns = ['Item', 'Region', 'Avg_Price', 'Price_Std', 'Count']
        
        # Find items with significant price differences
        price_differences = []
        
        for item in df_clean['Item'].unique():
            item_prices = item_region_analysis[item_region_analysis['Item'] == item]
            if len(item_prices) > 1:  # Item exists in multiple regions
                min_price = item_prices['Avg_Price'].min()
                max_price = item_prices['Avg_Price'].max()
                price_diff = max_price - min_price
                price_diff_pct = (price_diff / min_price * 100) if min_price > 0 else 0
                
                if price_diff_pct > 10:  # Significant difference
                    min_region = item_prices.loc[item_prices['Avg_Price'].idxmin(), 'Region']
                    max_region = item_prices.loc[item_prices['Avg_Price'].idxmax(), 'Region']
                    
                    price_differences.append({
                        'Item': item,
                        'Min_Price': min_price,
                        'Max_Price': max_price,
                        'Price_Difference': price_diff,
                        'Difference_%': price_diff_pct,
                        'Cheapest_Region': min_region,
                        'Most_Expensive_Region': max_region
                    })
        
        if price_differences:
            price_diff_df = pd.DataFrame(price_differences)
            price_diff_df = price_diff_df.sort_values('Difference_%', ascending=False)
            
            # Summary metrics
            avg_price_diff = price_diff_df['Difference_%'].mean()
            max_price_diff = price_diff_df['Difference_%'].max()
            items_with_diff = len(price_diff_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Items with Price Gaps", items_with_diff)
            with col2:
                st.metric("Avg Price Difference", f"{avg_price_diff:.1f}%")
            with col3:
                st.metric("Max Price Difference", f"{max_price_diff:.1f}%")
            
            # Price optimization opportunities
            st.subheader("🎯 Price Optimization Opportunities")
            
            st.dataframe(
                price_diff_df.head(15).style.format({
                    'Min_Price': '${:.2f}',
                    'Max_Price': '${:.2f}',
                    'Price_Difference': '${:.2f}',
                    'Difference_%': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            # Potential savings calculation
            if 'Line Total' in df_clean.columns:
                st.subheader("💰 Potential Savings")
                
                total_potential_savings = 0
                savings_details = []
                
                for _, row in price_diff_df.head(10).iterrows():  # Top 10 opportunities
                    item = row['Item']
                    expensive_region = row['Most_Expensive_Region']
                    price_diff_pct = row['Difference_%'] / 100
                    
                    # Calculate spend in expensive region for this item
                    expensive_spend = df_clean[
                        (df_clean['Item'] == item) & 
                        (df_clean[region_col] == expensive_region)
                    ]['Line Total'].sum()
                    
                    potential_savings = expensive_spend * price_diff_pct
                    total_potential_savings += potential_savings
                    
                    savings_details.append({
                        'Item': item,
                        'Expensive_Region_Spend': expensive_spend,
                        'Potential_Savings': potential_savings
                    })
                
                st.metric("Total Potential Savings (Top 10)", f"${total_potential_savings:,.0f}")
                
                if savings_details:
                    savings_df = pd.DataFrame(savings_details)
                    fig = px.bar(savings_df.head(8), 
                                x='Potential_Savings', 
                                y='Item',
                                orientation='h',
                                title="Potential Savings by Item")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No significant price differences found across regions.")
    
    with tab3:
        st.subheader("🤝 Vendor Consolidation Opportunities")
        
        # Vendor analysis across regions
        vendor_region_analysis = df_clean.groupby(['Vendor Name', region_col]).agg({
            'Line Total': 'sum' if 'Line Total' in df_clean.columns else 'count',
            'Item': 'nunique',
            'Unit Price': 'mean'
        }).reset_index()
        
        # Find vendors operating in multiple regions
        multi_region_vendors = vendor_region_analysis.groupby('Vendor Name')[region_col].nunique()
        multi_region_vendors = multi_region_vendors[multi_region_vendors > 1].index
        
        if len(multi_region_vendors) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Multi-Region Vendors", len(multi_region_vendors))
            with col2:
                single_region_vendors = len(df_clean['Vendor Name'].unique()) - len(multi_region_vendors)
                st.metric("Single-Region Vendors", single_region_vendors)
            
            # Multi-region vendor details
            st.subheader("🌍 Multi-Region Vendors")
            
            multi_region_details = []
            for vendor in multi_region_vendors:
                vendor_data = vendor_region_analysis[vendor_region_analysis['Vendor Name'] == vendor]
                regions_served = vendor_data[region_col].tolist()
                total_spend = vendor_data['Line Total'].sum() if 'Line Total' in vendor_data.columns else vendor_data['Line Total'].sum()
                
                multi_region_details.append({
                    'Vendor': vendor,
                    'Regions': ', '.join(regions_served),
                    'Region_Count': len(regions_served),
                    'Total_Business': total_spend
                })
            
            multi_region_df = pd.DataFrame(multi_region_details)
            multi_region_df = multi_region_df.sort_values('Total_Business', ascending=False)
            
            st.dataframe(
                multi_region_df.style.format({
                    'Total_Business': '${:,.0f}' if 'Line Total' in df_clean.columns else '{:,.0f}'
                }),
                use_container_width=True
            )
            
            # Consolidation recommendations
            st.subheader("💡 Consolidation Recommendations")
            
            # Find items purchased from different vendors in different regions
            consolidation_opps = []
            
            for item in df_clean['Item'].unique():
                item_vendors = df_clean[df_clean['Item'] == item].groupby([region_col, 'Vendor Name']).size().reset_index()
                item_vendors.columns = ['Region', 'Vendor', 'Transactions']
                
                regions_with_item = item_vendors['Region'].nunique()
                vendors_for_item = item_vendors['Vendor'].nunique()
                
                if regions_with_item > 1 and vendors_for_item > 1:
                    total_transactions = item_vendors['Transactions'].sum()
                    consolidation_opps.append({
                        'Item': item,
                        'Regions': regions_with_item,
                        'Vendors': vendors_for_item,
                        'Total_Transactions': total_transactions
                    })
            
            if consolidation_opps:
                consolidation_df = pd.DataFrame(consolidation_opps)
                consolidation_df = consolidation_df.sort_values('Total_Transactions', ascending=False)
                
                st.write("**Items with Consolidation Potential:**")
                st.dataframe(consolidation_df.head(10), use_container_width=True)
                
                st.markdown("""
                **Recommended Actions:**
                1. Negotiate enterprise-wide contracts with multi-region vendors
                2. Standardize preferred vendors across regions for common items
                3. Leverage combined purchasing power for better pricing
                4. Establish regional procurement coordination
                """)
            else:
                st.info("Limited consolidation opportunities identified with current data.")
        
        else:
            st.info("No vendors found operating across multiple regions.")

if __name__ == "__main__":
    st.set_page_config(page_title="Cross-Region Vendor Optimization", layout="wide")
    
    # Sample data
    regions = ['North', 'South', 'East', 'West']
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    items = ['Widget', 'Gadget', 'Tool', 'Supply']
    
    sample_data = {
        'Region': np.random.choice(regions, 200),
        'Vendor Name': np.random.choice(vendors, 200),
        'Item': np.random.choice(items, 200),
        'Unit Price': np.random.uniform(10, 100, 200),
        'Qty Delivered': np.random.randint(1, 50, 200)
    }
    df = pd.DataFrame(sample_data)
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    display(df)
