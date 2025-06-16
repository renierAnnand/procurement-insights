import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def display(df):
    """Seasonal Price Optimization Module - Simplified Version"""
    st.header("🌟 Seasonal Price Optimization")
    st.markdown("Optimize purchase timing based on seasonal price patterns for maximum cost savings.")
    
    # Basic data validation
    required_columns = ['Creation Date', 'Unit Price', 'Item']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("This module requires: Creation Date, Unit Price, and Item columns")
        return
    
    # Clean and prepare data
    df_clean = df.copy()
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date', 'Unit Price', 'Item'])
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Add date components
    df_clean['Year'] = df_clean['Creation Date'].dt.year
    df_clean['Month'] = df_clean['Creation Date'].dt.month
    df_clean['Quarter'] = df_clean['Creation Date'].dt.quarter
    df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name()
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📊 Price Patterns", "📅 Optimal Timing", "💰 Savings Calculator"])
    
    with tab1:
        st.subheader("📊 Seasonal Price Patterns")
        
        # Item selection
        items = sorted(df_clean['Item'].unique())
        selected_item = st.selectbox("Select Item for Analysis", items)
        
        if selected_item:
            item_data = df_clean[df_clean['Item'] == selected_item]
            
            if len(item_data) < 5:
                st.warning(f"Not enough data points for {selected_item} (need at least 5)")
            else:
                # Monthly price trends
                monthly_prices = item_data.groupby('Month_Name')['Unit Price'].agg(['mean', 'std', 'count']).reset_index()
                
                # Order months correctly
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                monthly_prices['Month_Order'] = monthly_prices['Month_Name'].apply(
                    lambda x: month_order.index(x) if x in month_order else 12
                )
                monthly_prices = monthly_prices.sort_values('Month_Order')
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_price = item_data['Unit Price'].mean()
                    st.metric("Average Price", f"${avg_price:.2f}")
                with col2:
                    min_month = monthly_prices.loc[monthly_prices['mean'].idxmin(), 'Month_Name']
                    min_price = monthly_prices['mean'].min()
                    st.metric("Cheapest Month", min_month, f"${min_price:.2f}")
                with col3:
                    max_month = monthly_prices.loc[monthly_prices['mean'].idxmax(), 'Month_Name']
                    max_price = monthly_prices['mean'].max()
                    st.metric("Most Expensive Month", max_month, f"${max_price:.2f}")
                
                # Price trend chart
                fig = px.line(monthly_prices, x='Month_Name', y='mean',
                             title=f"Monthly Price Trends - {selected_item}",
                             labels={'mean': 'Average Unit Price ($)', 'Month_Name': 'Month'})
                fig.update_traces(line=dict(width=3))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Price volatility
                st.subheader("📈 Price Volatility Analysis")
                
                price_volatility = (monthly_prices['std'] / monthly_prices['mean'] * 100).fillna(0)
                volatility_df = pd.DataFrame({
                    'Month': monthly_prices['Month_Name'],
                    'Volatility %': price_volatility
                })
                
                fig = px.bar(volatility_df, x='Month', y='Volatility %',
                            title="Price Volatility by Month",
                            labels={'Volatility %': 'Price Volatility (%)'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📅 Optimal Purchase Timing")
        
        # Calculate seasonal indices for all items
        seasonal_analysis = []
        
        for item in df_clean['Item'].unique():
            item_data = df_clean[df_clean['Item'] == item]
            if len(item_data) >= 6:  # Need minimum data points
                monthly_avg = item_data.groupby('Month')['Unit Price'].mean()
                overall_avg = item_data['Unit Price'].mean()
                
                for month in monthly_avg.index:
                    seasonal_index = (monthly_avg[month] / overall_avg) * 100
                    seasonal_analysis.append({
                        'Item': item,
                        'Month': month,
                        'Month_Name': pd.to_datetime(f'2024-{month:02d}-01').strftime('%B'),
                        'Seasonal_Index': seasonal_index,
                        'Price_Level': 'Low' if seasonal_index < 95 else 'High' if seasonal_index > 105 else 'Normal'
                    })
        
        if seasonal_analysis:
            seasonal_df = pd.DataFrame(seasonal_analysis)
            
            # Best buying opportunities
            st.subheader("🎯 Best Buying Opportunities")
            
            low_price_items = seasonal_df[seasonal_df['Price_Level'] == 'Low'].groupby('Month_Name').size().reset_index()
            low_price_items.columns = ['Month', 'Items_With_Low_Prices']
            low_price_items = low_price_items.sort_values('Items_With_Low_Prices', ascending=False)
            
            if len(low_price_items) > 0:
                st.write("**Months with Most Low-Price Items:**")
                st.dataframe(low_price_items.head(6), use_container_width=True)
                
                # Heatmap of seasonal patterns
                pivot_data = seasonal_df.pivot(index='Item', columns='Month_Name', values='Seasonal_Index')
                
                fig = px.imshow(pivot_data.iloc[:10],  # Show top 10 items
                               title="Seasonal Price Heatmap (Top 10 Items)",
                               labels=dict(x="Month", y="Item", color="Seasonal Index"),
                               color_continuous_scale="RdYlGn_r")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Purchase recommendations
            st.subheader("💡 Purchase Recommendations")
            
            current_month = datetime.now().month
            current_month_name = datetime.now().strftime('%B')
            
            current_month_items = seasonal_df[
                (seasonal_df['Month'] == current_month) & 
                (seasonal_df['Price_Level'] == 'Low')
            ]
            
            if len(current_month_items) > 0:
                st.success(f"🎉 **Great Timing!** {len(current_month_items)} items are at low prices in {current_month_name}")
                st.dataframe(
                    current_month_items[['Item', 'Seasonal_Index']].sort_values('Seasonal_Index').head(10),
                    use_container_width=True
                )
            else:
                st.info(f"ℹ️ No items are at particularly low prices in {current_month_name}")
    
    with tab3:
        st.subheader("💰 Potential Savings Calculator")
        
        # Calculate potential savings from optimal timing
        if len(seasonal_analysis) > 0:
            seasonal_df = pd.DataFrame(seasonal_analysis)
            
            # For each item, calculate potential savings
            savings_analysis = []
            
            for item in seasonal_df['Item'].unique():
                item_seasonal = seasonal_df[seasonal_df['Item'] == item]
                min_index = item_seasonal['Seasonal_Index'].min()
                max_index = item_seasonal['Seasonal_Index'].max()
                
                potential_savings = ((max_index - min_index) / max_index) * 100
                
                # Get actual spend data
                item_spend = df_clean[df_clean['Item'] == item]
                if 'Line Total' in item_spend.columns:
                    annual_spend = item_spend['Line Total'].sum()
                else:
                    annual_spend = (item_spend['Unit Price'] * item_spend.get('Qty Delivered', 1)).sum()
                
                estimated_savings = annual_spend * (potential_savings / 100)
                
                best_month = item_seasonal.loc[item_seasonal['Seasonal_Index'].idxmin(), 'Month_Name']
                worst_month = item_seasonal.loc[item_seasonal['Seasonal_Index'].idxmax(), 'Month_Name']
                
                savings_analysis.append({
                    'Item': item,
                    'Annual_Spend': annual_spend,
                    'Potential_Savings_%': potential_savings,
                    'Estimated_Savings_$': estimated_savings,
                    'Best_Month': best_month,
                    'Worst_Month': worst_month
                })
            
            savings_df = pd.DataFrame(savings_analysis)
            savings_df = savings_df.sort_values('Estimated_Savings_$', ascending=False)
            
            # Summary metrics
            total_potential_savings = savings_df['Estimated_Savings_$'].sum()
            total_annual_spend = savings_df['Annual_Spend'].sum()
            avg_savings_percent = (total_potential_savings / total_annual_spend * 100) if total_annual_spend > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Potential Savings", f"${total_potential_savings:,.0f}")
            with col2:
                st.metric("Average Savings %", f"{avg_savings_percent:.1f}%")
            with col3:
                st.metric("Items Analyzed", len(savings_df))
            
            # Top savings opportunities
            st.subheader("🎯 Top Savings Opportunities")
            
            display_df = savings_df.head(15)[['Item', 'Annual_Spend', 'Potential_Savings_%', 'Estimated_Savings_$', 'Best_Month']]
            
            st.dataframe(
                display_df.style.format({
                    'Annual_Spend': '${:,.0f}',
                    'Potential_Savings_%': '{:.1f}%',
                    'Estimated_Savings_$': '${:,.0f}'
                }),
                use_container_width=True
            )
            
            # Savings by category chart
            fig = px.bar(savings_df.head(10), 
                        x='Estimated_Savings_$', 
                        y='Item',
                        orientation='h',
                        title="Top 10 Items by Potential Savings",
                        labels={'Estimated_Savings_$': 'Potential Annual Savings ($)'})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Implementation roadmap
            st.subheader("🗺️ Implementation Roadmap")
            
            st.markdown("""
            **Phase 1 (Month 1-2): Quick Wins**
            - Focus on top 5 items with highest savings potential
            - Implement purchase timing recommendations
            - Set up price tracking alerts
            
            **Phase 2 (Month 3-6): Systematic Implementation**
            - Extend to top 20 items
            - Develop seasonal procurement calendar
            - Train procurement team on optimal timing
            
            **Phase 3 (Month 6+): Advanced Optimization**
            - Implement automated price monitoring
            - Develop predictive pricing models
            - Integrate with procurement planning systems
            """)
            
            # Export recommendations
            if st.button("📥 Export Seasonal Analysis"):
                csv = savings_df.to_csv(index=False)
                st.download_button(
                    label="Download Seasonal Analysis Report",
                    data=csv,
                    file_name=f"seasonal_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("Need more data points to calculate seasonal savings opportunities.")

if __name__ == "__main__":
    # For testing standalone
    st.set_page_config(page_title="Seasonal Price Optimization", layout="wide")
    
    # Sample data for testing
    sample_data = {
        'Creation Date': pd.date_range('2023-01-01', '2024-12-31', freq='W'),
        'Item': np.random.choice(['Widget A', 'Widget B', 'Widget C'], 104),
        'Unit Price': np.random.uniform(10, 100, 104),
        'Qty Delivered': np.random.randint(1, 50, 104)
    }
    df = pd.DataFrame(sample_data)
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    
    display(df)
