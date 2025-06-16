import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def display(df):
    """Contracting Opportunities Module - Simplified Version"""
    st.header("🤝 Contracting Opportunities")
    st.markdown("Identify and prioritize strategic contracting opportunities for cost savings.")
    
    # Basic data validation
    required_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("This module requires: Vendor Name, Unit Price, and Qty Delivered columns")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    # Calculate line total if missing
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Contract Identification", "📊 Vendor Analysis", "💰 Savings Potential"])
    
    with tab1:
        st.subheader("🎯 Contract Opportunity Identification")
        
        # Vendor spend analysis
        vendor_analysis = df_clean.groupby('Vendor Name').agg({
            'Line Total': ['sum', 'count', 'mean'],
            'Unit Price': ['std', 'mean'],
            'Item': 'nunique'
        }).round(2)
        
        vendor_analysis.columns = ['Total Spend', 'Transaction Count', 'Avg Transaction', 'Price Volatility', 'Avg Price', 'Unique Items']
        vendor_analysis['CV'] = (vendor_analysis['Price Volatility'] / vendor_analysis['Avg Price']).fillna(0)
        vendor_analysis = vendor_analysis.sort_values('Total Spend', ascending=False)
        
        # Contract suitability scoring
        def calculate_contract_score(row):
            score = 0
            # High spend (40% weight)
            if row['Total Spend'] > vendor_analysis['Total Spend'].quantile(0.8):
                score += 40
            elif row['Total Spend'] > vendor_analysis['Total Spend'].quantile(0.6):
                score += 25
            
            # Frequent transactions (30% weight)
            if row['Transaction Count'] > vendor_analysis['Transaction Count'].quantile(0.8):
                score += 30
            elif row['Transaction Count'] > vendor_analysis['Transaction Count'].quantile(0.6):
                score += 20
            
            # Multiple items (20% weight)  
            if row['Unique Items'] > vendor_analysis['Unique Items'].quantile(0.7):
                score += 20
            elif row['Unique Items'] > vendor_analysis['Unique Items'].quantile(0.5):
                score += 10
            
            # Price stability (10% weight)
            if row['CV'] < vendor_analysis['CV'].quantile(0.3):
                score += 10
            elif row['CV'] < vendor_analysis['CV'].quantile(0.6):
                score += 5
            
            return score
        
        vendor_analysis['Contract Score'] = vendor_analysis.apply(calculate_contract_score, axis=1)
        vendor_analysis['Priority'] = pd.cut(vendor_analysis['Contract Score'], 
                                           bins=[0, 50, 75, 100], 
                                           labels=['Low', 'Medium', 'High'])
        
        # Priority distribution
        priority_counts = vendor_analysis['Priority'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Vendors", len(vendor_analysis))
        with col2:
            st.metric("High Priority", priority_counts.get('High', 0))
        with col3:
            st.metric("Medium Priority", priority_counts.get('Medium', 0))
        with col4:
            st.metric("Low Priority", priority_counts.get('Low', 0))
        
        # Top contract opportunities
        st.subheader("🌟 Top Contract Opportunities")
        
        top_opportunities = vendor_analysis.head(15)[['Total Spend', 'Transaction Count', 'Unique Items', 'Contract Score', 'Priority']]
        
        st.dataframe(
            top_opportunities.style.format({
                'Total Spend': '${:,.0f}',
                'Transaction Count': '{:.0f}',
                'Contract Score': '{:.0f}'
            }),
            use_container_width=True
        )
        
        # Opportunity scatter plot
        fig = px.scatter(vendor_analysis, 
                        x='Transaction Count', 
                        y='Total Spend',
                        size='Unique Items',
                        color='Priority',
                        hover_name=vendor_analysis.index,
                        title="Contract Opportunities by Spend and Frequency",
                        labels={'Transaction Count': 'Number of Transactions', 'Total Spend': 'Total Annual Spend ($)'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Vendor Performance Analysis")
        
        # Vendor selection
        high_priority_vendors = vendor_analysis[vendor_analysis['Priority'] == 'High'].index.tolist()
        
        if high_priority_vendors:
            selected_vendor = st.selectbox("Select Vendor for Detailed Analysis", high_priority_vendors)
            
            if selected_vendor:
                vendor_data = df_clean[df_clean['Vendor Name'] == selected_vendor]
                
                # Vendor metrics
                total_spend = vendor_data['Line Total'].sum()
                transaction_count = len(vendor_data)
                unique_items = vendor_data['Item'].nunique()
                avg_order_value = vendor_data['Line Total'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Spend", f"${total_spend:,.0f}")
                with col2:
                    st.metric("Transactions", transaction_count)
                with col3:
                    st.metric("Unique Items", unique_items)
                with col4:
                    st.metric("Avg Order Value", f"${avg_order_value:,.0f}")
                
                # Monthly spend trend (if date available)
                if 'Creation Date' in vendor_data.columns:
                    vendor_data['Creation Date'] = pd.to_datetime(vendor_data['Creation Date'], errors='coerce')
                    monthly_spend = vendor_data.groupby(vendor_data['Creation Date'].dt.to_period('M'))['Line Total'].sum().reset_index()
                    monthly_spend['Creation Date'] = monthly_spend['Creation Date'].astype(str)
                    
                    fig = px.line(monthly_spend, x='Creation Date', y='Line Total',
                                 title=f"Monthly Spend Trend - {selected_vendor}",
                                 labels={'Line Total': 'Monthly Spend ($)', 'Creation Date': 'Month'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top items
                st.subheader(f"🔝 Top Items - {selected_vendor}")
                
                item_analysis = vendor_data.groupby('Item').agg({
                    'Line Total': 'sum',
                    'Unit Price': 'mean',
                    'Qty Delivered': 'sum'
                }).round(2)
                item_analysis = item_analysis.sort_values('Line Total', ascending=False).head(10)
                
                st.dataframe(
                    item_analysis.style.format({
                        'Line Total': '${:,.0f}',
                        'Unit Price': '${:.2f}',
                        'Qty Delivered': '{:,.0f}'
                    }),
                    use_container_width=True
                )
        else:
            st.info("No high-priority vendors identified. Adjust analysis parameters or check data.")
    
    with tab3:
        st.subheader("💰 Contract Savings Potential")
        
        # Savings assumptions
        col1, col2 = st.columns(2)
        with col1:
            expected_savings_rate = st.slider("Expected Contract Savings (%)", 2, 15, 8) / 100
        with col2:
            contract_implementation_cost = st.number_input("Implementation Cost per Contract ($)", 1000, 50000, 10000)
        
        # Calculate savings for high priority vendors
        high_priority_analysis = vendor_analysis[vendor_analysis['Priority'] == 'High'].copy()
        
        if len(high_priority_analysis) > 0:
            high_priority_analysis['Potential Savings'] = high_priority_analysis['Total Spend'] * expected_savings_rate
            high_priority_analysis['Net Savings'] = high_priority_analysis['Potential Savings'] - contract_implementation_cost
            high_priority_analysis['ROI'] = (high_priority_analysis['Net Savings'] / contract_implementation_cost * 100).round(1)
            
            # Filter positive ROI
            positive_roi = high_priority_analysis[high_priority_analysis['ROI'] > 0]
            
            if len(positive_roi) > 0:
                total_potential_savings = positive_roi['Potential Savings'].sum()
                total_net_savings = positive_roi['Net Savings'].sum()
                total_investment = len(positive_roi) * contract_implementation_cost
                portfolio_roi = (total_net_savings / total_investment * 100) if total_investment > 0 else 0
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Potential Annual Savings", f"${total_potential_savings:,.0f}")
                with col2:
                    st.metric("Net Annual Savings", f"${total_net_savings:,.0f}")
                with col3:
                    st.metric("Total Investment", f"${total_investment:,.0f}")
                with col4:
                    st.metric("Portfolio ROI", f"{portfolio_roi:.1f}%")
                
                # Contract prioritization
                st.subheader("🎯 Contract Implementation Priority")
                
                priority_display = positive_roi[['Total Spend', 'Potential Savings', 'Net Savings', 'ROI']].sort_values('ROI', ascending=False)
                
                st.dataframe(
                    priority_display.style.format({
                        'Total Spend': '${:,.0f}',
                        'Potential Savings': '${:,.0f}',
                        'Net Savings': '${:,.0f}',
                        'ROI': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # ROI visualization
                fig = px.bar(priority_display.reset_index().head(10), 
                            x='ROI', 
                            y='Vendor Name',
                            orientation='h',
                            title="Top 10 Contracts by ROI",
                            labels={'ROI': 'Return on Investment (%)'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Implementation roadmap
                st.subheader("🗺️ Implementation Roadmap")
                
                st.markdown(f"""
                **Phase 1 (Months 1-3): High ROI Contracts**
                - Target top {min(3, len(positive_roi))} vendors with highest ROI
                - Estimated savings: ${positive_roi.head(3)['Net Savings'].sum():,.0f}
                
                **Phase 2 (Months 4-8): Medium ROI Contracts**  
                - Implement contracts for remaining profitable opportunities
                - Total program savings: ${total_net_savings:,.0f}
                
                **Phase 3 (Months 9-12): Performance Management**
                - Monitor contract performance and compliance
                - Identify additional contracting opportunities
                """)
                
            else:
                st.warning("No vendors show positive ROI at current assumptions. Consider adjusting parameters.")
        else:
            st.info("No high-priority contracting opportunities identified.")

if __name__ == "__main__":
    st.set_page_config(page_title="Contracting Opportunities", layout="wide")
    
    # Sample data
    vendors = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']
    sample_data = {
        'Vendor Name': np.random.choice(vendors, 200),
        'Item': [f'Item {i}' for i in np.random.choice(range(1, 21), 200)],
        'Unit Price': np.random.uniform(10, 100, 200),
        'Qty Delivered': np.random.randint(1, 50, 200)
    }
    df = pd.DataFrame(sample_data)
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    display(df)
