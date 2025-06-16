import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def display(df):
    """Spend Categorization & Anomaly Detection Module - Simplified Version"""
    st.header("📊 Spend Categorization & Anomaly Detection")
    st.markdown("AI-powered spend categorization and anomaly detection for complete spend visibility.")
    
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
    tab1, tab2, tab3 = st.tabs(["🏷️ Spend Categorization", "🚨 Anomaly Detection", "📋 Insights"])
    
    with tab1:
        st.subheader("🏷️ Automatic Spend Categorization")
        
        # Simple categorization based on item descriptions
        def categorize_spend(row):
            item_desc = str(row.get('Item Description', '')).lower()
            item_name = str(row.get('Item', '')).lower()
            vendor_name = str(row.get('Vendor Name', '')).lower()
            
            combined_text = f"{item_desc} {item_name} {vendor_name}"
            
            # Category keywords
            if any(keyword in combined_text for keyword in ['computer', 'software', 'laptop', 'tech', 'it']):
                return 'IT & Technology'
            elif any(keyword in combined_text for keyword in ['office', 'paper', 'pen', 'supplies', 'stationery']):
                return 'Office Supplies'
            elif any(keyword in combined_text for keyword in ['material', 'steel', 'aluminum', 'raw']):
                return 'Raw Materials'
            elif any(keyword in combined_text for keyword in ['maintenance', 'repair', 'service']):
                return 'Maintenance & Repair'
            elif any(keyword in combined_text for keyword in ['consulting', 'professional', 'training']):
                return 'Professional Services'
            elif any(keyword in combined_text for keyword in ['marketing', 'advertisement', 'promotion']):
                return 'Marketing & Sales'
            elif any(keyword in combined_text for keyword in ['cleaning', 'security', 'facility']):
                return 'Facilities'
            elif any(keyword in combined_text for keyword in ['freight', 'shipping', 'transport']):
                return 'Transportation'
            else:
                return 'Other'
        
        if st.button("🔄 Categorize Spending", type="primary"):
            with st.spinner("Categorizing spending..."):
                # Apply categorization
                df_clean['Category'] = df_clean.apply(categorize_spend, axis=1)
                
                # Category summary
                category_summary = df_clean.groupby('Category').agg({
                    'Line Total': 'sum',
                    'Vendor Name': 'nunique',
                    'Item': 'nunique'
                }).round(2)
                category_summary.columns = ['Total Spend', 'Unique Vendors', 'Unique Items']
                category_summary['Spend %'] = (category_summary['Total Spend'] / category_summary['Total Spend'].sum() * 100).round(1)
                category_summary = category_summary.sort_values('Total Spend', ascending=False)
                
                # Display summary
                total_spend = category_summary['Total Spend'].sum()
                largest_category = category_summary.index[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Categories", len(category_summary))
                with col2:
                    st.metric("Total Spend", f"${total_spend:,.0f}")
                with col3:
                    st.metric("Largest Category", largest_category)
                
                # Category breakdown
                st.subheader("📊 Category Breakdown")
                
                st.dataframe(
                    category_summary.style.format({
                        'Total Spend': '${:,.0f}',
                        'Spend %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(values=category_summary['Total Spend'], 
                                names=category_summary.index,
                                title="Spend Distribution by Category")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(category_summary.reset_index(), 
                                x='Total Spend', 
                                y='Category',
                                orientation='h',
                                title="Spend by Category")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store categorized data
                st.session_state['categorized_data'] = df_clean
    
    with tab2:
        st.subheader("🚨 Anomaly Detection")
        
        # Parameters
        contamination_rate = st.slider("Anomaly Detection Sensitivity (%)", 1, 10, 5) / 100
        
        if st.button("🔍 Detect Anomalies", type="primary"):
            with st.spinner("Detecting anomalies using AI..."):
                
                # Prepare features for anomaly detection
                features = ['Unit Price', 'Qty Delivered', 'Line Total']
                feature_data = df_clean[features].fillna(df_clean[features].median())
                
                # Scale features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(feature_data)
                
                # Apply Isolation Forest
                iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                anomaly_labels = iso_forest.fit_predict(scaled_features)
                
                # Add anomaly labels
                df_clean['Is_Anomaly'] = anomaly_labels == -1
                
                # Anomaly summary
                total_anomalies = df_clean['Is_Anomaly'].sum()
                anomaly_spend = df_clean[df_clean['Is_Anomaly']]['Line Total'].sum()
                total_spend = df_clean['Line Total'].sum()
                anomaly_percent = (anomaly_spend / total_spend * 100) if total_spend > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Anomalies Detected", total_anomalies)
                with col2:
                    st.metric("Anomaly Spend", f"${anomaly_spend:,.0f}")
                with col3:
                    st.metric("Anomaly %", f"{anomaly_percent:.1f}%")
                
                if total_anomalies > 0:
                    # Anomaly details
                    st.subheader("🚨 Detected Anomalies")
                    
                    anomaly_data = df_clean[df_clean['Is_Anomaly']].sort_values('Line Total', ascending=False)
                    display_cols = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Line Total']
                    
                    # Show available columns only
                    available_cols = [col for col in display_cols if col in anomaly_data.columns]
                    
                    st.dataframe(
                        anomaly_data[available_cols].head(20).style.format({
                            'Unit Price': '${:.2f}',
                            'Qty Delivered': '{:.1f}',
                            'Line Total': '${:,.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Anomaly visualization
                    fig = px.scatter(df_clean, 
                                    x='Unit Price', 
                                    y='Qty Delivered',
                                    color='Is_Anomaly',
                                    size='Line Total',
                                    title="Anomaly Detection: Price vs Quantity",
                                    labels={'Is_Anomaly': 'Anomaly'},
                                    color_discrete_map={True: 'red', False: 'blue'})
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Anomaly patterns
                    st.subheader("🔍 Anomaly Patterns")
                    
                    # Price anomalies
                    high_price_anomalies = anomaly_data[anomaly_data['Unit Price'] > df_clean['Unit Price'].quantile(0.95)]
                    if len(high_price_anomalies) > 0:
                        st.write(f"**High Price Anomalies:** {len(high_price_anomalies)} transactions with unusually high unit prices")
                    
                    # Quantity anomalies  
                    high_qty_anomalies = anomaly_data[anomaly_data['Qty Delivered'] > df_clean['Qty Delivered'].quantile(0.95)]
                    if len(high_qty_anomalies) > 0:
                        st.write(f"**High Quantity Anomalies:** {len(high_qty_anomalies)} transactions with unusually high quantities")
                    
                    # Vendor anomalies
                    vendor_anomaly_counts = anomaly_data['Vendor Name'].value_counts()
                    if len(vendor_anomaly_counts) > 0:
                        st.write("**Vendors with Most Anomalies:**")
                        st.dataframe(vendor_anomaly_counts.head(10).reset_index(), use_container_width=True)
                
                else:
                    st.success("No significant anomalies detected with current sensitivity settings.")
    
    with tab3:
        st.subheader("📋 Key Insights & Recommendations")
        
        # Data quality insights
        st.subheader("📊 Data Quality Summary")
        
        data_quality_metrics = {
            'Total Records': len(df_clean),
            'Complete Records': len(df_clean.dropna()),
            'Unique Vendors': df_clean['Vendor Name'].nunique(),
            'Unique Items': df_clean['Item'].nunique() if 'Item' in df_clean.columns else 'N/A',
            'Date Range': f"{df_clean['Creation Date'].min().date()} to {df_clean['Creation Date'].max().date()}" if 'Creation Date' in df_clean.columns else 'N/A'
        }
        
        for metric, value in data_quality_metrics.items():
            st.write(f"**{metric}:** {value}")
        
        # Spending insights
        st.subheader("💡 Spending Insights")
        
        # Vendor concentration
        vendor_spend = df_clean.groupby('Vendor Name')['Line Total'].sum().sort_values(ascending=False)
        top_5_share = (vendor_spend.head(5).sum() / vendor_spend.sum() * 100)
        
        insights = [
            f"Top 5 vendors represent {top_5_share:.1f}% of total spend",
            f"Average order value: ${df_clean['Line Total'].mean():,.0f}",
            f"Median order value: ${df_clean['Line Total'].median():,.0f}",
            f"Price range: ${df_clean['Unit Price'].min():.2f} - ${df_clean['Unit Price'].max():,.2f}"
        ]
        
        for insight in insights:
            st.write(f"• {insight}")
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        
        recommendations = [
            "**Vendor Consolidation**: Consider consolidating purchases with top-performing vendors",
            "**Contract Negotiations**: Focus on high-spend vendors for better pricing",
            "**Process Improvements**: Investigate anomalies for potential process issues",
            "**Data Quality**: Improve item descriptions for better categorization",
            "**Regular Monitoring**: Set up automated anomaly detection alerts"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Action items
        st.subheader("✅ Next Steps")
        
        st.markdown("""
        **Week 1-2:**
        - Review and validate detected anomalies
        - Improve data quality and categorization
        
        **Week 3-4:**
        - Implement vendor consolidation opportunities
        - Set up regular monitoring dashboards
        
        **Month 2:**
        - Negotiate contracts with top vendors
        - Establish procurement KPIs and alerts
        """)

if __name__ == "__main__":
    st.set_page_config(page_title="Spend Categorization & Anomaly Detection", layout="wide")
    
    # Sample data
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D']
    items = ['Widget', 'Gadget', 'Tool', 'Supply', 'Equipment']
    
    sample_data = {
        'Vendor Name': np.random.choice(vendors, 150),
        'Item': np.random.choice(items, 150),
        'Unit Price': np.random.uniform(5, 200, 150),
        'Qty Delivered': np.random.randint(1, 100, 150)
    }
    df = pd.DataFrame(sample_data)
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    display(df)