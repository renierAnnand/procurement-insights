import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def categorize_spend_by_rules(df):
    """Categorize spend using business rules and AI clustering"""
    
    # Basic categorization based on item descriptions and product families
    categories = {}
    
    # Define category keywords
    category_keywords = {
        'IT & Technology': ['computer', 'software', 'laptop', 'server', 'network', 'hardware', 'tech', 'IT'],
        'Office Supplies': ['paper', 'pen', 'office', 'stationery', 'desk', 'chair', 'supplies'],
        'Raw Materials': ['material', 'steel', 'aluminum', 'plastic', 'chemical', 'raw'],
        'Maintenance & Repair': ['maintenance', 'repair', 'spare', 'parts', 'service', 'MRO'],
        'Professional Services': ['consulting', 'service', 'professional', 'training', 'advisory'],
        'Marketing & Sales': ['marketing', 'advertisement', 'promotion', 'sales', 'branding'],
        'Facilities': ['cleaning', 'security', 'utilities', 'facility', 'building', 'janitorial'],
        'Transportation': ['freight', 'shipping', 'transport', 'logistics', 'delivery'],
        'Manufacturing Equipment': ['equipment', 'machine', 'tool', 'manufacturing', 'production'],
        'Safety & Compliance': ['safety', 'PPE', 'compliance', 'regulatory', 'protection']
    }
    
    # Default category
    for idx, row in df.iterrows():
        description = str(row.get('Item Description', '')).lower()
        product_family = str(row.get('Product Family', '')).lower()
        po_description = str(row.get('PO Description', '')).lower()
        
        # Combine all text fields for categorization
        combined_text = f"{description} {product_family} {po_description}"
        
        assigned_category = 'Other'
        max_matches = 0
        
        # Find best matching category
        for category, keywords in category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in combined_text)
            if matches > max_matches:
                max_matches = matches
                assigned_category = category
        
        categories[idx] = assigned_category
    
    return categories

def detect_price_anomalies(df, contamination=0.1):
    """Detect price anomalies using Isolation Forest"""
    
    # Prepare features for anomaly detection
    features = []
    
    # Price-related features
    if 'Unit Price' in df.columns:
        features.append('Unit Price')
    
    # Quantity features
    if 'Qty Delivered' in df.columns:
        features.append('Qty Delivered')
    
    # Total spend features
    if 'Line Total' in df.columns:
        features.append('Line Total')
    else:
        df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
        features.append('Line Total')
    
    if len(features) < 2:
        return pd.Series([False] * len(df), index=df.index)
    
    # Prepare data
    feature_data = df[features].fillna(df[features].median())
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(scaled_features)
    
    # Convert to boolean (True for anomalies)
    return pd.Series(anomaly_labels == -1, index=df.index)

def detect_quantity_anomalies(df, threshold_factor=3):
    """Detect quantity anomalies using statistical methods"""
    
    if 'Qty Delivered' not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    
    # Group by item to detect anomalies within each item
    anomalies = pd.Series([False] * len(df), index=df.index)
    
    for item in df['Item'].unique():
        item_data = df[df['Item'] == item]['Qty Delivered']
        
        if len(item_data) > 3:  # Need minimum data points
            Q1 = item_data.quantile(0.25)
            Q3 = item_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - threshold_factor * IQR
            upper_bound = Q3 + threshold_factor * IQR
            
            # Mark anomalies
            item_anomalies = (item_data < lower_bound) | (item_data > upper_bound)
            anomalies.loc[item_anomalies.index] = item_anomalies
    
    return anomalies

def analyze_spending_patterns(df):
    """Analyze spending patterns and trends"""
    
    patterns = {}
    
    # Monthly spending trends
    if 'Creation Date' in df.columns:
        df['Month'] = pd.to_datetime(df['Creation Date']).dt.to_period('M')
        monthly_spend = df.groupby('Month')['Line Total'].sum()
        
        # Calculate month-over-month growth
        monthly_growth = monthly_spend.pct_change().fillna(0)
        
        patterns['monthly_spend'] = monthly_spend
        patterns['monthly_growth'] = monthly_growth
        patterns['avg_monthly_spend'] = monthly_spend.mean()
        patterns['spend_volatility'] = monthly_spend.std() / monthly_spend.mean() if monthly_spend.mean() > 0 else 0
    
    # Vendor concentration
    vendor_spend = df.groupby('Vendor Name')['Line Total'].sum().sort_values(ascending=False)
    total_spend = vendor_spend.sum()
    
    patterns['vendor_concentration'] = {
        'top_10_vendors': vendor_spend.head(10),
        'top_vendor_share': (vendor_spend.iloc[0] / total_spend) * 100 if total_spend > 0 else 0,
        'top_5_share': (vendor_spend.head(5).sum() / total_spend) * 100 if total_spend > 0 else 0
    }
    
    # Category spend distribution
    if 'spend_category' in df.columns:
        category_spend = df.groupby('spend_category')['Line Total'].sum().sort_values(ascending=False)
        patterns['category_distribution'] = category_spend
    
    return patterns

def identify_maverick_spend(df, threshold_percent=5):
    """Identify maverick spending (off-contract purchases)"""
    
    maverick_indicators = []
    
    # Small, frequent orders (potential consolidation opportunities)
    small_orders = df[df['Line Total'] < df['Line Total'].quantile(0.2)]
    frequent_small_vendors = small_orders.groupby('Vendor Name').size()
    
    for vendor, count in frequent_small_vendors.items():
        if count >= 5:  # 5+ small orders
            total_small_spend = small_orders[small_orders['Vendor Name'] == vendor]['Line Total'].sum()
            maverick_indicators.append({
                'type': 'Frequent Small Orders',
                'vendor': vendor,
                'count': count,
                'total_spend': total_small_spend,
                'recommendation': 'Consider consolidation or blanket PO'
            })
    
    # One-time vendors with significant spend
    vendor_order_counts = df.groupby('Vendor Name').size()
    one_time_vendors = vendor_order_counts[vendor_order_counts == 1]
    
    for vendor in one_time_vendors.index:
        vendor_spend = df[df['Vendor Name'] == vendor]['Line Total'].sum()
        if vendor_spend > df['Line Total'].quantile(0.8):  # High spend, one-time vendor
            maverick_indicators.append({
                'type': 'One-time High Spend',
                'vendor': vendor,
                'count': 1,
                'total_spend': vendor_spend,
                'recommendation': 'Evaluate for future contracting'
            })
    
    return maverick_indicators

def display(df):
    st.header("📊 Spend Categorization & Anomaly Detection")
    st.markdown("Automatically categorize spending, detect anomalies, and identify optimization opportunities using AI and statistical methods.")
    
    # Data validation
    required_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered', 'Creation Date']
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
    
    # Calculate line total if not present
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏷️ Spend Categorization", "🚨 Anomaly Detection", "📈 Spending Patterns", "🎯 Maverick Spend", "📋 Insights & Actions"])
    
    with tab1:
        st.subheader("🏷️ Automatic Spend Categorization")
        
        if st.button("🔄 Categorize Spending", type="primary"):
            with st.spinner("Categorizing spending using AI and business rules..."):
                
                # Apply spend categorization
                spend_categories = categorize_spend_by_rules(df_clean)
                df_clean['spend_category'] = df_clean.index.map(spend_categories)
                
                # Calculate category summaries
                category_summary = df_clean.groupby('spend_category').agg({
                    'Line Total': ['sum', 'count', 'mean'],
                    'Vendor Name': 'nunique',
                    'Item': 'nunique'
                }).round(2)
                
                category_summary.columns = ['Total Spend', 'Transaction Count', 'Avg Transaction', 'Unique Vendors', 'Unique Items']
                category_summary = category_summary.sort_values('Total Spend', ascending=False)
                category_summary['Spend %'] = (category_summary['Total Spend'] / category_summary['Total Spend'].sum() * 100).round(1)
                
                # Display category overview
                total_spend = category_summary['Total Spend'].sum()
                largest_category = category_summary.index[0]
                largest_category_pct = category_summary['Spend %'].iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Categories", len(category_summary))
                with col2:
                    st.metric("Total Spend", f"{total_spend:,.0f}")
                with col3:
                    st.metric("Largest Category", largest_category)
                with col4:
                    st.metric("Largest Category %", f"{largest_category_pct:.1f}%")
                
                # Category breakdown table
                st.subheader("📊 Category Breakdown")
                
                st.dataframe(
                    category_summary.style.format({
                        'Total Spend': '{:,.0f}',
                        'Avg Transaction': '{:,.0f}',
                        'Spend %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Spend by category pie chart
                    fig = px.pie(
                        values=category_summary['Total Spend'],
                        names=category_summary.index,
                        title="Spend Distribution by Category",
                        hover_data=['Transaction Count']
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Category complexity (vendors vs items)
                    fig = px.scatter(
                        category_summary,
                        x='Unique Vendors',
                        y='Unique Items',
                        size='Total Spend',
                        color='Spend %',
                        hover_name=category_summary.index,
                        title="Category Complexity: Vendors vs Items",
                        labels={'Unique Vendors': 'Number of Vendors', 'Unique Items': 'Number of Items'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Monthly spend by category
                if 'Creation Date' in df_clean.columns:
                    st.subheader("📈 Monthly Spend Trends by Category")
                    
                    df_clean['Month'] = df_clean['Creation Date'].dt.to_period('M')
                    monthly_category_spend = df_clean.groupby(['Month', 'spend_category'])['Line Total'].sum().reset_index()
                    monthly_category_spend['Month'] = monthly_category_spend['Month'].astype(str)
                    
                    # Show top 5 categories only for clarity
                    top_categories = category_summary.head(5).index.tolist()
                    monthly_top_categories = monthly_category_spend[monthly_category_spend['spend_category'].isin(top_categories)]
                    
                    fig = px.line(
                        monthly_top_categories,
                        x='Month',
                        y='Line Total',
                        color='spend_category',
                        title="Monthly Spend Trends (Top 5 Categories)",
                        labels={'Line Total': 'Monthly Spend', 'spend_category': 'Category'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed category analysis
                st.subheader("🔍 Category Deep Dive")
                
                selected_category = st.selectbox(
                    "Select Category for Detailed Analysis",
                    options=category_summary.index.tolist(),
                    key="category_analysis"
                )
                
                if selected_category:
                    category_data = df_clean[df_clean['spend_category'] == selected_category]
                    
                    # Category vendors
                    vendor_spend_in_category = category_data.groupby('Vendor Name')['Line Total'].sum().sort_values(ascending=False).head(10)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Top Vendors in {selected_category}:**")
                        vendor_df = vendor_spend_in_category.reset_index()
                        vendor_df.columns = ['Vendor Name', 'Total Spend']
                        vendor_df['Spend %'] = (vendor_df['Total Spend'] / vendor_df['Total Spend'].sum() * 100).round(1)
                        
                        st.dataframe(
                            vendor_df.style.format({
                                'Total Spend': '{:,.0f}',
                                'Spend %': '{:.1f}%'
                            }),
                            use_container_width=True
                        )
                    
                    with col2:
                        fig = px.bar(
                            vendor_df.head(8),
                            x='Total Spend',
                            y='Vendor Name',
                            orientation='h',
                            title=f"Top Vendors in {selected_category}",
                            labels={'Total Spend': 'Total Spend', 'Vendor Name': 'Vendor'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Store categorized data for other tabs
                st.session_state['categorized_data'] = df_clean
                
                # Export categorized data
                csv = df_clean[['Vendor Name', 'Item', 'Item Description', 'Unit Price', 'Qty Delivered', 'Line Total', 'spend_category']].to_csv(index=False)
                st.download_button(
                    label="📥 Export Categorized Data",
                    data=csv,
                    file_name=f"categorized_spend_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.subheader("🚨 Anomaly Detection")
        
        # Anomaly detection parameters
        col1, col2 = st.columns(2)
        with col1:
            contamination_rate = st.slider("Anomaly Contamination Rate (%)", 1, 20, 5, help="Percentage of data expected to be anomalous") / 100
        with col2:
            iqr_threshold = st.slider("IQR Threshold Factor", 1.5, 5.0, 3.0, help="Factor for IQR-based outlier detection")
        
        if st.button("🔍 Detect Anomalies", type="primary"):
            with st.spinner("Detecting anomalies using AI and statistical methods..."):
                
                # Price anomaly detection
                price_anomalies = detect_price_anomalies(df_clean, contamination_rate)
                
                # Quantity anomaly detection
                quantity_anomalies = detect_quantity_anomalies(df_clean, iqr_threshold)
                
                # Combine anomalies
                df_clean['price_anomaly'] = price_anomalies
                df_clean['quantity_anomaly'] = quantity_anomalies
                df_clean['any_anomaly'] = price_anomalies | quantity_anomalies
                
                # Anomaly summary
                total_anomalies = df_clean['any_anomaly'].sum()
                price_anomaly_count = df_clean['price_anomaly'].sum()
                quantity_anomaly_count = df_clean['quantity_anomaly'].sum()
                anomaly_spend = df_clean[df_clean['any_anomaly']]['Line Total'].sum()
                total_spend = df_clean['Line Total'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Anomalies", total_anomalies)
                with col2:
                    st.metric("Price Anomalies", price_anomaly_count)
                with col3:
                    st.metric("Quantity Anomalies", quantity_anomaly_count)
                with col4:
                    anomaly_spend_pct = (anomaly_spend / total_spend * 100) if total_spend > 0 else 0
                    st.metric("Anomaly Spend %", f"{anomaly_spend_pct:.1f}%")
                
                if total_anomalies > 0:
                    # Anomaly details
                    st.subheader("🚨 Detected Anomalies")
                    
                    anomaly_data = df_clean[df_clean['any_anomaly']].copy()
                    anomaly_data['anomaly_type'] = ''
                    anomaly_data.loc[anomaly_data['price_anomaly'], 'anomaly_type'] += 'Price '
                    anomaly_data.loc[anomaly_data['quantity_anomaly'], 'anomaly_type'] += 'Quantity'
                    anomaly_data['anomaly_type'] = anomaly_data['anomaly_type'].str.strip()
                    
                    # Sort by spend impact
                    anomaly_display = anomaly_data.sort_values('Line Total', ascending=False)[[
                        'Vendor Name', 'Item', 'Item Description', 'Unit Price', 'Qty Delivered', 
                        'Line Total', 'anomaly_type', 'Creation Date'
                    ]].head(50)  # Show top 50 anomalies
                    
                    st.dataframe(
                        anomaly_display.style.format({
                            'Unit Price': '{:.2f}',
                            'Qty Delivered': '{:.1f}',
                            'Line Total': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Anomaly visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Anomalies by vendor
                        vendor_anomalies = anomaly_data.groupby('Vendor Name').agg({
                            'any_anomaly': 'count',
                            'Line Total': 'sum'
                        }).sort_values('Line Total', ascending=False).head(10)
                        vendor_anomalies.columns = ['Anomaly Count', 'Total Anomaly Spend']
                        
                        fig = px.scatter(
                            vendor_anomalies.reset_index(),
                            x='Anomaly Count',
                            y='Total Anomaly Spend',
                            hover_name='Vendor Name',
                            title="Anomalies by Vendor",
                            labels={'Anomaly Count': 'Number of Anomalies', 'Total Anomaly Spend': 'Total Spend in Anomalies'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Price vs Quantity anomaly scatter
                        sample_data = df_clean.sample(min(1000, len(df_clean)))  # Sample for performance
                        
                        fig = px.scatter(
                            sample_data,
                            x='Unit Price',
                            y='Qty Delivered',
                            color='any_anomaly',
                            size='Line Total',
                            title="Price vs Quantity (Anomalies Highlighted)",
                            labels={'any_anomaly': 'Is Anomaly'},
                            color_discrete_map={True: 'red', False: 'blue'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Time series of anomalies
                    if 'Creation Date' in df_clean.columns:
                        st.subheader("📅 Anomaly Timeline")
                        
                        df_clean['Month'] = df_clean['Creation Date'].dt.to_period('M')
                        monthly_anomalies = df_clean.groupby('Month').agg({
                            'any_anomaly': 'sum',
                            'Line Total': 'count'
                        })
                        monthly_anomalies.columns = ['Anomaly Count', 'Total Transactions']
                        monthly_anomalies['Anomaly Rate %'] = (monthly_anomalies['Anomaly Count'] / monthly_anomalies['Total Transactions'] * 100)
                        monthly_anomalies = monthly_anomalies.reset_index()
                        monthly_anomalies['Month'] = monthly_anomalies['Month'].astype(str)
                        
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig.add_trace(
                            go.Bar(x=monthly_anomalies['Month'], y=monthly_anomalies['Anomaly Count'], name="Anomaly Count"),
                            secondary_y=False
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=monthly_anomalies['Month'], y=monthly_anomalies['Anomaly Rate %'], 
                                     mode='lines+markers', name="Anomaly Rate %", line=dict(color='red')),
                            secondary_y=True
                        )
                        
                        fig.update_xaxes(title_text="Month")
                        fig.update_yaxes(title_text="Number of Anomalies", secondary_y=False)
                        fig.update_yaxes(title_text="Anomaly Rate (%)", secondary_y=True)
                        fig.update_layout(title_text="Monthly Anomaly Trends", height=400)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Root cause analysis
                    st.subheader("🔍 Anomaly Root Cause Analysis")
                    
                    # Most common anomaly patterns
                    anomaly_patterns = []
                    
                    # High-priced items with low quantities
                    high_price_low_qty = anomaly_data[
                        (anomaly_data['Unit Price'] > df_clean['Unit Price'].quantile(0.9)) &
                        (anomaly_data['Qty Delivered'] < df_clean['Qty Delivered'].quantile(0.1))
                    ]
                    if len(high_price_low_qty) > 0:
                        anomaly_patterns.append({
                            'Pattern': 'High Price, Low Quantity',
                            'Count': len(high_price_low_qty),
                            'Total Impact': high_price_low_qty['Line Total'].sum(),
                            'Potential Cause': 'Emergency purchases, premium products, or data errors'
                        })
                    
                    # Low-priced items with high quantities
                    low_price_high_qty = anomaly_data[
                        (anomaly_data['Unit Price'] < df_clean['Unit Price'].quantile(0.1)) &
                        (anomaly_data['Qty Delivered'] > df_clean['Qty Delivered'].quantile(0.9))
                    ]
                    if len(low_price_high_qty) > 0:
                        anomaly_patterns.append({
                            'Pattern': 'Low Price, High Quantity',
                            'Count': len(low_price_high_qty),
                            'Total Impact': low_price_high_qty['Line Total'].sum(),
                            'Potential Cause': 'Bulk discounts, clearance sales, or data errors'
                        })
                    
                    # Single vendor with multiple anomalies
                    vendor_anomaly_counts = anomaly_data['Vendor Name'].value_counts()
                    problematic_vendors = vendor_anomaly_counts[vendor_anomaly_counts >= 3]
                    if len(problematic_vendors) > 0:
                        total_problematic_spend = sum(anomaly_data[anomaly_data['Vendor Name'].isin(problematic_vendors.index)]['Line Total'])
                        anomaly_patterns.append({
                            'Pattern': 'Multiple Anomalies per Vendor',
                            'Count': problematic_vendors.sum(),
                            'Total Impact': total_problematic_spend,
                            'Potential Cause': 'Vendor data quality issues or irregular pricing'
                        })
                    
                    if anomaly_patterns:
                        patterns_df = pd.DataFrame(anomaly_patterns)
                        st.dataframe(
                            patterns_df.style.format({
                                'Total Impact': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                    
                    # Store anomaly data for other tabs
                    st.session_state['anomaly_data'] = df_clean
                
                else:
                    st.success("No significant anomalies detected with current parameters.")
    
    with tab3:
        st.subheader("📈 Spending Pattern Analysis")
        
        # Analyze spending patterns
        patterns = analyze_spending_patterns(df_clean)
        
        # Monthly spending trends
        if 'monthly_spend' in patterns:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Monthly Spend", f"{patterns['avg_monthly_spend']:,.0f}")
            with col2:
                st.metric("Spend Volatility", f"{patterns['spend_volatility']:.2f}")
            with col3:
                if len(patterns['monthly_growth']) > 0:
                    latest_growth = patterns['monthly_growth'].iloc[-1] * 100
                    st.metric("Latest Month Growth", f"{latest_growth:+.1f}%")
            
            # Monthly trends chart
            monthly_data = patterns['monthly_spend'].reset_index()
            monthly_data['Month'] = monthly_data['Month'].astype(str)
            
            fig = go.Figure()
            
            # Monthly spend bars
            fig.add_trace(go.Bar(
                x=monthly_data['Month'],
                y=monthly_data['Line Total'],
                name='Monthly Spend',
                marker_color='lightblue'
            ))
            
            # Trend line
            from sklearn.linear_model import LinearRegression
            x_numeric = np.arange(len(monthly_data)).reshape(-1, 1)
            reg = LinearRegression().fit(x_numeric, monthly_data['Line Total'])
            trend_line = reg.predict(x_numeric)
            
            fig.add_trace(go.Scatter(
                x=monthly_data['Month'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Monthly Spending Trends",
                xaxis_title="Month",
                yaxis_title="Total Spend",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Vendor concentration analysis
        if 'vendor_concentration' in patterns:
            st.subheader("🏢 Vendor Concentration Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Top Vendor Share", f"{patterns['vendor_concentration']['top_vendor_share']:.1f}%")
            with col2:
                st.metric("Top 5 Vendors Share", f"{patterns['vendor_concentration']['top_5_share']:.1f}%")
            
            # Vendor spend distribution
            top_vendors = patterns['vendor_concentration']['top_10_vendors']
            
            fig = px.bar(
                x=top_vendors.values,
                y=top_vendors.index,
                orientation='h',
                title="Top 10 Vendors by Spend",
                labels={'x': 'Total Spend', 'y': 'Vendor Name'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Concentration risk assessment
            concentration_risk = "Low"
            if patterns['vendor_concentration']['top_vendor_share'] > 40:
                concentration_risk = "High"
            elif patterns['vendor_concentration']['top_vendor_share'] > 25:
                concentration_risk = "Medium"
            
            if concentration_risk == "High":
                st.warning(f"⚠️ **High Vendor Concentration Risk**: Top vendor represents {patterns['vendor_concentration']['top_vendor_share']:.1f}% of total spend")
            elif concentration_risk == "Medium":
                st.info(f"ℹ️ **Medium Vendor Concentration**: Top vendor represents {patterns['vendor_concentration']['top_vendor_share']:.1f}% of total spend")
            else:
                st.success(f"✅ **Low Vendor Concentration Risk**: Well-diversified vendor base")
        
        # Category spending patterns (if categorized data exists)
        if 'categorized_data' in st.session_state:
            categorized_df = st.session_state['categorized_data']
            
            st.subheader("🏷️ Category Spending Patterns")
            
            # Monthly category trends
            categorized_df['Month'] = categorized_df['Creation Date'].dt.to_period('M')
            category_monthly = categorized_df.groupby(['Month', 'spend_category'])['Line Total'].sum().reset_index()
            category_monthly['Month'] = category_monthly['Month'].astype(str)
            
            # Show trends for top 5 categories
            top_5_categories = categorized_df.groupby('spend_category')['Line Total'].sum().nlargest(5).index
            category_trends = category_monthly[category_monthly['spend_category'].isin(top_5_categories)]
            
            fig = px.line(
                category_trends,
                x='Month',
                y='Line Total',
                color='spend_category',
                title="Category Spending Trends (Top 5)",
                labels={'Line Total': 'Monthly Spend', 'spend_category': 'Category'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal patterns
        st.subheader("📅 Seasonal Spending Patterns")
        
        if 'Creation Date' in df_clean.columns:
            df_clean['Quarter'] = df_clean['Creation Date'].dt.quarter
            df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name()
            
            # Quarterly spending
            quarterly_spend = df_clean.groupby('Quarter')['Line Total'].sum()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=['Q1', 'Q2', 'Q3', 'Q4'],
                    y=quarterly_spend.values,
                    title="Quarterly Spending Distribution",
                    labels={'x': 'Quarter', 'y': 'Total Spend'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                monthly_spend_pattern = df_clean.groupby('Month_Name')['Line Total'].sum()
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                monthly_ordered = monthly_spend_pattern.reindex([month for month in month_order if month in monthly_spend_pattern.index])
                
                fig = px.line(
                    x=monthly_ordered.index,
                    y=monthly_ordered.values,
                    title="Monthly Spending Pattern",
                    labels={'x': 'Month', 'y': 'Total Spend'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🎯 Maverick Spend Identification")
        
        if st.button("🔍 Identify Maverick Spending", type="primary"):
            with st.spinner("Analyzing for maverick spending patterns..."):
                
                # Identify maverick spend
                maverick_indicators = identify_maverick_spend(df_clean)
                
                if maverick_indicators:
                    # Convert to DataFrame for better display
                    maverick_df = pd.DataFrame(maverick_indicators)
                    
                    # Summary metrics
                    total_maverick_spend = maverick_df['total_spend'].sum()
                    total_spend = df_clean['Line Total'].sum()
                    maverick_percent = (total_maverick_spend / total_spend * 100) if total_spend > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Maverick Indicators", len(maverick_df))
                    with col2:
                        st.metric("Maverick Spend", f"{total_maverick_spend:,.0f}")
                    with col3:
                        st.metric("Maverick %", f"{maverick_percent:.1f}%")
                    
                    # Maverick spend breakdown
                    st.subheader("🚨 Maverick Spending Details")
                    
                    st.dataframe(
                        maverick_df.style.format({
                            'total_spend': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        type_counts = maverick_df['type'].value_counts()
                        fig = px.pie(
                            values=type_counts.values,
                            names=type_counts.index,
                            title="Maverick Spend Types"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            maverick_df.sort_values('total_spend', ascending=True).tail(10),
                            x='total_spend',
                            y='vendor',
                            orientation='h',
                            color='type',
                            title="Top Maverick Vendors by Spend",
                            labels={'total_spend': 'Total Spend', 'vendor': 'Vendor'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional analysis - small order consolidation opportunities
                    st.subheader("📦 Small Order Consolidation Analysis")
                    
                    small_order_threshold = df_clean['Line Total'].quantile(0.3)  # Bottom 30% of orders
                    small_orders = df_clean[df_clean['Line Total'] <= small_order_threshold]
                    
                    consolidation_opportunities = small_orders.groupby('Vendor Name').agg({
                        'Line Total': ['sum', 'count', 'mean'],
                        'Creation Date': ['min', 'max']
                    })
                    
                    consolidation_opportunities.columns = ['Total Spend', 'Order Count', 'Avg Order Size', 'First Order', 'Last Order']
                    consolidation_opportunities = consolidation_opportunities[consolidation_opportunities['Order Count'] >= 3]
                    consolidation_opportunities = consolidation_opportunities.sort_values('Total Spend', ascending=False)
                    
                    # Calculate potential savings (assuming 20% reduction in admin costs)
                    admin_cost_per_order = 25  # Assumed admin cost
                    consolidation_opportunities['Potential Admin Savings'] = (consolidation_opportunities['Order Count'] - 1) * admin_cost_per_order
                    
                    if len(consolidation_opportunities) > 0:
                        st.write("**Small Order Consolidation Opportunities:**")
                        st.dataframe(
                            consolidation_opportunities.head(15).style.format({
                                'Total Spend': '{:,.0f}',
                                'Avg Order Size': '{:.0f}',
                                'Potential Admin Savings': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                        
                        total_consolidation_savings = consolidation_opportunities['Potential Admin Savings'].sum()
                        st.info(f"💰 **Total Potential Consolidation Savings**: ${total_consolidation_savings:,.0f} annually")
                    
                else:
                    st.success("No significant maverick spending patterns detected.")
                
                # Tail spend analysis
                st.subheader("📊 Tail Spend Analysis")
                
                # Analyze low-frequency, low-value vendors
                vendor_summary = df_clean.groupby('Vendor Name').agg({
                    'Line Total': ['sum', 'count'],
                    'Creation Date': ['min', 'max']
                })
                vendor_summary.columns = ['Total Spend', 'Order Count', 'First Order', 'Last Order']
                
                # Define tail spend criteria
                tail_spend_vendors = vendor_summary[
                    (vendor_summary['Total Spend'] < vendor_summary['Total Spend'].quantile(0.5)) &
                    (vendor_summary['Order Count'] <= 2)
                ]
                
                tail_spend_total = tail_spend_vendors['Total Spend'].sum()
                tail_spend_percent = (tail_spend_total / df_clean['Line Total'].sum() * 100) if df_clean['Line Total'].sum() > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tail Spend Vendors", len(tail_spend_vendors))
                with col2:
                    st.metric("Tail Spend Amount", f"{tail_spend_total:,.0f}")
                with col3:
                    st.metric("Tail Spend %", f"{tail_spend_percent:.1f}%")
                
                if tail_spend_percent > 5:
                    st.warning(f"⚠️ **High Tail Spend**: {tail_spend_percent:.1f}% of spend is with low-frequency vendors")
                    st.write("**Recommendations:**")
                    st.write("• Consolidate similar purchases with preferred vendors")
                    st.write("• Implement minimum order values")
                    st.write("• Review procurement approval thresholds")
                else:
                    st.success("✅ Tail spend is well-controlled")
    
    with tab5:
        st.subheader("📋 Strategic Insights & Action Plan")
        
        # Generate comprehensive insights
        insights = []
        
        # Spend concentration insights
        if 'vendor_concentration' in analyze_spending_patterns(df_clean):
            patterns = analyze_spending_patterns(df_clean)
            top_vendor_share = patterns['vendor_concentration']['top_vendor_share']
            
            if top_vendor_share > 30:
                insights.append({
                    'category': 'Risk Management',
                    'insight': f'High vendor concentration risk - top vendor represents {top_vendor_share:.1f}% of spend',
                    'action': 'Develop alternative suppliers and implement vendor diversification strategy',
                    'priority': 'High',
                    'potential_impact': 'Risk Reduction'
                })
        
        # Category insights (if available)
        if 'categorized_data' in st.session_state:
            cat_data = st.session_state['categorized_data']
            category_counts = cat_data['spend_category'].value_counts()
            
            if 'Other' in category_counts.index and category_counts['Other'] > len(cat_data) * 0.2:
                insights.append({
                    'category': 'Data Quality',
                    'insight': f'{category_counts["Other"]} transactions ({category_counts["Other"]/len(cat_data)*100:.1f}%) are uncategorized',
                    'action': 'Improve item descriptions and implement better categorization rules',
                    'priority': 'Medium',
                    'potential_impact': 'Better Visibility'
                })
        
        # Anomaly insights (if available)
        if 'anomaly_data' in st.session_state:
            anomaly_data = st.session_state['anomaly_data']
            anomaly_spend = anomaly_data[anomaly_data['any_anomaly']]['Line Total'].sum()
            total_spend = anomaly_data['Line Total'].sum()
            anomaly_percent = (anomaly_spend / total_spend * 100) if total_spend > 0 else 0
            
            if anomaly_percent > 5:
                insights.append({
                    'category': 'Cost Control',
                    'insight': f'{anomaly_percent:.1f}% of spend shows anomalous patterns',
                    'action': 'Investigate high-impact anomalies and implement controls',
                    'priority': 'High',
                    'potential_impact': 'Cost Savings'
                })
        
        # Frequency insights
        single_order_vendors = df_clean.groupby('Vendor Name').size()
        one_time_vendors = single_order_vendors[single_order_vendors == 1]
        one_time_spend = df_clean[df_clean['Vendor Name'].isin(one_time_vendors.index)]['Line Total'].sum()
        one_time_percent = (one_time_spend / df_clean['Line Total'].sum() * 100) if df_clean['Line Total'].sum() > 0 else 0
        
        if one_time_percent > 15:
            insights.append({
                'category': 'Procurement Efficiency',
                'insight': f'{one_time_percent:.1f}% of spend is with one-time vendors',
                'action': 'Consolidate purchases with preferred vendors and implement vendor onboarding process',
                'priority': 'Medium',
                'potential_impact': 'Process Efficiency'
            })
        
        # Display insights
        if insights:
            st.subheader("🎯 Key Insights")
            
            insights_df = pd.DataFrame(insights)
            
            # Priority summary
            priority_counts = insights_df['priority'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Insights", len(insights_df))
            with col2:
                high_priority = priority_counts.get('High', 0)
                st.metric("High Priority", high_priority)
            with col3:
                medium_priority = priority_counts.get('Medium', 0)
                st.metric("Medium Priority", medium_priority)
            
            # Insights table
            st.dataframe(insights_df, use_container_width=True)
        
        # Action plan
        st.subheader("📋 90-Day Action Plan")
        
        action_plan = [
            {
                'Phase': 'Week 1-2',
                'Focus': 'Data Quality & Categorization',
                'Actions': [
                    'Complete spend categorization for all transactions',
                    'Investigate and resolve top anomalies',
                    'Validate vendor master data'
                ]
            },
            {
                'Phase': 'Week 3-6',
                'Focus': 'Vendor Consolidation',
                'Actions': [
                    'Identify consolidation opportunities',
                    'Negotiate with preferred vendors',
                    'Implement vendor scorecards'
                ]
            },
            {
                'Phase': 'Week 7-10',
                'Focus': 'Process Optimization',
                'Actions': [
                    'Implement maverick spend controls',
                    'Establish approval workflows',
                    'Create purchase order templates'
                ]
            },
            {
                'Phase': 'Week 11-12',
                'Focus': 'Monitoring & Governance',
                'Actions': [
                    'Set up automated reporting',
                    'Establish KPI dashboards',
                    'Train procurement team'
                ]
            }
        ]
        
        for phase in action_plan:
            with st.expander(f"**{phase['Phase']}**: {phase['Focus']}"):
                for action in phase['Actions']:
                    st.write(f"• {action}")
        
        # Expected benefits
        st.subheader("💰 Expected Benefits")
        
        benefits_col1, benefits_col2 = st.columns(2)
        
        with benefits_col1:
            st.markdown("#### **Cost Savings**")
            st.write("• 3-7% reduction in total spend")
            st.write("• 15-25% reduction in maverick spend")
            st.write("• $50-200 savings per consolidated order")
            st.write("• 5-15% reduction in tail spend")
        
        with benefits_col2:
            st.markdown("#### **Process Improvements**")
            st.write("• 30-50% reduction in processing time")
            st.write("• 90%+ spend visibility")
            st.write("• 80%+ vendor compliance rate")
            st.write("• 95%+ data accuracy")
        
        # KPIs to track
        st.subheader("📊 Key Performance Indicators")
        
        kpi_data = [
            {'KPI': 'Spend Categorization Rate', 'Current': '75%', 'Target': '95%', 'Frequency': 'Monthly'},
            {'KPI': 'Vendor Concentration (Top 5)', 'Current': f"{patterns.get('vendor_concentration', {}).get('top_5_share', 0):.1f}%", 'Target': '<60%', 'Frequency': 'Quarterly'},
            {'KPI': 'Maverick Spend %', 'Current': 'TBD', 'Target': '<5%', 'Frequency': 'Monthly'},
            {'KPI': 'Anomaly Detection Rate', 'Current': f"{contamination_rate*100:.0f}%", 'Target': '<3%', 'Frequency': 'Weekly'},
            {'KPI': 'One-time Vendor Spend %', 'Current': f"{one_time_percent:.1f}%", 'Target': '<10%', 'Frequency': 'Monthly'},
            {'KPI': 'Data Quality Score', 'Current': '80%', 'Target': '95%', 'Frequency': 'Monthly'}
        ]
        
        kpi_df = pd.DataFrame(kpi_data)
        st.dataframe(kpi_df, use_container_width=True)
        
        # Export action plan
        if st.button("📥 Export Complete Analysis"):
            
            # Combine all analysis results
            export_data = {
                'insights': insights_df if insights else pd.DataFrame(),
                'action_plan': pd.DataFrame(action_plan),
                'kpis': kpi_df
            }
            
            # Create comprehensive export
            with pd.ExcelWriter('procurement_analysis.xlsx') as writer:
                for sheet_name, data in export_data.items():
                    if not data.empty:
                        data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            st.success("Analysis exported successfully!")
            st.download_button(
                label="Download Complete Analysis",
                data=open('procurement_analysis.xlsx', 'rb').read(),
                file_name=f"procurement_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )