import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import datetime
import warnings
warnings.filterwarnings('ignore')

# Global regional currency configuration
REGION_CONFIG = {
    'North America': {
        'countries': ['USA', 'Canada'],
        'primary_currency': 'USD',
        'currencies': ['USD', 'CAD'],
        'timezone': 'America/New_York',
        'flag': 'NA'
    },
    'Latin America': {
        'countries': ['Mexico', 'Colombia', 'Brazil', 'Argentina', 'Peru', 'Chile'],
        'primary_currency': 'USD',
        'currencies': ['MXN', 'COP', 'BRL', 'ARS', 'PEN', 'CLP', 'USD'],
        'timezone': 'America/Mexico_City',
        'flag': 'LATAM'
    },
    'Europe': {
        'countries': ['Germany', 'France', 'Spain', 'Italy', 'UK', 'Netherlands'],
        'primary_currency': 'EUR',
        'currencies': ['EUR', 'GBP', 'CHF'],
        'timezone': 'Europe/London',
        'flag': 'EU'
    },
    'Middle East': {
        'countries': ['UAE', 'Saudi Arabia', 'Qatar', 'Kuwait', 'Bahrain', 'Oman'],
        'primary_currency': 'AED',
        'currencies': ['AED', 'SAR', 'QAR', 'KWD', 'BHD', 'OMR'],
        'timezone': 'Asia/Dubai',
        'flag': 'ME'
    },
    'Africa': {
        'countries': ['South Africa', 'Nigeria', 'Egypt', 'Kenya', 'Ghana', 'Morocco'],
        'primary_currency': 'ZAR',
        'currencies': ['ZAR', 'NGN', 'EGP', 'KES', 'GHS', 'MAD'],
        'timezone': 'Africa/Johannesburg',
        'flag': 'AF'
    },
    'Asia Pacific': {
        'countries': ['Japan', 'China', 'India', 'Australia', 'Singapore', 'South Korea'],
        'primary_currency': 'USD',
        'currencies': ['JPY', 'CNY', 'INR', 'AUD', 'SGD', 'KRW'],
        'timezone': 'Asia/Tokyo',
        'flag': 'APAC'
    }
}

# Global currency configuration
CURRENCY_CONFIG = {
    # Americas
    'USD': {'symbol': '$', 'name': 'US Dollar'},
    'CAD': {'symbol': 'C$', 'name': 'Canadian Dollar'},
    'MXN': {'symbol': '$', 'name': 'Mexican Peso'},
    'COP': {'symbol': '$', 'name': 'Colombian Peso'},
    'BRL': {'symbol': 'R$', 'name': 'Brazilian Real'},
    'ARS': {'symbol': '$', 'name': 'Argentine Peso'},
    'PEN': {'symbol': 'S/', 'name': 'Peruvian Sol'},
    'CLP': {'symbol': '$', 'name': 'Chilean Peso'},
    
    # Europe
    'EUR': {'symbol': 'EUR', 'name': 'Euro'},
    'GBP': {'symbol': 'GBP', 'name': 'British Pound'},
    'CHF': {'symbol': 'CHF', 'name': 'Swiss Franc'},
    
    # Middle East
    'AED': {'symbol': 'AED', 'name': 'UAE Dirham'},
    'SAR': {'symbol': 'SAR', 'name': 'Saudi Riyal'},
    'QAR': {'symbol': 'QAR', 'name': 'Qatari Riyal'},
    'KWD': {'symbol': 'KWD', 'name': 'Kuwaiti Dinar'},
    'BHD': {'symbol': 'BHD', 'name': 'Bahraini Dinar'},
    'OMR': {'symbol': 'OMR', 'name': 'Omani Rial'},
    
    # Africa
    'ZAR': {'symbol': 'R', 'name': 'South African Rand'},
    'NGN': {'symbol': 'NGN', 'name': 'Nigerian Naira'},
    'EGP': {'symbol': 'EGP', 'name': 'Egyptian Pound'},
    'KES': {'symbol': 'KSh', 'name': 'Kenyan Shilling'},
    'GHS': {'symbol': 'GHS', 'name': 'Ghanaian Cedi'},
    'MAD': {'symbol': 'MAD', 'name': 'Moroccan Dirham'},
    
    # Asia Pacific
    'JPY': {'symbol': 'JPY', 'name': 'Japanese Yen'},
    'CNY': {'symbol': 'CNY', 'name': 'Chinese Yuan'},
    'INR': {'symbol': 'INR', 'name': 'Indian Rupee'},
    'AUD': {'symbol': 'A$', 'name': 'Australian Dollar'},
    'SGD': {'symbol': 'S$', 'name': 'Singapore Dollar'},
    'KRW': {'symbol': 'KRW', 'name': 'South Korean Won'},
}

def detect_region_from_data(df):
    """Detect region based on vendor names and data patterns"""
    vendor_names = ' '.join(df['Vendor Name'].astype(str).str.lower())
    
    # Regional vendor indicators
    regional_indicators = {
        'Latin America': ['colombia', 'sas', 'ltda', 'bogota', 'brasil', 'mexico', 'argentina'],
        'Europe': ['gmbh', 'sarl', 'ltd', 'plc', 'spa', 'bv', 'ag', 'london', 'paris'],
        'North America': ['corp', 'inc', 'llc', 'co', 'corporation', 'canada', 'toronto'],
        'Middle East': ['dubai', 'abu dhabi', 'saudi', 'qatar', 'kuwait', 'emirates', 'fze', 'fzco'],
        'Africa': ['south africa', 'nigeria', 'egypt', 'kenya', 'ghana', 'pty', 'proprietary'],
        'Asia Pacific': ['co.ltd', 'pte', 'singapore', 'tokyo', 'mumbai', 'private limited']
    }
    
    # Score each region
    region_scores = {}
    for region, indicators in regional_indicators.items():
        score = sum(1 for indicator in indicators if indicator in vendor_names)
        region_scores[region] = score
    
    # Find best match
    if region_scores:
        best_region = max(region_scores.items(), key=lambda x: x[1])
        if best_region[1] > 0:
            return best_region[0]
    
    return 'North America'  # Default

def format_currency(value, currency='USD', show_decimals=True):
    """Format currency with proper formatting"""
    try:
        config = CURRENCY_CONFIG.get(currency, CURRENCY_CONFIG['USD'])
        symbol = config['symbol']
        
        if currency in ['JPY', 'COP', 'CLP', 'KRW'] and not show_decimals:
            return f"{symbol} {value:,.0f}"
        elif show_decimals:
            return f"{symbol} {value:,.2f}"
        else:
            return f"{symbol} {value:,.0f}"
    except:
        return f"${value:,.2f}"

def get_currency_multiplier(currency):
    """Get multiplier for currency display"""
    high_value_currencies = ['COP', 'JPY', 'CLP', 'KRW', 'NGN']
    if currency in high_value_currencies:
        return 1000
    return 1

def ml_categorize_spending(df, n_clusters=8):
    """ML-based spend categorization using TF-IDF and K-Means"""
    
    # Prepare text features
    text_features = []
    for _, row in df.iterrows():
        vendor = str(row.get('Vendor Name', ''))
        item = str(row.get('Item', ''))
        desc = str(row.get('Item Description', ''))
        combined_text = f"{vendor} {item} {desc}".lower()
        text_features.append(combined_text)
    
    try:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(text_features)
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Create category names
        feature_names = vectorizer.get_feature_names_out()
        category_names = []
        
        for i in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[i]
            top_indices = cluster_center.argsort()[-3:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            category_name = create_category_name(top_terms)
            category_names.append(category_name)
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['ML_Category'] = [category_names[label] for label in cluster_labels]
        df_result['Cluster_ID'] = cluster_labels
        
        return df_result, vectorizer, kmeans, category_names
        
    except Exception as e:
        st.error(f"ML Categorization failed: {str(e)}")
        df_result = df.copy()
        df_result['ML_Category'] = df_result.apply(rule_based_categorize, axis=1)
        return df_result, None, None, []

def create_category_name(top_terms):
    """Create meaningful category names from top terms"""
    term_to_category = {
        'energy': 'Energy and Utilities',
        'siemens': 'Industrial Equipment',
        'service': 'Professional Services',
        'field': 'Field Services',
        'equipment': 'Equipment and Machinery',
        'material': 'Raw Materials',
        'office': 'Office Supplies',
        'transport': 'Transportation',
        'maintenance': 'Maintenance and Repair',
        'software': 'IT and Technology',
        'consulting': 'Consulting Services'
    }
    
    for term in top_terms:
        for keyword, category in term_to_category.items():
            if keyword in term.lower():
                return category
    
    if top_terms:
        return f"Category: {top_terms[0].title()}"
    
    return "Miscellaneous"

def rule_based_categorize(row):
    """Fallback rule-based categorization"""
    item_desc = str(row.get('Item Description', '')).lower()
    item_name = str(row.get('Item', '')).lower()
    vendor_name = str(row.get('Vendor Name', '')).lower()
    
    combined_text = f"{item_desc} {item_name} {vendor_name}"
    
    categories = {
        'Energy and Utilities': ['energy', 'electricity', 'gas', 'fuel', 'utility', 'power'],
        'Industrial Equipment': ['siemens', 'equipment', 'machinery', 'industrial', 'motor'],
        'Professional Services': ['consulting', 'professional', 'training', 'legal', 'audit', 'service'],
        'Field Services': ['field', 'exp-cos', 'installation', 'maintenance'],
        'Transportation': ['freight', 'shipping', 'transport', 'logistics', 'delivery', 'vehicle'],
        'IT and Technology': ['computer', 'software', 'laptop', 'tech', 'it', 'hardware'],
        'Office Supplies': ['office', 'paper', 'pen', 'supplies', 'stationery'],
        'Raw Materials': ['material', 'steel', 'aluminum', 'raw', 'metal', 'chemical']
    }
    
    for category, keywords in categories.items():
        if any(keyword in combined_text for keyword in keywords):
            return category
    
    return 'Other'

def create_vendor_multiselect(vendors, key):
    """Create vendor multiselect with Select All/Deselect All functionality"""
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write("**Select Vendors:**")
    
    with col2:
        select_all = st.button("Select All", key=f"select_all_{key}")
    
    with col3:
        deselect_all = st.button("Deselect All", key=f"deselect_all_{key}")
    
    # Handle select/deselect all
    if select_all:
        st.session_state[f"selected_vendors_{key}"] = vendors
    elif deselect_all:
        st.session_state[f"selected_vendors_{key}"] = []
    
    # Initialize session state
    if f"selected_vendors_{key}" not in st.session_state:
        st.session_state[f"selected_vendors_{key}"] = vendors[:10]
    
    selected_vendors = st.multiselect(
        "Vendors",
        options=vendors,
        default=st.session_state[f"selected_vendors_{key}"],
        key=f"multiselect_{key}",
        label_visibility="collapsed"
    )
    
    # Update session state
    st.session_state[f"selected_vendors_{key}"] = selected_vendors
    
    return selected_vendors

def analyze_anomaly_trends(df, date_column='Creation Date'):
    """Analyze anomaly trends over time"""
    if date_column not in df.columns or 'Is_Anomaly' not in df.columns:
        return pd.DataFrame()
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by month
    df['Year_Month'] = df[date_column].dt.to_period('M')
    
    # Calculate anomaly metrics by month
    monthly_trends = df.groupby('Year_Month').agg({
        'Is_Anomaly': ['sum', 'count'],
        'Line Total': ['sum', 'mean']
    }).round(2)
    
    # Flatten column names
    monthly_trends.columns = ['Anomalies', 'Total_Transactions', 'Total_Spend', 'Avg_Spend']
    monthly_trends['Anomaly_Rate'] = (monthly_trends['Anomalies'] / monthly_trends['Total_Transactions'] * 100).round(2)
    monthly_trends['Anomaly_Spend'] = df[df['Is_Anomaly']].groupby('Year_Month')['Line Total'].sum()
    monthly_trends['Anomaly_Spend'] = monthly_trends['Anomaly_Spend'].fillna(0)
    
    # Reset index
    monthly_trends = monthly_trends.reset_index()
    monthly_trends['Date'] = monthly_trends['Year_Month'].dt.to_timestamp()
    
    return monthly_trends

def display(df):
    """Enhanced Spend Categorization and Anomaly Detection Module"""
    st.header("Advanced Spend Analytics and Anomaly Detection")
    st.markdown("AI-powered spend categorization with ML clustering, regional filtering, and time-series anomaly analysis.")
    
    # Regional Configuration Sidebar
    with st.sidebar:
        st.subheader("Regional Settings")
        
        # Auto-detect region
        detected_region = detect_region_from_data(df)
        
        # Region selection
        region_options = list(REGION_CONFIG.keys())
        region_idx = region_options.index(detected_region) if detected_region in region_options else 0
        selected_region = st.selectbox(
            "Select Region",
            region_options,
            index=region_idx,
            format_func=lambda x: f"[{REGION_CONFIG[x]['flag']}] {x}"
        )
        
        # Get region config
        region_config = REGION_CONFIG[selected_region]
        primary_currency = region_config['primary_currency']
        available_currencies = region_config['currencies']
        
        # Currency selection
        selected_currency = st.selectbox(
            "Currency",
            available_currencies,
            index=0 if primary_currency not in available_currencies else available_currencies.index(primary_currency)
        )
        
        # Currency display options
        show_decimals = st.checkbox("Show Decimals", value=True)
        currency_multiplier = get_currency_multiplier(selected_currency)
        
        if currency_multiplier > 1:
            scale_large_numbers = st.checkbox(
                f"Scale Large Numbers (divide by {currency_multiplier:,.0f})", 
                value=False
            )
        else:
            scale_large_numbers = False
        
        st.info(f"Detected: {detected_region}")
        st.info(f"Countries: {', '.join(region_config['countries'][:3])}")
    
    # Data validation and cleaning
    required_columns = ['Vendor Name', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean and prepare data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Apply currency scaling
    if scale_large_numbers:
        df_clean['Unit Price'] = df_clean['Unit Price'] / currency_multiplier
        df_clean['Line Total'] = df_clean['Line Total'] / currency_multiplier
        currency_suffix = f" (divided by {currency_multiplier:,.0f})"
    else:
        currency_suffix = ""
    
    # Currency formatter
    def format_amount(value, decimals=None):
        if decimals is None:
            decimals = show_decimals
        return format_currency(value, selected_currency, decimals)
    
    # Vendor filtering
    st.subheader("Vendor Filtering")
    all_vendors = sorted(df_clean['Vendor Name'].unique())
    selected_vendors = create_vendor_multiselect(all_vendors, "main")
    
    # Filter data by selected vendors
    if selected_vendors:
        df_filtered = df_clean[df_clean['Vendor Name'].isin(selected_vendors)]
    else:
        df_filtered = df_clean
        st.warning("No vendors selected. Showing all data.")
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Region", selected_region)
    with col2:
        st.metric("Currency", f"{selected_currency} {CURRENCY_CONFIG[selected_currency]['symbol']}")
    with col3:
        st.metric("Records", f"{len(df_filtered):,}")
    with col4:
        st.metric("Total Spend", format_amount(df_filtered['Line Total'].sum(), False))
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["ML Categorization", "Anomaly Detection", "Anomaly Trends", "Insights"])
    
    with tab1:
        st.subheader(f"Machine Learning-Based Categorization {currency_suffix}")
        
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Categories", 3, 15, 8)
        with col2:
            use_ml = st.checkbox("Use ML Clustering", value=True)
        
        if st.button("Analyze and Categorize", type="primary"):
            with st.spinner("Running ML categorization..."):
                
                if use_ml:
                    df_categorized, vectorizer, kmeans, category_names = ml_categorize_spending(df_filtered, n_clusters)
                    category_column = 'ML_Category'
                    st.success("ML categorization completed!")
                else:
                    df_categorized = df_filtered.copy()
                    df_categorized['ML_Category'] = df_categorized.apply(rule_based_categorize, axis=1)
                    category_column = 'ML_Category'
                    st.success("Rule-based categorization completed!")
                
                # Category analysis
                category_summary = df_categorized.groupby(category_column).agg({
                    'Line Total': 'sum',
                    'Vendor Name': 'nunique',
                    'Item': 'nunique' if 'Item' in df_categorized.columns else 'size'
                }).round(2)
                category_summary.columns = ['Total Spend', 'Unique Vendors', 'Unique Items']
                category_summary['Spend Percent'] = (category_summary['Total Spend'] / category_summary['Total Spend'].sum() * 100).round(1)
                category_summary = category_summary.sort_values('Total Spend', ascending=False)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Category Performance")
                    
                    # Format for display
                    category_display = category_summary.copy()
                    category_display['Total Spend'] = category_display['Total Spend'].apply(lambda x: format_amount(x, False))
                    category_display['Spend Percent'] = category_display['Spend Percent'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(category_display, use_container_width=True)
                
                with col2:
                    # Category visualization
                    fig = px.pie(
                        values=category_summary['Total Spend'], 
                        names=category_summary.index,
                        title=f"Spend Distribution by Category{currency_suffix}"
                    )
                    fig.update_traces(textinfo='label+percent')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store categorized data
                st.session_state['categorized_data'] = df_categorized
    
    with tab2:
        st.subheader(f"Regional Anomaly Detection {currency_suffix}")
        
        # Anomaly detection parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            contamination_rate = st.slider("Sensitivity (%)", 1, 10, 5) / 100
        with col2:
            min_amount = st.number_input(
                f"Min Amount ({selected_currency})", 
                min_value=0.0, 
                value=1000.0 if not scale_large_numbers else 1.0
            )
        with col3:
            date_range_filter = st.checkbox("Filter by Date Range", value=False)
        
        # Date range selection
        if date_range_filter and 'Creation Date' in df_filtered.columns:
            df_filtered['Creation Date'] = pd.to_datetime(df_filtered['Creation Date'])
            min_date = df_filtered['Creation Date'].min().date()
            max_date = df_filtered['Creation Date'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
            
            # Filter by date range
            df_filtered = df_filtered[
                (df_filtered['Creation Date'].dt.date >= start_date) &
                (df_filtered['Creation Date'].dt.date <= end_date)
            ]
        
        if st.button("Detect Regional Anomalies", type="primary"):
            with st.spinner("Detecting anomalies within region..."):
                
                # Filter by minimum amount and region
                df_anomaly = df_filtered[df_filtered['Line Total'] >= min_amount].copy()
                
                if len(df_anomaly) == 0:
                    st.warning("No data above minimum threshold for anomaly detection.")
                else:
                    # Prepare features
                    features = ['Unit Price', 'Qty Delivered', 'Line Total']
                    feature_data = df_anomaly[features].fillna(df_anomaly[features].median())
                    
                    # Scale features
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(feature_data)
                    
                    # Anomaly detection
                    iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(scaled_features)
                    df_anomaly['Is_Anomaly'] = anomaly_labels == -1
                    
                    # Results
                    total_anomalies = df_anomaly['Is_Anomaly'].sum()
                    anomaly_spend = df_anomaly[df_anomaly['Is_Anomaly']]['Line Total'].sum()
                    total_spend = df_anomaly['Line Total'].sum()
                    anomaly_percent = (anomaly_spend / total_spend * 100) if total_spend > 0 else 0
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Anomalies Found", total_anomalies)
                    with col2:
                        st.metric("Anomaly Spend", format_amount(anomaly_spend, False))
                    with col3:
                        st.metric("Anomaly Rate", f"{anomaly_percent:.1f}%")
                    with col4:
                        st.metric("Region", selected_region)
                    
                    if total_anomalies > 0:
                        # Anomaly details
                        st.subheader("Detected Anomalies")
                        
                        anomaly_data = df_anomaly[df_anomaly['Is_Anomaly']].sort_values('Line Total', ascending=False)
                        
                        # Display top anomalies
                        display_cols = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Line Total']
                        available_cols = [col for col in display_cols if col in anomaly_data.columns]
                        
                        anomaly_display = anomaly_data[available_cols].head(20).copy()
                        if 'Unit Price' in anomaly_display.columns:
                            anomaly_display['Unit Price'] = anomaly_display['Unit Price'].apply(format_amount)
                        if 'Line Total' in anomaly_display.columns:
                            anomaly_display['Line Total'] = anomaly_display['Line Total'].apply(lambda x: format_amount(x, False))
                        
                        st.dataframe(anomaly_display, use_container_width=True)
                        
                        # Anomaly visualization
                        fig = px.scatter(
                            df_anomaly, 
                            x='Unit Price', 
                            y='Qty Delivered',
                            color='Is_Anomaly',
                            size='Line Total',
                            title=f"Anomaly Detection: {selected_region} Region{currency_suffix}",
                            labels={'Is_Anomaly': 'Anomaly'},
                            color_discrete_map={True: 'red', False: 'blue'},
                            hover_data=['Vendor Name']
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store anomaly data
                        st.session_state['anomaly_data'] = df_anomaly
                    
                    else:
                        st.success("No significant anomalies detected with current settings.")
    
    with tab3:
        st.subheader("Anomaly Trends Over Time")
        
        if 'anomaly_data' in st.session_state and 'Creation Date' in st.session_state['anomaly_data'].columns:
            anomaly_df = st.session_state['anomaly_data']
            
            # Analyze trends
            monthly_trends = analyze_anomaly_trends(anomaly_df, 'Creation Date')
            
            if not monthly_trends.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Anomaly count over time
                    fig = px.line(
                        monthly_trends, 
                        x='Date', 
                        y='Anomalies',
                        title="Anomaly Count Over Time",
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Anomaly rate over time
                    fig = px.line(
                        monthly_trends, 
                        x='Date', 
                        y='Anomaly_Rate',
                        title="Anomaly Rate (%) Over Time",
                        markers=True
                    )
                    fig.update_layout(height=400)
                    fig.update_yaxis(title="Anomaly Rate (%)")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly spend trend
                fig = px.bar(
                    monthly_trends, 
                    x='Date', 
                    y='Anomaly_Spend',
                    title=f"Anomalous Spending Over Time {currency_suffix}",
                    text='Anomalies'
                )
                fig.update_layout(height=400)
                fig.update_yaxis(title=f"Anomaly Spend ({selected_currency})")
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("No anomaly trend data available. Run anomaly detection first.")
        
        else:
            st.info("No anomaly data available. Please run anomaly detection first.")
    
    with tab4:
        st.subheader(f"Regional Insights and Recommendations")
        
        # Regional summary
        st.subheader(f"{selected_region} Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Region", selected_region)
            st.metric("Primary Currency", f"{primary_currency} {CURRENCY_CONFIG[primary_currency]['symbol']}")
        with col2:
            st.metric("Vendors Analyzed", len(selected_vendors))
            st.metric("Total Records", f"{len(df_filtered):,}")
        with col3:
            st.metric("Date Range", 
                     f"{df_filtered['Creation Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Creation Date'].max().strftime('%Y-%m-%d')}" 
                     if 'Creation Date' in df_filtered.columns else "N/A")
        
        # Key insights
        st.subheader("Key Insights")
        
        # Vendor concentration
        vendor_spend = df_filtered.groupby('Vendor Name')['Line Total'].sum().sort_values(ascending=False)
        top_3_share = (vendor_spend.head(3).sum() / vendor_spend.sum() * 100)
        
        insights = [
            f"**Vendor Concentration**: Top 3 vendors account for {top_3_share:.1f}% of regional spend",
            f"**Average Transaction**: {format_amount(df_filtered['Line Total'].mean(), False)} in {selected_region}",
        ]
        
        for insight in insights:
            st.write(f"• {insight}")
        
        # Regional recommendations
        st.subheader("Regional Recommendations")
        
        region_recommendations = {
            'Latin America': [
                "Consider local currency hedging strategies for volatility",
                "Leverage regional trade agreements for cost optimization",
                "Focus on local supplier development programs"
            ],
            'North America': [
                "Optimize cross-border procurement with trade benefits",
                "Leverage nearshoring opportunities",
                "Consider consolidated distribution centers"
            ],
            'Europe': [
                "Maximize EU single market benefits",
                "Monitor post-Brexit supplier relationships",
                "Consider multi-currency hedging strategies"
            ],
            'Middle East': [
                "Take advantage of regional trading hubs",
                "Consider free zone benefits for imports",
                "Leverage stable currency pegs for predictability"
            ],
            'Africa': [
                "Diversify across multiple markets",
                "Monitor currency volatility and plan accordingly",
                "Consider local partnerships for regulatory compliance"
            ],
            'Asia Pacific': [
                "Diversify supply chain across countries",
                "Monitor multi-currency volatility with hedging",
                "Leverage regional financial hubs"
            ]
        }
        
        recommendations = region_recommendations.get(selected_region, [])
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Export functionality
        st.subheader("Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Filtered Data"):
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    "Download Filtered CSV",
                    csv,
                    f"spend_data_{selected_region}_{selected_currency}.csv",
                    "text/csv"
                )
        
        with col2:
            if 'categorized_data' in st.session_state:
                if st.button("Export Categorized Data"):
                    csv = st.session_state['categorized_data'].to_csv(index=False)
                    st.download_button(
                        "Download Categorized CSV",
                        csv,
                        f"categorized_spend_{selected_region}.csv",
                        "text/csv"
                    )
        
        with col3:
            if 'anomaly_data' in st.session_state:
                if st.button("Export Anomaly Data"):
                    csv = st.session_state['anomaly_data'].to_csv(index=False)
                    st.download_button(
                        "Download Anomaly CSV",
                        csv,
                        f"anomalies_{selected_region}_{selected_currency}.csv",
                        "text/csv"
                    )

if __name__ == "__main__":
    st.set_page_config(page_title="Advanced Spend Analytics", layout="wide")
    
    # Generate sample data
    np.random.seed(42)
    
    # Global regions with realistic data
    regions_data = {
        'Latin America': {
            'vendors': ['AZUL ENERGY COLOMBIA S.A.S.', 'SIEMENS SAS', 'DISTRIBUTION PLUS S.A.S', 'SMARTPROCESS COLOMBIA SAS', 'AUTOAMERICA S.A.'],
            'price_range': (50000, 5000000),
            'currency': 'COP'
        },
        'North America': {
            'vendors': ['ACME CORP', 'TECH SOLUTIONS INC', 'INDUSTRIAL SUPPLY LLC', 'TRANSPORT SERVICES', 'OFFICE DEPOT INC'],
            'price_range': (50, 5000),
            'currency': 'USD'
        },
        'Europe': {
            'vendors': ['SIEMENS AG', 'TOTAL ENERGIES SARL', 'INDUSTRIAL SOLUTIONS GMBH', 'EURO TRANSPORT LTD', 'OFFICE SOLUTIONS PLC'],
            'price_range': (40, 4000),
            'currency': 'EUR'
        },
        'Middle East': {
            'vendors': ['EMIRATES TRADING LLC', 'GULF INDUSTRIAL FZE', 'QATAR ENERGY SERVICES WLL', 'SAUDI TECH ESTABLISHMENT', 'DUBAI LOGISTICS FZCO'],
            'price_range': (150, 15000),
            'currency': 'AED'
        },
        'Africa': {
            'vendors': ['SOUTH AFRICAN MINING PTY LTD', 'NIGERIA INDUSTRIAL LIMITED', 'EGYPT CONSTRUCTION CO', 'KENYA SERVICES LTD', 'GHANA TRADING CC'],
            'price_range': (800, 80000),
            'currency': 'ZAR'
        },
        'Asia Pacific': {
            'vendors': ['TOKYO INDUSTRIAL CO LTD', 'SINGAPORE TECH PTE', 'MUMBAI SERVICES PRIVATE LIMITED', 'SYDNEY EQUIPMENT PTY', 'BANGKOK TRADING CO'],
            'price_range': (3000, 300000),
            'currency': 'JPY'
        }
    }
    
    # Generate mixed regional data
    all_data = []
    
    for region, config in regions_data.items():
        n_records = 100
        
        sample_data = {
            'Vendor Name': np.random.choice(config['vendors'], n_records),
            'Item': [f"Item-{np.random.randint(1000, 9999)}" for _ in range(n_records)],
            'Unit Price': np.random.uniform(config['price_range'][0], config['price_range'][1], n_records),
            'Qty Delivered': np.random.randint(1, 100, n_records),
            'Creation Date': pd.date_range('2024-01-01', periods=n_records, freq='D'),
            'Region': region,
            'Currency': config['currency']
        }
        
        region_df = pd.DataFrame(sample_data)
        region_df['Line Total'] = region_df['Unit Price'] * region_df['Qty Delivered']
        all_data.append(region_df)
    
    # Combine all regional data
    df = pd.concat(all_data, ignore_index=True)
    
    display(df)
