import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO

# Import all your analysis modules
import lot_size_optimization
import contracting_opportunities
import seasonal_price_optimization
import spend_categorization_anomaly
import duplicates
import cross_region
import reorder_prediction

# Configure the page
st.set_page_config(
    page_title="Procurement Analytics Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_sample_data():
    """Generate comprehensive sample procurement data"""
    np.random.seed(42)
    
    # Sample vendors and items
    vendors = [
        'Tech Solutions Inc', 'Office Supplies Corp', 'Industrial Materials Ltd',
        'Global Electronics', 'Professional Services LLC', 'Equipment Rental Co',
        'Quality Components', 'Logistics Partners', 'Manufacturing Supplies',
        'Digital Services Group', 'Safety Equipment Co', 'Facility Management Inc'
    ]
    
    items = [
        'Laptop Computer', 'Office Chair', 'Steel Rod', 'Printer Cartridge',
        'Consulting Services', 'Forklift Rental', 'Electronic Component',
        'Shipping Service', 'Raw Material', 'Software License',
        'Safety Equipment', 'Cleaning Service', 'Training Program',
        'Maintenance Kit', 'Office Supplies', 'IT Hardware'
    ]
    
    # Generate sample data
    n_records = 500
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    data = {
        'Vendor Name': np.random.choice(vendors, n_records),
        'Item': np.random.choice(range(1000, 2000), n_records),  # Item IDs
        'Item Description': np.random.choice(items, n_records),
        'Unit Price': np.random.uniform(5, 500, n_records),
        'Qty Delivered': np.random.randint(1, 100, n_records),
        'Creation Date': pd.to_datetime(np.random.choice(pd.date_range(start_date, end_date), n_records)),
        'W/H': np.random.choice(['Warehouse A', 'Warehouse B', 'Warehouse C', 'Central'], n_records),
        'Category': np.random.choice(['IT', 'Office', 'Manufacturing', 'Services', 'Facilities'], n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Add calculated fields
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    df['Qty Rejected'] = np.random.randint(0, df['Qty Delivered'] // 10 + 1, n_records)
    
    # Add some seasonal patterns to prices
    df['Month'] = df['Creation Date'].dt.month
    seasonal_multiplier = 1 + 0.1 * np.sin(2 * np.pi * df['Month'] / 12)
    df['Unit Price'] = df['Unit Price'] * seasonal_multiplier
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    
    return df

def display_dashboard_overview(df):
    """Display high-level dashboard metrics"""
    st.header("📊 Procurement Analytics Dashboard")
    st.markdown("**Comprehensive procurement insights and optimization platform**")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_spend = df['Line Total'].sum()
        st.metric("Total Spend", f"${total_spend:,.0f}")
    
    with col2:
        unique_vendors = df['Vendor Name'].nunique()
        st.metric("Active Vendors", unique_vendors)
    
    with col3:
        unique_items = df['Item'].nunique()
        st.metric("Unique Items", unique_items)
    
    with col4:
        avg_order_value = df['Line Total'].mean()
        st.metric("Avg Order Value", f"${avg_order_value:.0f}")
    
    with col5:
        total_orders = len(df)
        st.metric("Total Orders", total_orders)
    
    # Quick insights
    st.subheader("🎯 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top vendors by spend
        top_vendors = df.groupby('Vendor Name')['Line Total'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_vendors.values,
            y=top_vendors.index,
            orientation='h',
            title="Top 10 Vendors by Spend",
            labels={'x': 'Total Spend ($)', 'y': 'Vendor'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly spend trend
        if 'Creation Date' in df.columns:
            monthly_spend = df.groupby(df['Creation Date'].dt.to_period('M'))['Line Total'].sum()
            
            fig = px.line(
                x=monthly_spend.index.astype(str),
                y=monthly_spend.values,
                title="Monthly Spend Trend",
                labels={'x': 'Month', 'y': 'Spend ($)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.subheader("📋 Recent Transactions")
    st.dataframe(
        df.head(10).style.format({
            'Unit Price': '${:.2f}',
            'Line Total': '${:,.0f}'
        }),
        use_container_width=True
    )

def main():
    # Header
    st.title("🛒 Procurement Analytics Platform")
    st.markdown("**Advanced procurement insights and optimization suite**")
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Data source selection
        data_source = st.radio(
            "Choose data source:",
            ["📊 Sample Data", "📤 Upload CSV File"]
        )
        
        # Load data based on selection
        if data_source == "📊 Sample Data":
            df = load_sample_data()
            st.success(f"✅ Loaded {len(df)} sample records")
            
        else:  # Upload CSV
            uploaded_file = st.file_uploader(
                "Upload your procurement data (CSV)",
                type=['csv'],
                help="CSV should contain columns like: Vendor Name, Item, Unit Price, Qty Delivered, Creation Date"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} records from file")
                    
                    # Show column info
                    st.write("**Columns found:**")
                    st.write(list(df.columns))
                    
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    df = load_sample_data()
                    st.info("Using sample data instead")
            else:
                df = load_sample_data()
                st.info("Using sample data (upload a file to use your own)")
        
        # Navigation menu
        st.header("🎯 Analytics Modules")
        
        modules = {
            "📊 Dashboard Overview": "dashboard",
            "📦 LOT Size Optimization": "lot_size",
            "🤝 Contracting Opportunities": "contracting",
            "🌟 Seasonal Price Optimization": "seasonal",
            "📊 Spend Analysis & Anomalies": "spend_anomaly",
            "🔍 Duplicate Detection": "duplicates",
            "🌍 Cross-Region Analysis": "cross_region",
            "📈 Reorder Prediction": "reorder",
            "📋 Data Overview": "data_overview"
        }
        
        selected_module = st.selectbox(
            "Select Analysis Module:",
            list(modules.keys())
        )
        
        module_key = modules[selected_module]
        
        # Data quality check
        st.header("📊 Data Quality")
        if df is not None:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
            
            if completeness < 80:
                st.warning("⚠️ Low data quality detected")
            else:
                st.success("✅ Good data quality")
    
    # Main content area
    if df is not None and len(df) > 0:
        if module_key == "dashboard":
            display_dashboard_overview(df)
            
        elif module_key == "lot_size":
            lot_size_optimization.display(df)
            
        elif module_key == "contracting":
            contracting_opportunities.display(df)
            
        elif module_key == "seasonal":
            seasonal_price_optimization.display(df)
            
        elif module_key == "spend_anomaly":
            spend_categorization_anomaly.display(df)
            
        elif module_key == "duplicates":
            duplicates.display(df)
            
        elif module_key == "cross_region":
            cross_region.display(df)
            
        elif module_key == "reorder":
            reorder_prediction.display(df)
            
        elif module_key == "data_overview":
            st.header("📊 Detailed Data Overview")
            
            # Comprehensive data analysis
            st.subheader("📈 Data Statistics")
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Record Count:**")
                st.write(f"Total Records: {len(df):,}")
                st.write(f"Complete Records: {len(df.dropna()):,}")
                st.write(f"Date Range: {df['Creation Date'].min().date()} to {df['Creation Date'].max().date()}" if 'Creation Date' in df.columns else "No date data")
            
            with col2:
                st.write("**Vendor Analysis:**")
                st.write(f"Unique Vendors: {df['Vendor Name'].nunique()}")
                top_vendor = df.groupby('Vendor Name')['Line Total'].sum().index[0]
                st.write(f"Top Vendor: {top_vendor}")
                vendor_concentration = (df.groupby('Vendor Name')['Line Total'].sum().max() / df['Line Total'].sum() * 100)
                st.write(f"Vendor Concentration: {vendor_concentration:.1f}%")
            
            with col3:
                st.write("**Financial Summary:**")
                st.write(f"Total Spend: ${df['Line Total'].sum():,.0f}")
                st.write(f"Average Order: ${df['Line Total'].mean():.0f}")
                st.write(f"Largest Order: ${df['Line Total'].max():,.0f}")
            
            # Column information
            st.subheader("📋 Column Information")
            
            column_info = []
            for col in df.columns:
                col_info = {
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-Null Count': df[col].count(),
                    'Null Count': df[col].isnull().sum(),
                    'Unique Values': df[col].nunique()
                }
                column_info.append(col_info)
            
            col_df = pd.DataFrame(column_info)
            st.dataframe(col_df, use_container_width=True)
            
            # Data quality issues
            st.subheader("🔍 Data Quality Assessment")
            
            quality_issues = []
            
            # Check for missing data
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                for col, missing_count in missing_data[missing_data > 0].items():
                    quality_issues.append({
                        'Issue Type': 'Missing Data',
                        'Column': col,
                        'Count': missing_count,
                        'Percentage': f"{missing_count/len(df)*100:.1f}%"
                    })
            
            # Check for duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                quality_issues.append({
                    'Issue Type': 'Duplicate Records',
                    'Column': 'All',
                    'Count': duplicate_count,
                    'Percentage': f"{duplicate_count/len(df)*100:.1f}%"
                })
            
            # Check for negative values in price/quantity
            if 'Unit Price' in df.columns:
                negative_prices = (df['Unit Price'] < 0).sum()
                if negative_prices > 0:
                    quality_issues.append({
                        'Issue Type': 'Negative Prices',
                        'Column': 'Unit Price',
                        'Count': negative_prices,
                        'Percentage': f"{negative_prices/len(df)*100:.1f}%"
                    })
            
            if quality_issues:
                st.dataframe(pd.DataFrame(quality_issues), use_container_width=True)
            else:
                st.success("✅ No major data quality issues detected!")
            
            # Raw data preview
            st.subheader("📄 Raw Data Preview")
            
            # Pagination for large datasets
            page_size = 50
            total_pages = (len(df) - 1) // page_size + 1
            
            if total_pages > 1:
                page_num = st.selectbox("Select Page", range(1, total_pages + 1))
                start_idx = (page_num - 1) * page_size
                end_idx = min(start_idx + page_size, len(df))
                display_df = df.iloc[start_idx:end_idx]
                st.write(f"Showing records {start_idx + 1} to {end_idx} of {len(df)}")
            else:
                display_df = df
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export functionality
            if st.button("📥 Export Data Quality Report"):
                report_data = {
                    'summary': [
                        f"Total Records: {len(df)}",
                        f"Unique Vendors: {df['Vendor Name'].nunique()}",
                        f"Total Spend: ${df['Line Total'].sum():,.0f}",
                        f"Data Completeness: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%"
                    ],
                    'columns': col_df.to_dict('records'),
                    'quality_issues': quality_issues
                }
                
                # Convert to CSV format
                summary_text = "DATA QUALITY REPORT\n" + "="*50 + "\n"
                summary_text += "\n".join(report_data['summary']) + "\n\n"
                summary_text += "COLUMN INFORMATION\n" + "-"*30 + "\n"
                summary_text += col_df.to_string() + "\n\n"
                
                if quality_issues:
                    summary_text += "QUALITY ISSUES\n" + "-"*20 + "\n"
                    summary_text += pd.DataFrame(quality_issues).to_string()
                
                st.download_button(
                    label="Download Report",
                    data=summary_text,
                    file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
        
    else:
        st.error("No data available for analysis. Please check your data source.")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown("**Procurement Analytics Platform** | Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
