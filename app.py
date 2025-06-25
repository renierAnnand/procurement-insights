import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Import your optimization module
import lot_size_optimization

# Configure the page
st.set_page_config(
    page_title="Procurement Insights Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_sample_data():
    """Generate sample procurement data"""
    np.random.seed(42)
    items = ['Widget-A', 'Component-B', 'Material-C', 'Part-D', 'Tool-E', 
             'Supply-F', 'Equipment-G', 'Raw-Material-H', 'Assembly-I', 'Module-J']
    
    n_records = 200
    data = {
        'Item': np.random.choice(items, n_records),
        'Unit Price': np.random.uniform(5, 100, n_records),
        'Qty Delivered': np.random.randint(10, 500, n_records),
        'Supplier': np.random.choice(['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D'], n_records),
        'Order Date': pd.date_range('2024-01-01', '2024-12-31', periods=n_records),
        'Category': np.random.choice(['Electronics', 'Mechanical', 'Raw Materials', 'Tools'], n_records)
    }
    return pd.DataFrame(data)

def main():
    # Header
    st.title("🛒 Procurement Insights Dashboard")
    st.markdown("**Optimize your procurement strategies with data-driven insights**")
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("📋 Navigation")
        
        # Data source selection
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Sample Data", "Upload CSV File"]
        )
        
        # Load data based on selection
        if data_source == "Sample Data":
            df = load_sample_data()
            st.success(f"✅ Loaded {len(df)} sample records")
            
        else:  # Upload CSV
            uploaded_file = st.file_uploader(
                "Upload your procurement data (CSV)",
                type=['csv'],
                help="CSV should contain columns: Item, Unit Price, Qty Delivered"
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
                st.info("Using sample data (upload a file to use your own data)")
        
        # Page selection
        st.subheader("🎯 Analysis Modules")
        page = st.selectbox(
            "Choose analysis:",
            [
                "📦 LOT Size Optimization",
                "🏭 Supplier Analysis", 
                "💰 Cost Analysis",
                "📈 Performance Dashboard",
                "📊 Data Overview"
            ]
        )
    
    # Main content area
    if page == "📦 LOT Size Optimization":
        # Use your existing optimization module
        lot_size_optimization.display(df)
        
    elif page == "📊 Data Overview":
        st.header("📊 Data Overview")
        
        if df is not None and len(df) > 0:
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique Items", df['Item'].nunique() if 'Item' in df.columns else 0)
            with col3:
                if 'Unit Price' in df.columns:
                    avg_price = df['Unit Price'].mean()
                    st.metric("Avg Unit Price", f"${avg_price:.2f}")
                else:
                    st.metric("Avg Unit Price", "N/A")
            with col4:
                if 'Qty Delivered' in df.columns:
                    total_qty = df['Qty Delivered'].sum()
                    st.metric("Total Quantity", f"{total_qty:,.0f}")
                else:
                    st.metric("Total Quantity", "N/A")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Data quality check
            st.subheader("🔍 Data Quality")
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.warning("Missing data found:")
                st.write(missing_data[missing_data > 0])
            else:
                st.success("✅ No missing data found")
            
            # Basic visualizations
            if 'Item' in df.columns and 'Qty Delivered' in df.columns:
                st.subheader("📊 Quick Insights")
                
                # Top items by quantity
                top_items = df.groupby('Item')['Qty Delivered'].sum().sort_values(ascending=False).head(10)
                fig1 = px.bar(x=top_items.values, y=top_items.index, orientation='h',
                             title="Top 10 Items by Total Quantity")
                st.plotly_chart(fig1, use_container_width=True)
                
                # Price distribution
                if 'Unit Price' in df.columns:
                    fig2 = px.histogram(df, x='Unit Price', title="Unit Price Distribution", 
                                       nbins=30)
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No data available for analysis")
    
    elif page == "🏭 Supplier Analysis":
        st.header("🏭 Supplier Performance Analysis")
        
        if 'Supplier' in df.columns:
            # Supplier metrics
            supplier_stats = df.groupby('Supplier').agg({
                'Qty Delivered': ['sum', 'count'],
                'Unit Price': 'mean'
            }).round(2)
            
            supplier_stats.columns = ['Total Quantity', 'Order Count', 'Avg Unit Price']
            supplier_stats = supplier_stats.sort_values('Total Quantity', ascending=False)
            
            # Display supplier rankings
            st.subheader("📊 Supplier Rankings")
            st.dataframe(supplier_stats, use_container_width=True)
            
            # Supplier comparison chart
            fig = px.bar(x=supplier_stats.index, y=supplier_stats['Total Quantity'],
                        title="Total Quantity by Supplier")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No supplier data available in the dataset")
    
    elif page == "💰 Cost Analysis":
        st.header("💰 Cost Analysis")
        
        if 'Unit Price' in df.columns and 'Qty Delivered' in df.columns:
            # Calculate total costs
            df['Total Cost'] = df['Unit Price'] * df['Qty Delivered']
            
            # Cost metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                total_cost = df['Total Cost'].sum()
                st.metric("Total Procurement Cost", f"${total_cost:,.2f}")
            with col2:
                avg_order_cost = df['Total Cost'].mean()
                st.metric("Average Order Cost", f"${avg_order_cost:.2f}")
            with col3:
                median_order_cost = df['Total Cost'].median()
                st.metric("Median Order Cost", f"${median_order_cost:.2f}")
            
            # Cost by category/item
            if 'Category' in df.columns:
                category_costs = df.groupby('Category')['Total Cost'].sum().sort_values(ascending=False)
                fig = px.pie(values=category_costs.values, names=category_costs.index,
                           title="Cost Distribution by Category")
                st.plotly_chart(fig, use_container_width=True)
            
            # Top cost items
            item_costs = df.groupby('Item')['Total Cost'].sum().sort_values(ascending=False).head(10)
            fig2 = px.bar(x=item_costs.values, y=item_costs.index, orientation='h',
                         title="Top 10 Items by Total Cost")
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.info("Unit Price and Qty Delivered columns required for cost analysis")
    
    elif page == "📈 Performance Dashboard":
        st.header("📈 Performance Dashboard")
        
        # KPI Overview
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Item' in df.columns:
                item_variety = df['Item'].nunique()
                st.metric("Item Variety", item_variety)
            
        with col2:
            if 'Supplier' in df.columns:
                supplier_count = df['Supplier'].nunique()
                st.metric("Active Suppliers", supplier_count)
            
        with col3:
            if 'Order Date' in df.columns:
                order_frequency = len(df) / 365  # assuming 1 year of data
                st.metric("Orders per Day", f"{order_frequency:.1f}")
            
        with col4:
            if 'Unit Price' in df.columns and 'Qty Delivered' in df.columns:
                df['Total Cost'] = df['Unit Price'] * df['Qty Delivered']
                avg_order_value = df['Total Cost'].mean()
                st.metric("Avg Order Value", f"${avg_order_value:.2f}")
        
        # Performance trends (if date data available)
        if 'Order Date' in df.columns:
            st.subheader("📊 Trends Over Time")
            
            # Convert to datetime if not already
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            
            # Monthly trends
            monthly_orders = df.groupby(df['Order Date'].dt.to_period('M')).size()
            
            fig = px.line(x=monthly_orders.index.astype(str), y=monthly_orders.values,
                         title="Orders per Month")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date information needed for trend analysis")

    # Footer
    st.markdown("---")
    st.markdown("**Procurement Insights Dashboard** | Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
