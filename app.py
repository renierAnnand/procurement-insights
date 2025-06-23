import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import importlib.util

# Set page configuration
st.set_page_config(
    page_title="Procurement Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def load_module_dynamically(module_name, file_path):
    """Dynamically load a Python module from file path"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    except Exception as e:
        return None, str(e)

def check_module_availability():
    """Check which analytics modules are available"""
    modules_status = {}
    
    # Define expected modules and their file paths
    expected_modules = {
        'contracting_opportunities': './contracting_opportunities.py',
        'seasonal_price_optimization': './seasonal_price_optimization.py', 
        'spend_categorization_anomaly': './spend_categorization_anomaly.py',
        'lot_size_optimization': './lot_size_optimization.py',
        'cross_region': './cross_region.py',
        'duplicates': './duplicates.py',
        'reorder_prediction': './reorder_prediction.py'
    }
    
    for module_name, file_path in expected_modules.items():
        if os.path.exists(file_path):
            module, error = load_module_dynamically(module_name, file_path)
            if module is not None:
                modules_status[module_name] = {
                    'status': 'available',
                    'module': module,
                    'path': file_path
                }
            else:
                modules_status[module_name] = {
                    'status': 'error',
                    'error': error,
                    'path': file_path
                }
        else:
            modules_status[module_name] = {
                'status': 'missing',
                'path': file_path
            }
    
    return modules_status

def create_sample_data():
    """Create sample procurement data for testing"""
    np.random.seed(42)
    
    # Sample vendors and items
    vendors = ['Global Supplies Inc', 'Tech Solutions Ltd', 'Industrial Materials Co',
               'Office Essentials', 'Manufacturing Parts Ltd', 'Quality Components']
    items = range(1, 51)  # 50 different items
    departments = ['Engineering', 'Operations', 'IT', 'Facilities', 'R&D']
    warehouses = ['Warehouse A', 'Warehouse B', 'Warehouse C', 'Warehouse D']
    
    # Generate sample data
    n_records = 1000
    
    data = {
        'Vendor Name': np.random.choice(vendors, n_records),
        'Item': np.random.choice(items, n_records),
        'Item Description': [f"Product {item} - Description" for item in np.random.choice(items, n_records)],
        'Unit Price': np.random.uniform(5, 500, n_records),
        'Qty Delivered': np.random.randint(1, 100, n_records),
        'Creation Date': pd.date_range('2022-01-01', '2024-12-31', periods=n_records),
        'DEP': np.random.choice(departments, n_records),
        'W/H': np.random.choice(warehouses, n_records),
        'Product Family': np.random.choice(['Electronics', 'Office Supplies', 'Industrial', 'IT Equipment'], n_records)
    }
    
    df = pd.DataFrame(data)
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    
    return df

def display_module_status(modules_status):
    """Display the status of all analytics modules"""
    st.sidebar.markdown("### 📊 Analytics Modules Status")
    
    available_count = sum(1 for m in modules_status.values() if m['status'] == 'available')
    total_count = len(modules_status)
    
    st.sidebar.metric("Available Modules", f"{available_count}/{total_count}")
    
    for module_name, status in modules_status.items():
        display_name = module_name.replace('_', ' ').title()
        
        if status['status'] == 'available':
            st.sidebar.success(f"✅ {display_name}")
        elif status['status'] == 'error':
            st.sidebar.error(f"❌ {display_name}")
            with st.sidebar.expander(f"Error details - {display_name}"):
                st.code(status['error'])
        else:
            st.sidebar.warning(f"⚠️ {display_name} (Missing)")
            st.sidebar.caption(f"Expected: {status['path']}")

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Procurement Analytics Platform</h1>
        <p>Advanced AI-Powered Procurement Intelligence & Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check module availability
    modules_status = check_module_availability()
    
    # Sidebar with module status
    display_module_status(modules_status)
    
    # File upload section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Procurement Data",
        type=['csv', 'xlsx'],
        help="Upload your procurement/spend data file (CSV or Excel)"
    )
    
    # Load data
    df = None
    data_source = ""
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            data_source = f"Uploaded file: {uploaded_file.name}"
            st.sidebar.success(f"✅ Loaded {len(df)} records")
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
    
    # Option to use sample data
    if df is None:
        if st.sidebar.button("📝 Load Sample Data"):
            df = create_sample_data()
            data_source = "Sample data generated"
            st.sidebar.success("✅ Sample data loaded")
    
    # Main content area
    if df is not None:
        # Data overview
        st.subheader(f"📊 Data Overview - {data_source}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            unique_vendors = df['Vendor Name'].nunique() if 'Vendor Name' in df.columns else 0
            st.metric("Unique Vendors", unique_vendors)
        with col3:
            unique_items = df['Item'].nunique() if 'Item' in df.columns else 0
            st.metric("Unique Items", unique_items)
        with col4:
            if all(col in df.columns for col in ['Unit Price', 'Qty Delivered']):
                total_spend = (df['Unit Price'] * df['Qty Delivered']).sum()
                st.metric("Total Spend", f"${total_spend:,.0f}")
            else:
                st.metric("Data Quality", "Needs Review")
        
        # Data preview
        with st.expander("👀 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data quality check
            st.subheader("🔍 Data Quality Check")
            quality_issues = []
            
            # Check for required columns
            required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                quality_issues.append(f"Missing columns: {', '.join(missing_columns)}")
            
            # Check for null values
            null_counts = df.isnull().sum()
            significant_nulls = null_counts[null_counts > len(df) * 0.1]
            
            if len(significant_nulls) > 0:
                quality_issues.append(f"High null values in: {', '.join(significant_nulls.index)}")
            
            # Check for data types
            if 'Creation Date' in df.columns:
                try:
                    pd.to_datetime(df['Creation Date'])
                except:
                    quality_issues.append("Creation Date column needs date format conversion")
            
            if quality_issues:
                for issue in quality_issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ Data quality looks good!")
        
        # Module selection and execution
        st.markdown("---")
        st.subheader("🚀 Analytics Modules")
        
        # Get available modules
        available_modules = {name: status for name, status in modules_status.items() 
                           if status['status'] == 'available'}
        
        if available_modules:
            # Module selection
            module_options = {
                'contracting_opportunities': '🤝 Enhanced Contracting Opportunities',
                'seasonal_price_optimization': '🌟 Seasonal Price Optimization',
                'spend_categorization_anomaly': '📊 Spend Analysis & Anomaly Detection',
                'lot_size_optimization': '📦 LOT Size Optimization',
                'cross_region': '🌍 Cross-Region Analysis',
                'duplicates': '🔍 Duplicate Detection',
                'reorder_prediction': '📈 Reorder Prediction'
            }
            
            # Filter to available modules
            available_options = {k: v for k, v in module_options.items() if k in available_modules}
            
            selected_module = st.selectbox(
                "Select Analytics Module:",
                list(available_options.keys()),
                format_func=lambda x: available_options[x],
                index=0
            )
            
            # Module description
            module_descriptions = {
                'contracting_opportunities': "🎯 AI-powered contract analysis with advanced ML, real-time data integration, and comprehensive reporting. Includes vendor segmentation, risk assessment, and TCO analysis.",
                'seasonal_price_optimization': "📈 Advanced seasonal analysis with ML forecasting, market intelligence, and optimal timing recommendations.",
                'spend_categorization_anomaly': "🔍 AI-powered spend categorization with advanced anomaly detection using multiple ML algorithms.",
                'lot_size_optimization': "📊 Economic Order Quantity optimization with ML demand forecasting and inventory analysis.",
                'cross_region': "🌍 Cross-regional vendor optimization and consolidation analysis.",
                'duplicates': "🔎 Intelligent duplicate vendor and item detection using fuzzy matching.",
                'reorder_prediction': "📈 Smart reorder point prediction with statistical analysis."
            }
            
            if selected_module in module_descriptions:
                st.info(module_descriptions[selected_module])
            
            # Execute selected module
            st.markdown("---")
            
            try:
                module = available_modules[selected_module]['module']
                
                # Call the module's display function
                if hasattr(module, 'display'):
                    module.display(df)
                elif hasattr(module, 'display_enhanced_contracting'):
                    module.display_enhanced_contracting(df)
                else:
                    st.error(f"Module {selected_module} doesn't have a proper display function")
                    
            except Exception as e:
                st.error(f"Error running module {selected_module}: {str(e)}")
                
                # Show detailed error in expander
                with st.expander("🔧 Debug Information"):
                    st.code(f"Error: {str(e)}")
                    st.code(f"Module path: {available_modules[selected_module]['path']}")
        
        else:
            st.warning("⚠️ No analytics modules are currently available.")
            st.info("Please ensure the module files are in the correct location:")
            
            for module_name, status in modules_status.items():
                st.code(f"{module_name}.py - Expected at: {status['path']}")
    
    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        ## 🎯 Welcome to Procurement Analytics Platform
        
        **Get Started:**
        1. 📁 **Upload your data** using the sidebar file uploader
        2. 📊 **Or use sample data** to explore the platform
        3. 🚀 **Select an analytics module** to begin analysis
        
        ### 🌟 Platform Capabilities
        
        **🤝 Enhanced Contracting Opportunities**
        - AI-powered contract identification and optimization
        - Advanced ML models with ensemble methods
        - Real-time data integration and monitoring
        - Professional PDF and Excel report generation
        - Vendor segmentation and risk assessment
        
        **🌟 Seasonal Price Optimization**
        - ML-powered seasonal pattern detection
        - Optimal purchase timing recommendations
        - Market intelligence integration
        - Advanced forecasting with confidence intervals
        
        **📊 Spend Analysis & Anomaly Detection**
        - AI-powered spend categorization
        - Advanced anomaly detection with multiple algorithms
        - Real-time monitoring and alerting
        - Comprehensive data quality analysis
        
        **📦 LOT Size Optimization**
        - Economic Order Quantity (EOQ) calculations
        - ML-powered demand forecasting
        - Inventory optimization recommendations
        - Cost analysis and savings potential
        
        ### 🔧 System Requirements
        
        **Data Format:** CSV or Excel files with procurement/spend data
        
        **Required Columns:**
        - Vendor Name
        - Item (ID or description)
        - Unit Price
        - Qty Delivered
        - Creation Date
        
        **Optional Columns:**
        - Line Total
        - Department (DEP)
        - Warehouse (W/H)
        - Product Family
        - Item Description
        """)
        
        # Show module availability status
        st.markdown("### 📊 Module Availability")
        
        available_count = sum(1 for m in modules_status.values() if m['status'] == 'available')
        total_count = len(modules_status)
        
        if available_count == total_count:
            st.success(f"✅ All {total_count} analytics modules are available and ready!")
        elif available_count > 0:
            st.warning(f"⚠️ {available_count} of {total_count} modules are available. Some modules may be missing.")
        else:
            st.error("❌ No analytics modules are currently available. Please check module files.")
        
        # Quick setup guide
        with st.expander("🚀 Quick Setup Guide"):
            st.markdown("""
            **To get all modules working:**
            
            1. **Enhanced Contracting Module**: Save the enhanced contracting code as `contracting_opportunities.py`
            2. **Other Modules**: Ensure these files exist in your project directory:
               - `seasonal_price_optimization.py`
               - `spend_categorization_anomaly.py` 
               - `lot_size_optimization.py`
               - `cross_region.py`
               - `duplicates.py`
               - `reorder_prediction.py`
            
            3. **File Structure**:
            ```
            your_project/
            ├── app.py (this main file)
            ├── contracting_opportunities.py
            ├── seasonal_price_optimization.py
            ├── spend_categorization_anomaly.py
            ├── lot_size_optimization.py
            ├── cross_region.py
            ├── duplicates.py
            └── reorder_prediction.py
            ```
            
            4. **Run the app**: `streamlit run app.py`
            """)

if __name__ == "__main__":
    main()
