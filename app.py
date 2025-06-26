import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import all modules
import contracting_opportunities
import cross_region
import duplicates
import lot_size_optimization
import reorder_prediction
import seasonal_price_optimization
import spend_categorization_anomaly

def load_and_clean_data(uploaded_file):
    """Load and clean the uploaded data file"""
    try:
        # Read Excel file
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Please upload an Excel (.xlsx, .xls) or CSV file")
            return None
        
        # Data cleaning and transformation
        if not df.empty:
            # Convert numeric columns to numeric, replacing errors with NaN
            numeric_columns = ['Line Total', 'Qty Delivered']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values with 0 for numeric columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Calculate actual unit price from Line Total / Qty Delivered
            df['Actual_Unit_Price'] = np.where(
                (df['Qty Delivered'] > 0) & (df['Line Total'] > 0),
                df['Line Total'] / df['Qty Delivered'],
                0
            )
            
            # Use Ordered Date as Creation Date (since Creation Date is corrupted)
            if 'Ordered Date' in df.columns:
                df['Creation Date'] = pd.to_datetime(df['Ordered Date'], errors='coerce')
            elif 'Creation Date' in df.columns:
                df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce')
            else:
                # If no date columns available, create a placeholder
                df['Creation Date'] = pd.NaT
            
            # Set currency to SAR (since Unit Price column contains currency codes)
            df['Currency_Clean'] = 'SAR'
            
            # Replace the Unit Price column with calculated actual unit price
            df['Unit Price'] = df['Actual_Unit_Price']
            
            # Clean any missing or invalid data
            df = df.dropna(subset=['Vendor Name', 'Item'])
            
            # Filter out invalid numeric data (now that columns are properly numeric)
            df = df[df['Line Total'] > 0]
            df = df[df['Qty Delivered'] > 0]
            
            # Convert dates properly
            date_columns = ['Ordered Date', 'Delivered date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
        else:
            st.error("The uploaded file is empty")
            return None
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.error("Please check that your file contains the required columns and numeric data is properly formatted.")
        return None

def main():
    st.set_page_config(
        page_title="Procurement Analytics Suite", 
        page_icon="📊", 
        layout="wide"
    )
    
    st.title("📊 Procurement Analytics Suite")
    st.markdown("Advanced procurement analysis and optimization tools")
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your procurement data", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload Excel or CSV file with procurement data"
    )
    
    if uploaded_file is not None:
        # Load and clean data
        with st.spinner("Loading and processing data..."):
            df = load_and_clean_data(uploaded_file)
        
        if df is not None:
            # Data overview
            st.sidebar.success(f"✅ Data loaded successfully!")
            st.sidebar.metric("Total Records", len(df))
            
            # Safe date range display
            try:
                if 'Creation Date' in df.columns and not df['Creation Date'].isna().all():
                    min_date = df['Creation Date'].min()
                    max_date = df['Creation Date'].max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        st.sidebar.metric("Date Range", f"{min_date.date()} to {max_date.date()}")
                    else:
                        st.sidebar.info("Date range not available")
                else:
                    st.sidebar.info("Date information not available")
            except Exception:
                st.sidebar.info("Date range not available")
            
            # Module selection
            st.sidebar.header("🔧 Analysis Modules")
            
            analysis_modules = {
                "🤝 Contracting Opportunities": "contracting_opportunities",
                "🌍 Cross-Region Optimization": "cross_region", 
                "🔍 Duplicate Detection": "duplicates",
                "📦 LOT Size Optimization": "lot_size_optimization",
                "📈 Reorder Prediction": "reorder_prediction",
                "🌟 Seasonal Price Optimization": "seasonal_price_optimization",
                "📊 Spend Categorization & Anomaly Detection": "spend_categorization_anomaly"
            }
            
            selected_module = st.sidebar.selectbox(
                "Select Analysis Module",
                options=list(analysis_modules.keys()),
                index=0
            )
            
            # Run selected module
            module_name = analysis_modules[selected_module]
            
            if module_name == "contracting_opportunities":
                contracting_opportunities.display(df)
            elif module_name == "cross_region":
                cross_region.display(df)
            elif module_name == "duplicates":
                duplicates.display(df)
            elif module_name == "lot_size_optimization":
                lot_size_optimization.display(df)
            elif module_name == "reorder_prediction":
                reorder_prediction.display(df)
            elif module_name == "seasonal_price_optimization":
                seasonal_price_optimization.display(df)
            elif module_name == "spend_categorization_anomaly":
                spend_categorization_anomaly.display(df)
        
    else:
        # Welcome Page
        # Custom CSS for styling
        st.markdown("""
        <style>
        .welcome-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }
        .welcome-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .welcome-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            margin: 1.5rem 0 1rem 0;
            color: #333;
        }
        .module-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border-left: 4px solid #667eea;
        }
        .module-title {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 0.3rem;
        }
        .module-description {
            color: #666;
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="welcome-header">
            <div class="welcome-title">📊 Procurement Analytics Platform</div>
            <div class="welcome-subtitle">Advanced AI-Powered Procurement Intelligence & Optimization</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Welcome message
        st.markdown("### 🚀 Welcome to Procurement Analytics Platform")
        
        # Two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 File Format Requirements:")
            
            st.markdown("**Required Columns:**")
            st.markdown("""
            • **Vendor Name** (text): Supplier/vendor name
            • **Unit Price** (number): Price per unit (no $ symbols)  
            • **Qty Delivered** (number): Quantity delivered
            """)
            
            st.markdown("**Optional Columns:**")
            st.markdown("""
            • **Item** (text/number): Item ID or name
            • **Creation Date** (date): Order date
            • **Line Total** (number): Total amount
            • **Item Description** (text): Item description
            """)
            
            st.markdown("### ⚠️ Common Issues to Avoid:")
            st.markdown("""
            • $ symbols in price columns
            • Commas in numbers (1,000)
            • Text in numeric columns ("N/A", "TBD")
            • Inconsistent date formats
            """)
        
        with col2:
            st.markdown("### 📊 Available Analytics Modules:")
            
            modules_info = [
                {
                    "icon": "🤝",
                    "title": "Contracting Opportunities",
                    "description": "Identify optimal contracting opportunities based on spend analysis."
                },
                {
                    "icon": "🌟", 
                    "title": "Seasonal Price Optimization",
                    "description": "Analyze seasonal price patterns to optimize purchase timing."
                },
                {
                    "icon": "📊",
                    "title": "Spend Categorization & Anomaly Detection", 
                    "description": "AI-powered spend categorization and anomaly detection."
                },
                {
                    "icon": "📦",
                    "title": "LOT Size Optimization",
                    "description": "Economic Order Quantity (EOQ) analysis for inventory optimization."
                },
                {
                    "icon": "🌍",
                    "title": "Cross-Region Analysis",
                    "description": "Compare pricing and performance across different regions."
                },
                {
                    "icon": "🔍",
                    "title": "Duplicate Detection", 
                    "description": "Identify potential duplicate vendors and items."
                },
                {
                    "icon": "📈",
                    "title": "Reorder Prediction",
                    "description": "Smart reorder point calculation and demand forecasting."
                }
            ]
            
            for module in modules_info:
                st.markdown(f"""
                <div class="module-card">
                    <div class="module-title">{module['icon']} {module['title']}</div>
                    <div class="module-description">{module['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Upload instruction
        st.markdown("---")
        st.info("👆 **To get started:** Upload your procurement data file using the sidebar on the left, then select an analysis module.")
        
        # Sample data preview
        with st.expander("📊 View Sample Data Format"):
            sample_data = {
                'Vendor Name': ['ABC Suppliers', 'XYZ Corp', 'Tech Solutions Inc'],
                'Unit Price': [25.50, 150.00, 75.25],
                'Qty Delivered': [100, 25, 50],
                'Item': ['ITEM001', 'ITEM002', 'ITEM003'],
                'Item Description': ['Office Supplies', 'Computer Equipment', 'Software License'],
                'Line Total': [2550.00, 3750.00, 3762.50],
                'Creation Date': ['2024-01-15', '2024-01-16', '2024-01-17']
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()
