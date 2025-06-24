import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from math import sqrt

# Comprehensive Regional Currency Mapping
REGION_CURRENCIES = {
    # Middle East & GCC
    'Saudi Arabia': {'symbol': 'SR', 'name': 'SAR'},
    'UAE': {'symbol': 'AED', 'name': 'AED'},
    'Kuwait': {'symbol': 'KD', 'name': 'KWD'},
    'Qatar': {'symbol': 'QR', 'name': 'QAR'},
    'Bahrain': {'symbol': 'BD', 'name': 'BHD'},
    'Oman': {'symbol': 'OMR', 'name': 'OMR'},
    'Iraq': {'symbol': 'IQD', 'name': 'IQD'},
    'Iran': {'symbol': 'IRR', 'name': 'IRR'},
    'Israel': {'symbol': '₪', 'name': 'ILS'},
    'Turkey': {'symbol': '₺', 'name': 'TRY'},
    
    # North Africa & Levant
    'Egypt': {'symbol': 'E£', 'name': 'EGP'},
    'Jordan': {'symbol': 'JD', 'name': 'JOD'},
    'Lebanon': {'symbol': 'L£', 'name': 'LBP'},
    'Morocco': {'symbol': 'MAD', 'name': 'MAD'},
    'Tunisia': {'symbol': 'TND', 'name': 'TND'},
    'Algeria': {'symbol': 'DZD', 'name': 'DZD'},
    'Libya': {'symbol': 'LYD', 'name': 'LYD'},
    'Sudan': {'symbol': 'SDG', 'name': 'SDG'},
    
    # Major Global Economies
    'USA': {'symbol': '$', 'name': 'USD'},
    'Canada': {'symbol': 'C$', 'name': 'CAD'},
    'United Kingdom': {'symbol': '£', 'name': 'GBP'},
    'European Union': {'symbol': '€', 'name': 'EUR'},
    'Germany': {'symbol': '€', 'name': 'EUR'},
    'France': {'symbol': '€', 'name': 'EUR'},
    'Italy': {'symbol': '€', 'name': 'EUR'},
    'Spain': {'symbol': '€', 'name': 'EUR'},
    'Netherlands': {'symbol': '€', 'name': 'EUR'},
    'Switzerland': {'symbol': 'CHF', 'name': 'CHF'},
    'Norway': {'symbol': 'kr', 'name': 'NOK'},
    'Sweden': {'symbol': 'kr', 'name': 'SEK'},
    'Denmark': {'symbol': 'kr', 'name': 'DKK'},
    
    # Asia Pacific
    'China': {'symbol': '¥', 'name': 'CNY'},
    'Japan': {'symbol': '¥', 'name': 'JPY'},
    'South Korea': {'symbol': '₩', 'name': 'KRW'},
    'India': {'symbol': '₹', 'name': 'INR'},
    'Singapore': {'symbol': 'S$', 'name': 'SGD'},
    'Hong Kong': {'symbol': 'HK$', 'name': 'HKD'},
    'Taiwan': {'symbol': 'NT$', 'name': 'TWD'},
    'Thailand': {'symbol': '฿', 'name': 'THB'},
    'Malaysia': {'symbol': 'RM', 'name': 'MYR'},
    'Indonesia': {'symbol': 'Rp', 'name': 'IDR'},
    'Philippines': {'symbol': '₱', 'name': 'PHP'},
    'Vietnam': {'symbol': '₫', 'name': 'VND'},
    'Australia': {'symbol': 'A$', 'name': 'AUD'},
    'New Zealand': {'symbol': 'NZ$', 'name': 'NZD'},
    
    # Eastern Europe
    'Russia': {'symbol': '₽', 'name': 'RUB'},
    'Poland': {'symbol': 'zł', 'name': 'PLN'},
    'Czech Republic': {'symbol': 'Kč', 'name': 'CZK'},
    'Hungary': {'symbol': 'Ft', 'name': 'HUF'},
    'Romania': {'symbol': 'lei', 'name': 'RON'},
    'Bulgaria': {'symbol': 'лв', 'name': 'BGN'},
    'Ukraine': {'symbol': '₴', 'name': 'UAH'},
    
    # Latin America
    'Brazil': {'symbol': 'R$', 'name': 'BRL'},
    'Mexico': {'symbol': 'MX$', 'name': 'MXN'},
    'Argentina': {'symbol': 'AR$', 'name': 'ARS'},
    'Chile': {'symbol': 'CL$', 'name': 'CLP'},
    'Colombia': {'symbol': 'CO$', 'name': 'COP'},
    'Peru': {'symbol': 'S/', 'name': 'PEN'},
    'Venezuela': {'symbol': 'Bs', 'name': 'VES'},
    'Ecuador': {'symbol': '$', 'name': 'USD'},
    'Uruguay': {'symbol': 'UY$', 'name': 'UYU'},
    
    # Africa
    'South Africa': {'symbol': 'R', 'name': 'ZAR'},
    'Nigeria': {'symbol': '₦', 'name': 'NGN'},
    'Kenya': {'symbol': 'KSh', 'name': 'KES'},
    'Ghana': {'symbol': 'GH₵', 'name': 'GHS'},
    'Ethiopia': {'symbol': 'Br', 'name': 'ETB'},
    'Tanzania': {'symbol': 'TSh', 'name': 'TZS'},
    'Uganda': {'symbol': 'USh', 'name': 'UGX'},
    'Zambia': {'symbol': 'ZK', 'name': 'ZMW'},
    'Botswana': {'symbol': 'P', 'name': 'BWP'},
    'Namibia': {'symbol': 'N$', 'name': 'NAD'},
    
    # Other Important Economies
    'Pakistan': {'symbol': '₨', 'name': 'PKR'},
    'Bangladesh': {'symbol': '৳', 'name': 'BDT'},
    'Sri Lanka': {'symbol': 'Rs', 'name': 'LKR'},
    'Nepal': {'symbol': 'Rs', 'name': 'NPR'},
    'Myanmar': {'symbol': 'K', 'name': 'MMK'},
    'Cambodia': {'symbol': '៛', 'name': 'KHR'},
    'Laos': {'symbol': '₭', 'name': 'LAK'},
    'Afghanistan': {'symbol': '؋', 'name': 'AFN'},
    
    # Pacific Islands
    'Fiji': {'symbol': 'FJ$', 'name': 'FJD'},
    'Papua New Guinea': {'symbol': 'K', 'name': 'PGK'},
    'Samoa': {'symbol': 'WS$', 'name': 'WST'},
    'Tonga': {'symbol': 'T$', 'name': 'TOP'},
    
    # Caribbean
    'Jamaica': {'symbol': 'J$', 'name': 'JMD'},
    'Trinidad and Tobago': {'symbol': 'TT$', 'name': 'TTD'},
    'Barbados': {'symbol': 'Bds$', 'name': 'BBD'},
    'Bahamas': {'symbol': 'B$', 'name': 'BSD'},
    
    # Central Asia
    'Kazakhstan': {'symbol': '₸', 'name': 'KZT'},
    'Uzbekistan': {'symbol': 'сум', 'name': 'UZS'},
    'Turkmenistan': {'symbol': 'TMT', 'name': 'TMT'},
    'Kyrgyzstan': {'symbol': 'сом', 'name': 'KGS'},
    'Tajikistan': {'symbol': 'ТЈС', 'name': 'TJS'},
    
    # Global/Multi-Regional
    'Global USD': {'symbol': '$', 'name': 'USD'},
    'Global EUR': {'symbol': '€', 'name': 'EUR'},
    'Multi-Regional': {'symbol': '$', 'name': 'USD'}
}

def safe_numeric_filter(df, column, min_value=0):
    """Safely filter numeric columns, handling mixed data types"""
    try:
        # Convert to numeric, invalid values become NaN
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        # Create boolean mask for valid values above minimum
        mask = (numeric_series > min_value) & (numeric_series.notna())
        return df[mask]
    except Exception as e:
        st.error(f"Error filtering {column}: {str(e)}")
        return df

def safe_numeric_conversion(series):
    """Safely convert a series to numeric values"""
    try:
        return pd.to_numeric(series, errors='coerce')
    except:
        return series

def get_currency_info(region):
    """Get currency symbol and name for a region"""
    return REGION_CURRENCIES.get(region, {'symbol': '

def display(df):
    """Enhanced LOT Size Optimization Module"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management with regional support.")
    
    # Add methodology expander (enhances original's educational value)
    with st.expander("📚 EOQ Methodology & Assumptions", expanded=False):
        st.markdown("""
        **Economic Order Quantity (EOQ) Formula:** EOQ = √((2 × Annual Demand × Ordering Cost) / Holding Cost per Unit)
        
        **Key Assumptions:**
        - Constant demand rate throughout the year
        - Fixed ordering cost per order
        - Fixed holding cost per unit per year
        - No stockouts or backorders
        - Instant replenishment (zero lead time)
        
        **Cost Components:**
        - **Ordering Cost:** Cost incurred each time an order is placed
        - **Holding Cost:** Cost of storing one unit for one year (storage, insurance, obsolescence, etc.)
        - **Total Cost:** Sum of annual ordering and holding costs
        """)
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("This module requires: Item, Unit Price, Qty Delivered, and optionally Region columns")
        return
    
    # Clean data using safe numeric filtering
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    
    # Show initial data info for debugging
    st.info(f"📊 Initial data: {len(df)} rows → After removing nulls: {len(df_clean)} rows")
    
    # Convert columns to numeric safely
    df_clean['Unit Price'] = safe_numeric_conversion(df_clean['Unit Price'])
    df_clean['Qty Delivered'] = safe_numeric_conversion(df_clean['Qty Delivered'])
    
    # Remove rows with NaN values (conversion failures)
    rows_before_nan_removal = len(df_clean)
    df_clean = df_clean.dropna(subset=['Unit Price', 'Qty Delivered'])
    st.info(f"🔄 After numeric conversion: {rows_before_nan_removal} rows → After removing NaN: {len(df_clean)} rows")
    
    # Use safe filtering to avoid string/int comparison errors
    df_clean = safe_numeric_filter(df_clean, 'Unit Price', 0)
    df_clean = safe_numeric_filter(df_clean, 'Qty Delivered', 0)
    st.info(f"✅ Final clean data: {len(df_clean)} rows")
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        st.info("**Data Requirements:**")
        st.info("- 'Unit Price' must be positive numbers")
        st.info("- 'Qty Delivered' must be positive numbers") 
        st.info("- Remove any text values from numeric columns")
        
        # Show data sample for debugging
        if len(df) > 0:
            st.subheader("📋 Data Sample (First 5 rows)")
            st.dataframe(df.head())
        return
    
    # Region filter (if Region column exists)
    if 'Region' in df_clean.columns:
        st.sidebar.header("🌍 Regional Settings")
        available_regions = sorted(df_clean['Region'].unique())
        
        # Group regions by continent for better UX
        region_groups = {
            '🕌 Middle East & GCC': ['Saudi Arabia', 'UAE', 'Kuwait', 'Qatar', 'Bahrain', 'Oman', 'Iraq', 'Iran', 'Israel', 'Turkey'],
            '🏺 North Africa & Levant': ['Egypt', 'Jordan', 'Lebanon', 'Morocco', 'Tunisia', 'Algeria', 'Libya', 'Sudan'],
            '🌍 Americas': ['USA', 'Canada', 'Brazil', 'Mexico', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela', 'Ecuador', 'Uruguay'],
            '🇪🇺 Europe': ['United Kingdom', 'European Union', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Switzerland', 'Norway', 'Sweden', 'Denmark', 'Russia', 'Poland', 'Czech Republic', 'Hungary', 'Romania', 'Bulgaria', 'Ukraine'],
            '🌏 Asia Pacific': ['China', 'Japan', 'South Korea', 'India', 'Singapore', 'Hong Kong', 'Taiwan', 'Thailand', 'Malaysia', 'Indonesia', 'Philippines', 'Vietnam', 'Australia', 'New Zealand', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Myanmar'],
            '🌍 Africa': ['South Africa', 'Nigeria', 'Kenya', 'Ghana', 'Ethiopia', 'Tanzania', 'Uganda', 'Zambia', 'Botswana', 'Namibia'],
            '🌐 Global': ['Global USD', 'Global EUR', 'Multi-Regional']
        }
        
        # Display regions in organized groups if they exist in data
        organized_regions = []
        for group_name, group_regions in region_groups.items():
            group_available = [r for r in group_regions if r in available_regions]
            if group_available:
                st.sidebar.markdown(f"**{group_name}**")
                organized_regions.extend(group_available)
        
        # Add any remaining regions not in groups
        remaining_regions = [r for r in available_regions if r not in organized_regions]
        if remaining_regions:
            st.sidebar.markdown("**🏷️ Other**")
            organized_regions.extend(remaining_regions)
        
        selected_region = st.sidebar.selectbox("Select Region", available_regions)
        df_filtered = df_clean[df_clean['Region'] == selected_region]
        
        # Currency information
        currency_info = get_currency_info(selected_region)
        currency_symbol = currency_info['symbol']
        currency_name = currency_info['name']
        
        st.sidebar.success(f"**Currency:** {currency_name} ({currency_symbol})")
        
        # Additional currency info
        with st.sidebar.expander("💱 Currency Details"):
            st.markdown(f"""
            **Region:** {selected_region}  
            **Currency Code:** {currency_name}  
            **Symbol:** {currency_symbol}  
            **Records:** {len(df_filtered):,}
            """)
    else:
        # Manual region/currency selection when no Region column
        st.sidebar.header("🌍 Currency Settings")
        st.sidebar.info("💡 Add 'Region' column to your data for automatic regional filtering")
        
        # Manual currency selection
        currency_options = list(REGION_CURRENCIES.keys())
        selected_region = st.sidebar.selectbox("Select Currency Region", currency_options, index=currency_options.index('USA') if 'USA' in currency_options else 0)
        df_filtered = df_clean
        
        currency_info = get_currency_info(selected_region)
        currency_symbol = currency_info['symbol']
        currency_name = currency_info['name']
        
        st.sidebar.success(f"**Currency:** {currency_name} ({currency_symbol})")
    
    if len(df_filtered) == 0:
        st.warning(f"No data found for region: {selected_region}")
        return
    
    # Display current region info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📍 **Region:** {selected_region}")
    with col2:
        st.info(f"💰 **Currency:** {currency_name} ({currency_symbol})")
    with col3:
        st.info(f"📦 **Items:** {len(df_filtered):,} records")
    
    # Data quality info
    with st.expander("📊 Data Quality Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            unique_items = df_filtered['Item'].nunique()
            try:
                # Use safe numeric conversion for display
                numeric_prices = safe_numeric_conversion(df_filtered['Unit Price'])
                avg_unit_price = float(numeric_prices.mean())
                st.metric("Unique Items", unique_items)
                st.metric("Avg Unit Price", f"{currency_symbol}{avg_unit_price:.2f}")
            except:
                st.metric("Unique Items", unique_items)
                st.warning("Unit Price contains non-numeric data")
                
        with col2:
            try:
                # Use safe numeric conversion for display
                numeric_qty = safe_numeric_conversion(df_filtered['Qty Delivered'])
                total_qty = float(numeric_qty.sum())
                avg_qty = float(numeric_qty.mean())
                st.metric("Total Quantity", f"{total_qty:,.0f}")
                st.metric("Avg Order Size", f"{avg_qty:.0f}")
            except:
                st.warning("Qty Delivered contains non-numeric data")
        
        # Debug info
        st.subheader("🔍 Debug Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Unit Price Data Types:**")
            try:
                price_types = df_filtered['Unit Price'].apply(type).value_counts()
                st.write(price_types)
            except:
                st.write("Error analyzing Unit Price types")
        with col2:
            st.write("**Qty Delivered Data Types:**")
            try:
                qty_types = df_filtered['Qty Delivered'].apply(type).value_counts()
                st.write(qty_types)
            except:
                st.write("Error analyzing Qty Delivered types")
            
        # Show sample of problematic data
        try:
            non_numeric_prices = df_filtered[pd.to_numeric(df_filtered['Unit Price'], errors='coerce').isna()]
            non_numeric_qty = df_filtered[pd.to_numeric(df_filtered['Qty Delivered'], errors='coerce').isna()]
            
            if len(non_numeric_prices) > 0:
                st.warning(f"Found {len(non_numeric_prices)} rows with non-numeric Unit Price")
                st.write("Sample problematic Unit Price values:")
                st.write(non_numeric_prices[['Item', 'Unit Price']].head())
                
            if len(non_numeric_qty) > 0:
                st.warning(f"Found {len(non_numeric_qty)} rows with non-numeric Qty Delivered")
                st.write("Sample problematic Qty Delivered values:")
                st.write(non_numeric_qty[['Item', 'Qty Delivered']].head())
        except Exception as e:
            st.write(f"Error in debug analysis: {str(e)}")
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 EOQ Analysis", "💰 Cost Optimization"])
    
    with tab1:
        st.subheader("📊 Economic Order Quantity Analysis")
        
        # Enhanced Parameters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            holding_cost_type = st.radio("Holding Cost Type", ["Percentage (%)", "Fixed Amount"])
        
        with col2:
            if holding_cost_type == "Percentage (%)":
                holding_cost_rate = st.slider("Holding Cost Rate (%)", 5, 30, 15) / 100
                holding_cost_fixed = None
            else:
                holding_cost_fixed = st.number_input(f"Holding Cost ({currency_symbol})", 0.1, 50.0, 2.5, step=0.1)
                holding_cost_rate = None
        
        with col3:
            ordering_cost = st.number_input(f"Ordering Cost ({currency_symbol})", 50, 500, 100)
        
        with col4:
            working_days = st.number_input("Working Days/Year", 200, 365, 250)
            st.caption("ℹ️ For future lead time calculations")
        
        # Item selection
        items = sorted(df_filtered['Item'].unique())
        selected_item = st.selectbox("Select Item for EOQ Analysis", items)
        
        if selected_item:
            item_data = df_filtered[df_filtered['Item'] == selected_item]
            
            # Calculate demand and costs with proper validation
            try:
                annual_demand = float(item_data['Qty Delivered'].sum())
                avg_unit_cost = float(item_data['Unit Price'].mean())
                
                # Validate that we have positive numeric values
                if annual_demand <= 0 or avg_unit_cost <= 0:
                    st.error("Selected item has invalid data (zero or negative values)")
                    st.info(f"Annual Demand: {annual_demand}, Average Unit Cost: {avg_unit_cost}")
                    return
                
            except (ValueError, TypeError) as e:
                st.error(f"Error processing item data: {str(e)}")
                st.info("Please ensure Unit Price and Qty Delivered contain only numeric values")
                return
            
            # Calculate holding cost
            try:
                if holding_cost_type == "Percentage (%)":
                    holding_cost = float(avg_unit_cost * holding_cost_rate)
                    holding_cost_display = f"{holding_cost_rate*100:.1f}% of unit cost"
                else:
                    holding_cost = float(holding_cost_fixed)
                    holding_cost_display = f"{currency_symbol}{holding_cost_fixed:.2f} per unit"
                
                # Validate holding cost
                if holding_cost <= 0:
                    st.error("Holding cost must be positive")
                    return
                    
            except (ValueError, TypeError) as e:
                st.error(f"Error calculating holding cost: {str(e)}")
                return
            
            # EOQ calculation with comprehensive validation
            if annual_demand > 0 and holding_cost > 0:
                try:
                    # Ensure all inputs are properly converted to float
                    annual_demand = float(annual_demand)
                    ordering_cost = float(ordering_cost)
                    holding_cost = float(holding_cost)
                    
                    # Calculate EOQ
                    eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                    eoq = float(eoq)  # Ensure it's a float
                    
                    # Current average order size
                    current_avg_order = float(item_data['Qty Delivered'].mean())
                    
                    # Validate all calculated values
                    if not all(isinstance(x, (int, float)) and x > 0 for x in [eoq, current_avg_order]):
                        st.error("Invalid calculation results. Please check your data.")
                        return
                    
                    # Cost calculation functions
                    def ordering_cost_func(order_qty):
                        order_qty = float(order_qty)
                        if order_qty <= 0:
                            return float('inf')
                        return float((annual_demand / order_qty) * ordering_cost)
                    
                    def holding_cost_func(order_qty):
                        order_qty = float(order_qty)
                        return float((order_qty / 2) * holding_cost)
                    
                    def total_cost(order_qty):
                        order_qty = float(order_qty)
                        if order_qty <= 0:
                            return float('inf')
                        return float(ordering_cost_func(order_qty) + holding_cost_func(order_qty))
                    
                    eoq_cost = total_cost(eoq)
                    current_cost = total_cost(current_avg_order)
                    potential_savings = current_cost - eoq_cost
                    
                except Exception as e:
                    st.error(f"Error in EOQ calculation: {str(e)}")
                    st.info("Please check your data values and parameters.")
                    st.info(f"Debug info - Annual Demand: {annual_demand}, Holding Cost: {holding_cost}, Ordering Cost: {ordering_cost}")
                    return
            else:
                st.error("Invalid input values for EOQ calculation")
                st.info(f"Annual Demand: {annual_demand}, Holding Cost: {holding_cost}")
                return
                
                # Display results
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Annual Demand", f"{annual_demand:,.0f}")
                with col2:
                    st.metric("Optimal Order Qty (EOQ)", f"{eoq:.0f}")
                with col3:
                    st.metric("Current Avg Order", f"{current_avg_order:.0f}")
                with col4:
                    st.metric("Potential Savings", f"{currency_symbol}{potential_savings:,.0f}")
                with col5:
                    st.metric("Holding Cost", holding_cost_display)
                
                # Additional insights (similar to original's comprehensive approach)
                col1, col2, col3 = st.columns(3)
                with col1:
                    orders_per_year_eoq = annual_demand / eoq if eoq > 0 else 0
                    st.metric("Orders/Year (EOQ)", f"{orders_per_year_eoq:.1f}")
                with col2:
                    orders_per_year_current = annual_demand / current_avg_order if current_avg_order > 0 else 0
                    st.metric("Orders/Year (Current)", f"{orders_per_year_current:.1f}")
                with col3:
                    cycle_time_eoq = working_days / orders_per_year_eoq if orders_per_year_eoq > 0 else 0
                    st.metric("Days Between Orders (EOQ)", f"{cycle_time_eoq:.0f}")
                
                # Enhanced EOQ curve with cost breakdown
                try:
                    # Create order size range with proper numeric handling
                    start_range = max(10.0, float(eoq * 0.1))
                    end_range = float(eoq * 3)
                    step_size = max(1.0, float(eoq * 0.1))
                    
                    order_sizes = np.arange(start_range, end_range, step_size)
                    order_sizes = [float(x) for x in order_sizes]  # Ensure all are floats
                    
                    total_costs = [total_cost(q) for q in order_sizes]
                    ordering_costs = [ordering_cost_func(q) for q in order_sizes]
                    holding_costs = [holding_cost_func(q) for q in order_sizes]
                    
                except Exception as e:
                    st.error(f"Error generating chart data: {str(e)}")
                    st.info("Chart generation failed, but calculations above are still valid.")
                    return
                
                fig = go.Figure()
                
                # Add cost breakdown lines
                fig.add_trace(go.Scatter(
                    x=order_sizes, y=ordering_costs, 
                    name='Ordering Costs', 
                    line=dict(width=2, color='blue', dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=order_sizes, y=holding_costs, 
                    name='Holding Costs', 
                    line=dict(width=2, color='green', dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=order_sizes, y=total_costs, 
                    name='Total Cost', 
                    line=dict(width=3, color='red')
                ))
                
                # Add vertical lines for EOQ and current order size
                fig.add_vline(
                    x=eoq, line_dash="dash", line_color="red", line_width=2,
                    annotation_text=f"EOQ: {eoq:.0f}"
                )
                fig.add_vline(
                    x=current_avg_order, line_dash="dash", line_color="orange", line_width=2,
                    annotation_text=f"Current: {current_avg_order:.0f}"
                )
                
                # Add cost points
                fig.add_trace(go.Scatter(
                    x=[eoq], y=[eoq_cost],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name=f'EOQ Cost: {currency_symbol}{eoq_cost:,.0f}'
                ))
                fig.add_trace(go.Scatter(
                    x=[current_avg_order], y=[current_cost],
                    mode='markers',
                    marker=dict(size=10, color='orange'),
                    name=f'Current Cost: {currency_symbol}{current_cost:,.0f}'
                ))
                
                fig.update_layout(
                    title=f"EOQ Cost Analysis - {selected_item} ({selected_region})",
                    xaxis_title="Order Quantity",
                    yaxis_title=f"Annual Cost ({currency_symbol})",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cost breakdown table
                st.subheader("💡 Cost Breakdown Comparison")
                breakdown_data = {
                    'Scenario': ['Current Practice', 'Optimal EOQ'],
                    'Order Quantity': [current_avg_order, eoq],
                    f'Ordering Cost ({currency_symbol})': [ordering_cost_func(current_avg_order), ordering_cost_func(eoq)],
                    f'Holding Cost ({currency_symbol})': [holding_cost_func(current_avg_order), holding_cost_func(eoq)],
                    f'Total Cost ({currency_symbol})': [current_cost, eoq_cost]
                }
                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(
                    breakdown_df.style.format({
                        'Order Quantity': '{:.0f}',
                        f'Ordering Cost ({currency_symbol})': '{:,.0f}',
                        f'Holding Cost ({currency_symbol})': '{:,.0f}',
                        f'Total Cost ({currency_symbol})': '{:,.0f}'
                    }),
                    use_container_width=True
                )
    
    with tab2:
        st.subheader(f"💰 Portfolio Cost Optimization - {selected_region}")
        
        # Parameters from Tab 1 are automatically used here
        try:
            param_info = f"Holding Cost: {holding_cost_display}, Ordering Cost: {currency_symbol}{ordering_cost}, Working Days: {working_days}"
        except:
            param_info = "Parameters not set - please configure in EOQ Analysis tab first"
        
        st.info(f"📋 Using parameters from EOQ Analysis tab: {param_info}")
        
        # Use the same parameters from tab1 - maintain original behavior
        if 'holding_cost_type' not in locals():
            holding_cost_type = "Percentage (%)"
            holding_cost_rate = 0.15
            holding_cost_fixed = None
            ordering_cost = 100
        
        # Calculate EOQ for all items in the filtered region
        optimization_results = []
        
        for item in df_filtered['Item'].unique():
            item_data = df_filtered[df_filtered['Item'] == item]
            
            if len(item_data) >= 3:  # Need minimum data points for statistical reliability
                try:
                    annual_demand = float(item_data['Qty Delivered'].sum())
                    avg_unit_cost = float(item_data['Unit Price'].mean())
                    current_avg_order = float(item_data['Qty Delivered'].mean())
                    
                    # Skip items with invalid data
                    if annual_demand <= 0 or avg_unit_cost <= 0 or current_avg_order <= 0:
                        continue
                    
                    # Calculate holding cost based on type
                    if holding_cost_type == "Percentage (%)":
                        holding_cost = float(avg_unit_cost * holding_cost_rate)
                    else:
                        holding_cost = float(holding_cost_fixed)
                    
                except (ValueError, TypeError, AttributeError):
                    # Skip items with data conversion issues
                    continue
                
                if annual_demand > 0 and holding_cost > 0:
                    try:
                        eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                        
                        def total_cost(order_qty):
                            if order_qty <= 0:
                                return float('inf')
                            ordering_cost_total = (annual_demand / order_qty) * ordering_cost
                            holding_cost_total = (order_qty / 2) * holding_cost
                            return ordering_cost_total + holding_cost_total
                        
                        eoq_cost = total_cost(eoq)
                        current_cost = total_cost(current_avg_order)
                        potential_savings = current_cost - eoq_cost
                        
                        optimization_results.append({
                            'Item': item,
                            'Annual Demand': annual_demand,
                            'Current Avg Order': current_avg_order,
                            'Optimal EOQ': eoq,
                            'Current Cost': current_cost,
                            'EOQ Cost': eoq_cost,
                            'Potential Savings': potential_savings,
                            'Savings %': (potential_savings / current_cost * 100) if current_cost > 0 else 0
                        })
                    except Exception as e:
                        # Skip items with calculation errors but continue processing others
                        continue
        
        if optimization_results:
            results_df = pd.DataFrame(optimization_results)
            results_df = results_df.sort_values('Potential Savings', ascending=False)
            
            # Summary metrics
            total_savings = results_df['Potential Savings'].sum()
            avg_savings_pct = results_df['Savings %'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Potential Savings", f"{currency_symbol}{total_savings:,.0f}")
            with col2:
                st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
            with col3:
                st.metric("Items Analyzed", len(results_df))
            with col4:
                st.metric("Region", selected_region)
            
            # Top opportunities
            st.subheader("🎯 Top Optimization Opportunities")
            
            display_df = results_df.head(15)[['Item', 'Current Avg Order', 'Optimal EOQ', 'Potential Savings', 'Savings %']]
            
            # Format the dataframe
            formatted_df = display_df.copy()
            formatted_df['Current Avg Order'] = formatted_df['Current Avg Order'].apply(lambda x: f"{x:.0f}")
            formatted_df['Optimal EOQ'] = formatted_df['Optimal EOQ'].apply(lambda x: f"{x:.0f}")
            formatted_df['Potential Savings'] = formatted_df['Potential Savings'].apply(lambda x: f"{currency_symbol}{x:,.0f}")
            formatted_df['Savings %'] = formatted_df['Savings %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                results_df.head(10), 
                x='Potential Savings', 
                y='Item',
                orientation='h',
                title=f"Top 10 Items by Savings Potential - {selected_region}",
                labels={'Potential Savings': f'Potential Savings ({currency_symbol})'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export option
            if st.button("📊 Download Optimization Report"):
                export_df = results_df.copy()
                export_df['Region'] = selected_region
                export_df['Currency'] = currency_name
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"eoq_optimization_{selected_region.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        else:
            st.info(f"Need more data to perform EOQ optimization for {selected_region}.")

if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced LOT Size Optimization", layout="wide")
    
    # Enhanced sample data with regions - ensure all numeric columns are proper numbers
    regions = ['Saudi Arabia', 'UAE', 'Kuwait', 'Qatar', 'Bahrain', 'Egypt']
    np.random.seed(42)  # For reproducible results
    
    sample_data = {
        'Item': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D', 'Product E'], 150),
        'Unit Price': np.round(np.random.uniform(5.0, 50.0, 150), 2),  # Ensure float values
        'Qty Delivered': np.random.randint(10, 200, 150).astype(float),  # Convert to float
        'Region': np.random.choice(regions, 150)
    }
    df = pd.DataFrame(sample_data)
    
    # Ensure data types are correct
    df['Unit Price'] = df['Unit Price'].astype(float)
    df['Qty Delivered'] = df['Qty Delivered'].astype(float)
    
    display(df), 'name': 'USD'})

def display(df):
    """Enhanced LOT Size Optimization Module"""
    st.header("📦 LOT Size Optimization")
    st.markdown("Economic Order Quantity (EOQ) analysis for optimal inventory management with regional support.")
    
    # Add methodology expander (enhances original's educational value)
    with st.expander("📚 EOQ Methodology & Assumptions", expanded=False):
        st.markdown("""
        **Economic Order Quantity (EOQ) Formula:** EOQ = √((2 × Annual Demand × Ordering Cost) / Holding Cost per Unit)
        
        **Key Assumptions:**
        - Constant demand rate throughout the year
        - Fixed ordering cost per order
        - Fixed holding cost per unit per year
        - No stockouts or backorders
        - Instant replenishment (zero lead time)
        
        **Cost Components:**
        - **Ordering Cost:** Cost incurred each time an order is placed
        - **Holding Cost:** Cost of storing one unit for one year (storage, insurance, obsolescence, etc.)
        - **Total Cost:** Sum of annual ordering and holding costs
        """)
    
    # Basic data validation
    required_columns = ['Item', 'Unit Price', 'Qty Delivered']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("This module requires: Item, Unit Price, Qty Delivered, and optionally Region columns")
        return
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=required_columns)
    
    # Convert to numeric and handle errors
    df_clean['Unit Price'] = pd.to_numeric(df_clean['Unit Price'], errors='coerce')
    df_clean['Qty Delivered'] = pd.to_numeric(df_clean['Qty Delivered'], errors='coerce')
    
    # Remove rows with invalid numeric data
    df_clean = df_clean.dropna(subset=['Unit Price', 'Qty Delivered'])
    
    # Filter for positive values
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        st.info("**Data Requirements:**")
        st.info("- 'Unit Price' must be positive numbers")
        st.info("- 'Qty Delivered' must be positive numbers") 
        st.info("- Remove any text values from numeric columns")
        
        # Show data sample for debugging
        if len(df) > 0:
            st.subheader("📋 Data Sample (First 5 rows)")
            st.dataframe(df.head())
        return
    
    # Region filter (if Region column exists)
    if 'Region' in df_clean.columns:
        st.sidebar.header("🌍 Regional Settings")
        available_regions = sorted(df_clean['Region'].unique())
        
        # Group regions by continent for better UX
        region_groups = {
            '🕌 Middle East & GCC': ['Saudi Arabia', 'UAE', 'Kuwait', 'Qatar', 'Bahrain', 'Oman', 'Iraq', 'Iran', 'Israel', 'Turkey'],
            '🏺 North Africa & Levant': ['Egypt', 'Jordan', 'Lebanon', 'Morocco', 'Tunisia', 'Algeria', 'Libya', 'Sudan'],
            '🌍 Americas': ['USA', 'Canada', 'Brazil', 'Mexico', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela', 'Ecuador', 'Uruguay'],
            '🇪🇺 Europe': ['United Kingdom', 'European Union', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Switzerland', 'Norway', 'Sweden', 'Denmark', 'Russia', 'Poland', 'Czech Republic', 'Hungary', 'Romania', 'Bulgaria', 'Ukraine'],
            '🌏 Asia Pacific': ['China', 'Japan', 'South Korea', 'India', 'Singapore', 'Hong Kong', 'Taiwan', 'Thailand', 'Malaysia', 'Indonesia', 'Philippines', 'Vietnam', 'Australia', 'New Zealand', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Myanmar'],
            '🌍 Africa': ['South Africa', 'Nigeria', 'Kenya', 'Ghana', 'Ethiopia', 'Tanzania', 'Uganda', 'Zambia', 'Botswana', 'Namibia'],
            '🌐 Global': ['Global USD', 'Global EUR', 'Multi-Regional']
        }
        
        # Display regions in organized groups if they exist in data
        organized_regions = []
        for group_name, group_regions in region_groups.items():
            group_available = [r for r in group_regions if r in available_regions]
            if group_available:
                st.sidebar.markdown(f"**{group_name}**")
                organized_regions.extend(group_available)
        
        # Add any remaining regions not in groups
        remaining_regions = [r for r in available_regions if r not in organized_regions]
        if remaining_regions:
            st.sidebar.markdown("**🏷️ Other**")
            organized_regions.extend(remaining_regions)
        
        selected_region = st.sidebar.selectbox("Select Region", available_regions)
        df_filtered = df_clean[df_clean['Region'] == selected_region]
        
        # Currency information
        currency_info = get_currency_info(selected_region)
        currency_symbol = currency_info['symbol']
        currency_name = currency_info['name']
        
        st.sidebar.success(f"**Currency:** {currency_name} ({currency_symbol})")
        
        # Additional currency info
        with st.sidebar.expander("💱 Currency Details"):
            st.markdown(f"""
            **Region:** {selected_region}  
            **Currency Code:** {currency_name}  
            **Symbol:** {currency_symbol}  
            **Records:** {len(df_filtered):,}
            """)
    else:
        # Manual region/currency selection when no Region column
        st.sidebar.header("🌍 Currency Settings")
        st.sidebar.info("💡 Add 'Region' column to your data for automatic regional filtering")
        
        # Manual currency selection
        currency_options = list(REGION_CURRENCIES.keys())
        selected_region = st.sidebar.selectbox("Select Currency Region", currency_options, index=currency_options.index('USA') if 'USA' in currency_options else 0)
        df_filtered = df_clean
        
        currency_info = get_currency_info(selected_region)
        currency_symbol = currency_info['symbol']
        currency_name = currency_info['name']
        
        st.sidebar.success(f"**Currency:** {currency_name} ({currency_symbol})")
    
    if len(df_filtered) == 0:
        st.warning(f"No data found for region: {selected_region}")
        return
    
    # Display current region info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📍 **Region:** {selected_region}")
    with col2:
        st.info(f"💰 **Currency:** {currency_name} ({currency_symbol})")
    with col3:
        st.info(f"📦 **Items:** {len(df_filtered):,} records")
    
    # Data quality info
    with st.expander("📊 Data Quality Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            unique_items = df_filtered['Item'].nunique()
            try:
                avg_unit_price = float(df_filtered['Unit Price'].mean())
                st.metric("Unique Items", unique_items)
                st.metric("Avg Unit Price", f"{currency_symbol}{avg_unit_price:.2f}")
            except:
                st.metric("Unique Items", unique_items)
                st.warning("Unit Price contains non-numeric data")
                
        with col2:
            try:
                total_qty = float(df_filtered['Qty Delivered'].sum())
                avg_qty = float(df_filtered['Qty Delivered'].mean())
                st.metric("Total Quantity", f"{total_qty:,.0f}")
                st.metric("Avg Order Size", f"{avg_qty:.0f}")
            except:
                st.warning("Qty Delivered contains non-numeric data")
        
        # Debug info
        st.subheader("🔍 Debug Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Unit Price Data Types:**")
            price_types = df_filtered['Unit Price'].apply(type).value_counts()
            st.write(price_types)
        with col2:
            st.write("**Qty Delivered Data Types:**")
            qty_types = df_filtered['Qty Delivered'].apply(type).value_counts()
            st.write(qty_types)
            
        # Show sample of problematic data
        non_numeric_prices = df_filtered[pd.to_numeric(df_filtered['Unit Price'], errors='coerce').isna()]
        non_numeric_qty = df_filtered[pd.to_numeric(df_filtered['Qty Delivered'], errors='coerce').isna()]
        
        if len(non_numeric_prices) > 0:
            st.warning(f"Found {len(non_numeric_prices)} rows with non-numeric Unit Price")
            st.write("Sample problematic Unit Price values:")
            st.write(non_numeric_prices[['Item', 'Unit Price']].head())
            
        if len(non_numeric_qty) > 0:
            st.warning(f"Found {len(non_numeric_qty)} rows with non-numeric Qty Delivered")
            st.write("Sample problematic Qty Delivered values:")
            st.write(non_numeric_qty[['Item', 'Qty Delivered']].head())
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 EOQ Analysis", "💰 Cost Optimization"])
    
    with tab1:
        st.subheader("📊 Economic Order Quantity Analysis")
        
        # Enhanced Parameters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            holding_cost_type = st.radio("Holding Cost Type", ["Percentage (%)", "Fixed Amount"])
        
        with col2:
            if holding_cost_type == "Percentage (%)":
                holding_cost_rate = st.slider("Holding Cost Rate (%)", 5, 30, 15) / 100
                holding_cost_fixed = None
            else:
                holding_cost_fixed = st.number_input(f"Holding Cost ({currency_symbol})", 0.1, 50.0, 2.5, step=0.1)
                holding_cost_rate = None
        
        with col3:
            ordering_cost = st.number_input(f"Ordering Cost ({currency_symbol})", 50, 500, 100)
        
        with col4:
            working_days = st.number_input("Working Days/Year", 200, 365, 250)
            st.caption("ℹ️ For future lead time calculations")
        
        # Item selection
        items = sorted(df_filtered['Item'].unique())
        selected_item = st.selectbox("Select Item for EOQ Analysis", items)
        
        if selected_item:
            item_data = df_filtered[df_filtered['Item'] == selected_item]
            
            # Calculate demand and costs with proper validation
            try:
                annual_demand = float(item_data['Qty Delivered'].sum())
                avg_unit_cost = float(item_data['Unit Price'].mean())
                
                # Validate that we have positive numeric values
                if annual_demand <= 0 or avg_unit_cost <= 0:
                    st.error("Selected item has invalid data (zero or negative values)")
                    st.info(f"Annual Demand: {annual_demand}, Average Unit Cost: {avg_unit_cost}")
                    return
                
            except (ValueError, TypeError) as e:
                st.error(f"Error processing item data: {str(e)}")
                st.info("Please ensure Unit Price and Qty Delivered contain only numeric values")
                return
            
            # Calculate holding cost
            try:
                if holding_cost_type == "Percentage (%)":
                    holding_cost = float(avg_unit_cost * holding_cost_rate)
                    holding_cost_display = f"{holding_cost_rate*100:.1f}% of unit cost"
                else:
                    holding_cost = float(holding_cost_fixed)
                    holding_cost_display = f"{currency_symbol}{holding_cost_fixed:.2f} per unit"
                
                # Validate holding cost
                if holding_cost <= 0:
                    st.error("Holding cost must be positive")
                    return
                    
            except (ValueError, TypeError) as e:
                st.error(f"Error calculating holding cost: {str(e)}")
                return
            
            # EOQ calculation with comprehensive validation
            if annual_demand > 0 and holding_cost > 0:
                try:
                    # Ensure all inputs are properly converted to float
                    annual_demand = float(annual_demand)
                    ordering_cost = float(ordering_cost)
                    holding_cost = float(holding_cost)
                    
                    # Calculate EOQ
                    eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                    eoq = float(eoq)  # Ensure it's a float
                    
                    # Current average order size
                    current_avg_order = float(item_data['Qty Delivered'].mean())
                    
                    # Validate all calculated values
                    if not all(isinstance(x, (int, float)) and x > 0 for x in [eoq, current_avg_order]):
                        st.error("Invalid calculation results. Please check your data.")
                        return
                    
                    # Cost calculation functions
                    def ordering_cost_func(order_qty):
                        order_qty = float(order_qty)
                        if order_qty <= 0:
                            return float('inf')
                        return float((annual_demand / order_qty) * ordering_cost)
                    
                    def holding_cost_func(order_qty):
                        order_qty = float(order_qty)
                        return float((order_qty / 2) * holding_cost)
                    
                    def total_cost(order_qty):
                        order_qty = float(order_qty)
                        if order_qty <= 0:
                            return float('inf')
                        return float(ordering_cost_func(order_qty) + holding_cost_func(order_qty))
                    
                    eoq_cost = total_cost(eoq)
                    current_cost = total_cost(current_avg_order)
                    potential_savings = current_cost - eoq_cost
                    
                except Exception as e:
                    st.error(f"Error in EOQ calculation: {str(e)}")
                    st.info("Please check your data values and parameters.")
                    st.info(f"Debug info - Annual Demand: {annual_demand}, Holding Cost: {holding_cost}, Ordering Cost: {ordering_cost}")
                    return
            else:
                st.error("Invalid input values for EOQ calculation")
                st.info(f"Annual Demand: {annual_demand}, Holding Cost: {holding_cost}")
                return
                
                # Display results
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Annual Demand", f"{annual_demand:,.0f}")
                with col2:
                    st.metric("Optimal Order Qty (EOQ)", f"{eoq:.0f}")
                with col3:
                    st.metric("Current Avg Order", f"{current_avg_order:.0f}")
                with col4:
                    st.metric("Potential Savings", f"{currency_symbol}{potential_savings:,.0f}")
                with col5:
                    st.metric("Holding Cost", holding_cost_display)
                
                # Additional insights (similar to original's comprehensive approach)
                col1, col2, col3 = st.columns(3)
                with col1:
                    orders_per_year_eoq = annual_demand / eoq if eoq > 0 else 0
                    st.metric("Orders/Year (EOQ)", f"{orders_per_year_eoq:.1f}")
                with col2:
                    orders_per_year_current = annual_demand / current_avg_order if current_avg_order > 0 else 0
                    st.metric("Orders/Year (Current)", f"{orders_per_year_current:.1f}")
                with col3:
                    cycle_time_eoq = working_days / orders_per_year_eoq if orders_per_year_eoq > 0 else 0
                    st.metric("Days Between Orders (EOQ)", f"{cycle_time_eoq:.0f}")
                
                # Enhanced EOQ curve with cost breakdown
                try:
                    # Create order size range with proper numeric handling
                    start_range = max(10.0, float(eoq * 0.1))
                    end_range = float(eoq * 3)
                    step_size = max(1.0, float(eoq * 0.1))
                    
                    order_sizes = np.arange(start_range, end_range, step_size)
                    order_sizes = [float(x) for x in order_sizes]  # Ensure all are floats
                    
                    total_costs = [total_cost(q) for q in order_sizes]
                    ordering_costs = [ordering_cost_func(q) for q in order_sizes]
                    holding_costs = [holding_cost_func(q) for q in order_sizes]
                    
                except Exception as e:
                    st.error(f"Error generating chart data: {str(e)}")
                    st.info("Chart generation failed, but calculations above are still valid.")
                    return
                
                fig = go.Figure()
                
                # Add cost breakdown lines
                fig.add_trace(go.Scatter(
                    x=order_sizes, y=ordering_costs, 
                    name='Ordering Costs', 
                    line=dict(width=2, color='blue', dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=order_sizes, y=holding_costs, 
                    name='Holding Costs', 
                    line=dict(width=2, color='green', dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=order_sizes, y=total_costs, 
                    name='Total Cost', 
                    line=dict(width=3, color='red')
                ))
                
                # Add vertical lines for EOQ and current order size
                fig.add_vline(
                    x=eoq, line_dash="dash", line_color="red", line_width=2,
                    annotation_text=f"EOQ: {eoq:.0f}"
                )
                fig.add_vline(
                    x=current_avg_order, line_dash="dash", line_color="orange", line_width=2,
                    annotation_text=f"Current: {current_avg_order:.0f}"
                )
                
                # Add cost points
                fig.add_trace(go.Scatter(
                    x=[eoq], y=[eoq_cost],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name=f'EOQ Cost: {currency_symbol}{eoq_cost:,.0f}'
                ))
                fig.add_trace(go.Scatter(
                    x=[current_avg_order], y=[current_cost],
                    mode='markers',
                    marker=dict(size=10, color='orange'),
                    name=f'Current Cost: {currency_symbol}{current_cost:,.0f}'
                ))
                
                fig.update_layout(
                    title=f"EOQ Cost Analysis - {selected_item} ({selected_region})",
                    xaxis_title="Order Quantity",
                    yaxis_title=f"Annual Cost ({currency_symbol})",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cost breakdown table
                st.subheader("💡 Cost Breakdown Comparison")
                breakdown_data = {
                    'Scenario': ['Current Practice', 'Optimal EOQ'],
                    'Order Quantity': [current_avg_order, eoq],
                    f'Ordering Cost ({currency_symbol})': [ordering_cost_func(current_avg_order), ordering_cost_func(eoq)],
                    f'Holding Cost ({currency_symbol})': [holding_cost_func(current_avg_order), holding_cost_func(eoq)],
                    f'Total Cost ({currency_symbol})': [current_cost, eoq_cost]
                }
                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(
                    breakdown_df.style.format({
                        'Order Quantity': '{:.0f}',
                        f'Ordering Cost ({currency_symbol})': '{:,.0f}',
                        f'Holding Cost ({currency_symbol})': '{:,.0f}',
                        f'Total Cost ({currency_symbol})': '{:,.0f}'
                    }),
                    use_container_width=True
                )
    
    with tab2:
        st.subheader(f"💰 Portfolio Cost Optimization - {selected_region}")
        
        # Parameters from Tab 1 are automatically used here
        try:
            param_info = f"Holding Cost: {holding_cost_display}, Ordering Cost: {currency_symbol}{ordering_cost}, Working Days: {working_days}"
        except:
            param_info = "Parameters not set - please configure in EOQ Analysis tab first"
        
        st.info(f"📋 Using parameters from EOQ Analysis tab: {param_info}")
        
        # Use the same parameters from tab1 - maintain original behavior
        if 'holding_cost_type' not in locals():
            holding_cost_type = "Percentage (%)"
            holding_cost_rate = 0.15
            holding_cost_fixed = None
            ordering_cost = 100
        
        # Calculate EOQ for all items in the filtered region
        optimization_results = []
        
        for item in df_filtered['Item'].unique():
            item_data = df_filtered[df_filtered['Item'] == item]
            
            if len(item_data) >= 3:  # Need minimum data points for statistical reliability
                try:
                    annual_demand = float(item_data['Qty Delivered'].sum())
                    avg_unit_cost = float(item_data['Unit Price'].mean())
                    current_avg_order = float(item_data['Qty Delivered'].mean())
                    
                    # Skip items with invalid data
                    if annual_demand <= 0 or avg_unit_cost <= 0 or current_avg_order <= 0:
                        continue
                    
                    # Calculate holding cost based on type
                    if holding_cost_type == "Percentage (%)":
                        holding_cost = float(avg_unit_cost * holding_cost_rate)
                    else:
                        holding_cost = float(holding_cost_fixed)
                    
                except (ValueError, TypeError, AttributeError):
                    # Skip items with data conversion issues
                    continue
                
                if annual_demand > 0 and holding_cost > 0:
                    try:
                        eoq = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                        
                        def total_cost(order_qty):
                            if order_qty <= 0:
                                return float('inf')
                            ordering_cost_total = (annual_demand / order_qty) * ordering_cost
                            holding_cost_total = (order_qty / 2) * holding_cost
                            return ordering_cost_total + holding_cost_total
                        
                        eoq_cost = total_cost(eoq)
                        current_cost = total_cost(current_avg_order)
                        potential_savings = current_cost - eoq_cost
                        
                        optimization_results.append({
                            'Item': item,
                            'Annual Demand': annual_demand,
                            'Current Avg Order': current_avg_order,
                            'Optimal EOQ': eoq,
                            'Current Cost': current_cost,
                            'EOQ Cost': eoq_cost,
                            'Potential Savings': potential_savings,
                            'Savings %': (potential_savings / current_cost * 100) if current_cost > 0 else 0
                        })
                    except Exception as e:
                        # Skip items with calculation errors but continue processing others
                        continue
        
        if optimization_results:
            results_df = pd.DataFrame(optimization_results)
            results_df = results_df.sort_values('Potential Savings', ascending=False)
            
            # Summary metrics
            total_savings = results_df['Potential Savings'].sum()
            avg_savings_pct = results_df['Savings %'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Potential Savings", f"{currency_symbol}{total_savings:,.0f}")
            with col2:
                st.metric("Average Savings %", f"{avg_savings_pct:.1f}%")
            with col3:
                st.metric("Items Analyzed", len(results_df))
            with col4:
                st.metric("Region", selected_region)
            
            # Top opportunities
            st.subheader("🎯 Top Optimization Opportunities")
            
            display_df = results_df.head(15)[['Item', 'Current Avg Order', 'Optimal EOQ', 'Potential Savings', 'Savings %']]
            
            # Format the dataframe
            formatted_df = display_df.copy()
            formatted_df['Current Avg Order'] = formatted_df['Current Avg Order'].apply(lambda x: f"{x:.0f}")
            formatted_df['Optimal EOQ'] = formatted_df['Optimal EOQ'].apply(lambda x: f"{x:.0f}")
            formatted_df['Potential Savings'] = formatted_df['Potential Savings'].apply(lambda x: f"{currency_symbol}{x:,.0f}")
            formatted_df['Savings %'] = formatted_df['Savings %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                results_df.head(10), 
                x='Potential Savings', 
                y='Item',
                orientation='h',
                title=f"Top 10 Items by Savings Potential - {selected_region}",
                labels={'Potential Savings': f'Potential Savings ({currency_symbol})'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export option
            if st.button("📊 Download Optimization Report"):
                export_df = results_df.copy()
                export_df['Region'] = selected_region
                export_df['Currency'] = currency_name
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"eoq_optimization_{selected_region.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        else:
            st.info(f"Need more data to perform EOQ optimization for {selected_region}.")

if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced LOT Size Optimization", layout="wide")
    
    # Enhanced sample data with regions - ensure all numeric columns are proper numbers
    regions = ['Saudi Arabia', 'UAE', 'Kuwait', 'Qatar', 'Bahrain', 'Egypt']
    np.random.seed(42)  # For reproducible results
    
    sample_data = {
        'Item': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D', 'Product E'], 150),
        'Unit Price': np.round(np.random.uniform(5.0, 50.0, 150), 2),  # Ensure float values
        'Qty Delivered': np.random.randint(10, 200, 150).astype(float),  # Convert to float
        'Region': np.random.choice(regions, 150)
    }
    df = pd.DataFrame(sample_data)
    
    # Ensure data types are correct
    df['Unit Price'] = df['Unit Price'].astype(float)
    df['Qty Delivered'] = df['Qty Delivered'].astype(float)
    
    display(df)
