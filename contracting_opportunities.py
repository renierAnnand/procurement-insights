import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def calculate_vendor_performance_score(vendor_data):
    """Calculate comprehensive vendor performance score with robust error handling"""
    try:
        metrics = {}
        
        # Ensure we have required columns
        if vendor_data.empty:
            return {
                'overall_score': 0,
                'volume_consistency': 0,
                'price_stability': 0,
                'lead_time_consistency': 0.5,
                'delivery_performance': 0.5,
                'quality_score': 0.5
            }
        
        # Volume consistency (coefficient of variation of monthly orders)
        try:
            if 'Creation Date' in vendor_data.columns:
                monthly_orders = vendor_data.groupby(vendor_data['Creation Date'].dt.to_period('M')).size()
                if len(monthly_orders) > 1:
                    mean_orders = monthly_orders.mean()
                    if mean_orders > 0:
                        metrics['volume_consistency'] = max(0, 1 - (monthly_orders.std() / mean_orders))
                    else:
                        metrics['volume_consistency'] = 0
                else:
                    metrics['volume_consistency'] = 0
            else:
                metrics['volume_consistency'] = 0
        except Exception:
            metrics['volume_consistency'] = 0
        
        # Price stability (1 - coefficient of variation of unit prices)
        try:
            if 'Unit Price' in vendor_data.columns:
                unit_prices = pd.to_numeric(vendor_data['Unit Price'], errors='coerce').dropna()
                if len(unit_prices) > 1 and unit_prices.mean() > 0:
                    price_cv = unit_prices.std() / unit_prices.mean()
                    metrics['price_stability'] = max(0, 1 - price_cv)
                else:
                    metrics['price_stability'] = 0.5
            else:
                metrics['price_stability'] = 0.5
        except Exception:
            metrics['price_stability'] = 0.5
        
        # Lead time consistency (if available)
        try:
            if 'lead_time_days' in vendor_data.columns:
                lead_times = pd.to_numeric(vendor_data['lead_time_days'], errors='coerce').dropna()
                if len(lead_times) > 1 and lead_times.mean() > 0:
                    lt_cv = lead_times.std() / lead_times.mean()
                    metrics['lead_time_consistency'] = max(0, 1 - lt_cv)
                else:
                    metrics['lead_time_consistency'] = 0.5
            else:
                metrics['lead_time_consistency'] = 0.5
        except Exception:
            metrics['lead_time_consistency'] = 0.5
        
        # Delivery performance (if available)
        try:
            if 'delivery_delay_days' in vendor_data.columns:
                delays = pd.to_numeric(vendor_data['delivery_delay_days'], errors='coerce').dropna()
                if len(delays) > 0:
                    on_time_rate = len(delays[delays <= 0]) / len(delays)
                    metrics['delivery_performance'] = on_time_rate
                else:
                    metrics['delivery_performance'] = 0.5
            else:
                metrics['delivery_performance'] = 0.5
        except Exception:
            metrics['delivery_performance'] = 0.5
        
        # Quality score (based on rejection rates if available) - FIXED SECTION
        try:
            # Check if both required columns exist
            if 'Qty Rejected' in vendor_data.columns and 'Qty Delivered' in vendor_data.columns:
                # Convert to numeric and handle NaN values
                qty_delivered = pd.to_numeric(vendor_data['Qty Delivered'], errors='coerce').fillna(0)
                qty_rejected = pd.to_numeric(vendor_data['Qty Rejected'], errors='coerce').fillna(0)
                
                total_delivered = qty_delivered.sum()
                total_rejected = qty_rejected.sum()
                
                if total_delivered > 0:
                    rejection_rate = total_rejected / (total_delivered + total_rejected)
                    metrics['quality_score'] = max(0, 1 - rejection_rate)
                else:
                    metrics['quality_score'] = 0.5
            else:
                # If columns don't exist, use default neutral score
                metrics['quality_score'] = 0.5
                if 'Qty Rejected' not in vendor_data.columns:
                    st.info("'Qty Rejected' column not found. Using default quality score of 0.5.")
        except Exception as e:
            st.warning(f"Error calculating quality score: {str(e)}. Using default value.")
            metrics['quality_score'] = 0.5
        
        # Calculate weighted overall score
        weights = {
            'volume_consistency': 0.25,
            'price_stability': 0.25,
            'lead_time_consistency': 0.20,
            'delivery_performance': 0.20,
            'quality_score': 0.10
        }
        
        overall_score = sum(metrics[metric] * weights[metric] for metric in weights)
        
        return {
            'overall_score': overall_score,
            **metrics
        }
        
    except Exception as e:
        st.error(f"Error in vendor performance calculation: {str(e)}")
        return {
            'overall_score': 0,
            'volume_consistency': 0,
            'price_stability': 0,
            'lead_time_consistency': 0.5,
            'delivery_performance': 0.5,
            'quality_score': 0.5
        }

def analyze_contract_suitability(item_vendor_data, min_spend_threshold=10000, min_frequency_threshold=4):
    """Analyze suitability for contracting based on spend and frequency"""
    try:
        # Validate input data
        if item_vendor_data.empty:
            return {
                'total_spend': 0,
                'order_frequency': 0,
                'monthly_frequency': 0,
                'demand_predictability': 0,
                'suitability_score': 0,
                'recommendation': "Not Suitable",
                'months_span': 0
            }
        
        # Calculate total spend with proper error handling
        unit_prices = pd.to_numeric(item_vendor_data['Unit Price'], errors='coerce').fillna(0)
        qty_delivered = pd.to_numeric(item_vendor_data['Qty Delivered'], errors='coerce').fillna(0)
        total_spend = (unit_prices * qty_delivered).sum()
        
        order_frequency = len(item_vendor_data)
        
        # Calculate time span
        try:
            date_range = item_vendor_data['Creation Date'].max() - item_vendor_data['Creation Date'].min()
            months_span = date_range.days / 30 if date_range.days > 0 else 1
        except Exception:
            months_span = 1
        
        monthly_frequency = order_frequency / months_span
        
        # Demand predictability
        try:
            monthly_demand = item_vendor_data.groupby(item_vendor_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
            if len(monthly_demand) > 1 and monthly_demand.mean() > 0:
                demand_cv = monthly_demand.std() / monthly_demand.mean()
                demand_predictability = max(0, 1 - demand_cv)
            else:
                demand_predictability = 0.5
        except Exception:
            demand_predictability = 0.5
        
        # Contract suitability score
        spend_score = min(total_spend / min_spend_threshold, 1.0) if min_spend_threshold > 0 else 1.0
        frequency_score = min(monthly_frequency / (min_frequency_threshold / 12), 1.0)
        
        suitability_score = (spend_score * 0.4 + frequency_score * 0.3 + demand_predictability * 0.3)
        
        # Contract recommendation
        if suitability_score >= 0.7 and total_spend >= min_spend_threshold:
            recommendation = "High Priority"
        elif suitability_score >= 0.5 and total_spend >= min_spend_threshold * 0.5:
            recommendation = "Medium Priority"
        elif suitability_score >= 0.3:
            recommendation = "Low Priority"
        else:
            recommendation = "Not Suitable"
        
        return {
            'total_spend': total_spend,
            'order_frequency': order_frequency,
            'monthly_frequency': monthly_frequency,
            'demand_predictability': demand_predictability,
            'suitability_score': suitability_score,
            'recommendation': recommendation,
            'months_span': months_span
        }
        
    except Exception as e:
        st.error(f"Error in contract suitability analysis: {str(e)}")
        return {
            'total_spend': 0,
            'order_frequency': 0,
            'monthly_frequency': 0,
            'demand_predictability': 0,
            'suitability_score': 0,
            'recommendation': "Not Suitable",
            'months_span': 0
        }

def calculate_contract_savings_potential(historical_data, contract_terms):
    """Calculate potential savings from contracting with error handling"""
    try:
        # Validate input data
        if historical_data.empty:
            return []
        
        # Current average price with error handling
        unit_prices = pd.to_numeric(historical_data['Unit Price'], errors='coerce').dropna()
        if len(unit_prices) == 0:
            return []
        
        current_avg_price = unit_prices.mean()
        
        # Calculate annual volume
        qty_delivered = pd.to_numeric(historical_data['Qty Delivered'], errors='coerce').fillna(0)
        try:
            days_span = (historical_data['Creation Date'].max() - historical_data['Creation Date'].min()).days
            if days_span > 0:
                annual_volume = qty_delivered.sum() / (days_span / 365)
            else:
                annual_volume = qty_delivered.sum()
        except Exception:
            annual_volume = qty_delivered.sum()
        
        # Contract savings scenarios
        savings_scenarios = []
        
        for term in contract_terms:
            try:
                # Price reduction from volume commitment
                volume_discount = term.get('volume_discount', 0)
                contract_price = current_avg_price * (1 - volume_discount)
                
                # Administrative cost savings
                admin_savings_per_order = term.get('admin_savings', 0)
                
                try:
                    current_orders_per_year = len(historical_data) / (days_span / 365) if days_span > 0 else len(historical_data)
                except Exception:
                    current_orders_per_year = len(historical_data)
                
                contract_orders_per_year = term.get('orders_per_year', current_orders_per_year)
                
                total_admin_savings = (current_orders_per_year - contract_orders_per_year) * admin_savings_per_order
                
                # Total annual savings
                price_savings = (current_avg_price - contract_price) * annual_volume
                total_savings = price_savings + total_admin_savings
                
                savings_percent = (total_savings / (current_avg_price * annual_volume)) * 100 if (current_avg_price * annual_volume) > 0 else 0
                
                savings_scenarios.append({
                    'contract_term': term['name'],
                    'contract_price': contract_price,
                    'price_savings': price_savings,
                    'admin_savings': total_admin_savings,
                    'total_savings': total_savings,
                    'savings_percent': savings_percent
                })
            except Exception as e:
                st.warning(f"Error calculating savings for {term.get('name', 'Unknown')}: {str(e)}")
                continue
        
        return savings_scenarios
        
    except Exception as e:
        st.error(f"Error in savings calculation: {str(e)}")
        return []

def display(df):
    st.header("🤝 Contracting Opportunities")
    st.markdown("Identify optimal contracting opportunities based on spend analysis, vendor performance, and demand predictability.")
    
    # Data validation with better error handling
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("Required columns: Vendor Name, Item, Unit Price, Qty Delivered, Creation Date")
        return
    
    # Clean data with enhanced error handling
    try:
        df_clean = df.copy()
        
        # Convert numeric columns
        df_clean['Unit Price'] = pd.to_numeric(df_clean['Unit Price'], errors='coerce')
        df_clean['Qty Delivered'] = pd.to_numeric(df_clean['Qty Delivered'], errors='coerce')
        
        # Remove rows with invalid data
        df_clean = df_clean.dropna(subset=required_columns)
        df_clean = df_clean[df_clean['Unit Price'] > 0]
        df_clean = df_clean[df_clean['Qty Delivered'] > 0]
        
        # Convert dates
        df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Creation Date'])
        
        # Add Qty Rejected column if it doesn't exist
        if 'Qty Rejected' not in df_clean.columns:
            df_clean['Qty Rejected'] = 0
            st.info("'Qty Rejected' column not found in data. Setting all rejection quantities to 0 for analysis.")
        else:
            df_clean['Qty Rejected'] = pd.to_numeric(df_clean['Qty Rejected'], errors='coerce').fillna(0)
        
        if len(df_clean) == 0:
            st.warning("No valid data found after cleaning. Please check your data format.")
            return
            
        st.success(f"Data processed successfully! {len(df_clean)} valid records found.")
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return
    
    # Continue with the rest of the tabs (keeping the existing code for tabs)
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Contract Identification", "📊 Vendor Performance", "💰 Savings Analysis", "📋 Contract Portfolio", "⚙️ Contract Strategy"])
    
    with tab1:
        st.subheader("🎯 Contract Opportunity Identification")
        
        # Configuration parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_spend = st.number_input("Min Annual Spend Threshold", min_value=0, value=50000, step=10000)
        with col2:
            min_frequency = st.number_input("Min Annual Order Frequency", min_value=1, value=6, step=1)
        with col3:
            analysis_period = st.selectbox("Analysis Period", ["All Data", "Last 12 Months", "Last 6 Months"])
        
        # Filter data by period
        if analysis_period == "Last 12 Months":
            cutoff_date = df_clean['Creation Date'].max() - timedelta(days=365)
            analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
        elif analysis_period == "Last 6 Months":
            cutoff_date = df_clean['Creation Date'].max() - timedelta(days=180)
            analysis_df = df_clean[df_clean['Creation Date'] >= cutoff_date]
        else:
            analysis_df = df_clean
        
        if st.button("🔍 Identify Contract Opportunities", type="primary"):
            with st.spinner("Analyzing contract opportunities..."):
                try:
                    contract_opportunities = []
                    
                    # Analyze vendor-item combinations
                    vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
                    
                    for (vendor, item), group_data in vendor_item_combinations:
                        suitability_analysis = analyze_contract_suitability(
                            group_data, min_spend, min_frequency
                        )
                        
                        if suitability_analysis['recommendation'] != "Not Suitable":
                            vendor_performance = calculate_vendor_performance_score(group_data)
                            
                            item_desc = group_data['Item Description'].iloc[0] if 'Item Description' in group_data.columns else f"Item {item}"
                            
                            contract_opportunities.append({
                                'Vendor Name': vendor,
                                'Item': item,
                                'Item Description': item_desc[:50] + "..." if len(str(item_desc)) > 50 else str(item_desc),
                                'Annual Spend': suitability_analysis['total_spend'],
                                'Order Frequency': suitability_analysis['order_frequency'],
                                'Monthly Frequency': suitability_analysis['monthly_frequency'],
                                'Demand Predictability': suitability_analysis['demand_predictability'],
                                'Vendor Performance': vendor_performance['overall_score'],
                                'Suitability Score': suitability_analysis['suitability_score'],
                                'Contract Priority': suitability_analysis['recommendation'],
                                'Avg Unit Price': group_data['Unit Price'].mean(),
                                'Price Stability': vendor_performance['price_stability']
                            })
                    
                    if contract_opportunities:
                        opportunities_df = pd.DataFrame(contract_opportunities)
                        opportunities_df = opportunities_df.sort_values(['Suitability Score', 'Annual Spend'], ascending=[False, False])
                        
                        # Add option to limit results for large datasets
                        if len(opportunities_df) > 1000:
                            show_limit = st.checkbox(f"Show top 500 results only (dataset has {len(opportunities_df):,} records)", value=True)
                            if show_limit:
                                display_opportunities_df = opportunities_df.head(500)
                                st.info(f"Showing top 500 out of {len(opportunities_df):,} opportunities for performance.")
                            else:
                                display_opportunities_df = opportunities_df
                        else:
                            display_opportunities_df = opportunities_df
                        
                        # Summary metrics (use full dataset)
                        total_contract_spend = opportunities_df['Annual Spend'].sum()
                        high_priority_count = len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])
                        avg_suitability = opportunities_df['Suitability Score'].mean()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Opportunities", len(opportunities_df))
                        with col2:
                            st.metric("High Priority Items", high_priority_count)
                        with col3:
                            st.metric("Total Contract Spend", f"{total_contract_spend:,.0f}")
                        with col4:
                            st.metric("Avg Suitability Score", f"{avg_suitability:.2f}")
                        
                        # Priority distribution (use full dataset)
                        priority_counts = opportunities_df['Contract Priority'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(
                                values=priority_counts.values,
                                names=priority_counts.index,
                                title="Contract Priority Distribution",
                                color_discrete_map={
                                    'High Priority': '#ff6b6b',
                                    'Medium Priority': '#ffd93d',
                                    'Low Priority': '#6bcf7f'
                                }
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Top opportunities by spend (use limited dataset for display)
                            top_by_spend = display_opportunities_df.nlargest(10, 'Annual Spend')
                            
                            fig = px.bar(
                                top_by_spend,
                                x='Annual Spend',
                                y='Vendor Name',
                                color='Contract Priority',
                                title="Top 10 Opportunities by Spend",
                                orientation='h',
                                color_discrete_map={
                                    'High Priority': '#ff6b6b',
                                    'Medium Priority': '#ffd93d',
                                    'Low Priority': '#6bcf7f'
                                }
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results table
                        st.subheader("📋 Contract Opportunities Details")
                        
                        # Check dataframe size and apply styling only if reasonable
                        total_cells = len(display_opportunities_df) * len(display_opportunities_df.columns)
                        
                        if total_cells > 100000:  # Limit styling for large datasets
                            # Format specific columns for display without styling
                            formatted_df = display_opportunities_df.copy()
                            formatted_df['Annual Spend'] = formatted_df['Annual Spend'].apply(lambda x: f"{x:,.0f}")
                            formatted_df['Monthly Frequency'] = formatted_df['Monthly Frequency'].apply(lambda x: f"{x:.1f}")
                            formatted_df['Demand Predictability'] = formatted_df['Demand Predictability'].apply(lambda x: f"{x:.2f}")
                            formatted_df['Vendor Performance'] = formatted_df['Vendor Performance'].apply(lambda x: f"{x:.2f}")
                            formatted_df['Suitability Score'] = formatted_df['Suitability Score'].apply(lambda x: f"{x:.2f}")
                            formatted_df['Avg Unit Price'] = formatted_df['Avg Unit Price'].apply(lambda x: f"{x:.2f}")
                            formatted_df['Price Stability'] = formatted_df['Price Stability'].apply(lambda x: f"{x:.2f}")
                            
                            st.dataframe(formatted_df, use_container_width=True)
                            st.info(f"Large dataset detected ({total_cells:,} cells). Styling disabled for performance.")
                        else:
                            st.dataframe(
                                display_opportunities_df.style.format({
                                    'Annual Spend': '{:,.0f}',
                                    'Monthly Frequency': '{:.1f}',
                                    'Demand Predictability': '{:.2f}',
                                    'Vendor Performance': '{:.2f}',
                                    'Suitability Score': '{:.2f}',
                                    'Avg Unit Price': '{:.2f}',
                                    'Price Stability': '{:.2f}'
                                }),
                                use_container_width=True
                            )
                        
                        # Opportunity matrix (use limited dataset for display)
                        st.subheader("📊 Contract Opportunity Matrix")
                        
                        fig = px.scatter(
                            display_opportunities_df,
                            x='Suitability Score',
                            y='Vendor Performance',
                            size='Annual Spend',
                            color='Contract Priority',
                            hover_data=['Vendor Name', 'Item', 'Annual Spend'],
                            title="Contract Opportunity Matrix: Suitability vs Vendor Performance",
                            labels={
                                'Suitability Score': 'Contract Suitability Score',
                                'Vendor Performance': 'Vendor Performance Score'
                            },
                            color_discrete_map={
                                'High Priority': '#ff6b6b',
                                'Medium Priority': '#ffd93d',
                                'Low Priority': '#6bcf7f'
                            }
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Export opportunities
                        col1, col2 = st.columns(2)
                        with col1:
                            csv_full = opportunities_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Export All Opportunities",
                                data=csv_full,
                                file_name=f"contract_opportunities_full_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            if len(opportunities_df) != len(display_opportunities_df):
                                csv_limited = display_opportunities_df.to_csv(index=False)
                                st.download_button(
                                    label=f"📥 Export Displayed ({len(display_opportunities_df)} records)",
                                    data=csv_limited,
                                    file_name=f"contract_opportunities_top500_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv"
                                )
                        
                        # Store results for other tabs (use full dataset)
                        st.session_state['contract_opportunities'] = opportunities_df
                    
                    else:
                        st.info("No contract opportunities found with the current criteria.")
                        
                except Exception as e:
                    st.error(f"Error during contract opportunity analysis: {str(e)}")
    
    # Keep the rest of the tabs unchanged (tab2, tab3, tab4, tab5)
    # [The remaining tabs code continues as in the original file...]
    
    with tab2:
        st.subheader("📊 Vendor Performance Analysis")
        
        # Vendor selection
        vendor_options = sorted(df_clean["Vendor Name"].dropna().unique())
        selected_vendors = st.multiselect(
            "Select Vendors for Performance Analysis",
            options=vendor_options,
            default=vendor_options[:5] if len(vendor_options) >= 5 else vendor_options,
            key="performance_vendors"
        )
        
        if selected_vendors:
            try:
                vendor_performance_results = []
                
                for vendor in selected_vendors:
                    vendor_data = df_clean[df_clean["Vendor Name"] == vendor]
                    performance_metrics = calculate_vendor_performance_score(vendor_data)
                    
                    # Additional metrics
                    total_spend = (vendor_data['Unit Price'] * vendor_data['Qty Delivered']).sum()
                    unique_items = vendor_data['Item'].nunique()
                    avg_order_size = vendor_data['Qty Delivered'].mean()
                    order_frequency = len(vendor_data)
                    
                    vendor_performance_results.append({
                        'Vendor Name': vendor,
                        'Overall Score': performance_metrics['overall_score'],
                        'Volume Consistency': performance_metrics['volume_consistency'],
                        'Price Stability': performance_metrics['price_stability'],
                        'Lead Time Consistency': performance_metrics['lead_time_consistency'],
                        'Delivery Performance': performance_metrics['delivery_performance'],
                        'Quality Score': performance_metrics['quality_score'],
                        'Total Spend': total_spend,
                        'Unique Items': unique_items,
                        'Avg Order Size': avg_order_size,
                        'Order Frequency': order_frequency
                    })
                
                performance_df = pd.DataFrame(vendor_performance_results)
                performance_df = performance_df.sort_values('Overall Score', ascending=False)
                
                # Performance summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    top_performer = performance_df.iloc[0]['Vendor Name']
                    top_score = performance_df.iloc[0]['Overall Score']
                    st.metric("Top Performer", top_performer, f"{top_score:.2f}")
                with col2:
                    avg_performance = performance_df['Overall Score'].mean()
                    st.metric("Average Performance", f"{avg_performance:.2f}")
                with col3:
                    high_performers = len(performance_df[performance_df['Overall Score'] >= 0.7])
                    st.metric("High Performers (≥0.7)", high_performers)
                
                # Performance comparison table
                st.subheader("🏅 Vendor Performance Scorecard")
                
                # Check dataframe size and apply styling only if reasonable
                total_cells = len(performance_df) * len(performance_df.columns)
                
                if total_cells > 50000:  # Limit styling for large datasets
                    # Format specific columns for display without styling
                    display_df = performance_df.copy()
                    display_df['Overall Score'] = display_df['Overall Score'].apply(lambda x: f"{x:.2f}")
                    display_df['Volume Consistency'] = display_df['Volume Consistency'].apply(lambda x: f"{x:.2f}")
                    display_df['Price Stability'] = display_df['Price Stability'].apply(lambda x: f"{x:.2f}")
                    display_df['Lead Time Consistency'] = display_df['Lead Time Consistency'].apply(lambda x: f"{x:.2f}")
                    display_df['Delivery Performance'] = display_df['Delivery Performance'].apply(lambda x: f"{x:.2f}")
                    display_df['Quality Score'] = display_df['Quality Score'].apply(lambda x: f"{x:.2f}")
                    display_df['Total Spend'] = display_df['Total Spend'].apply(lambda x: f"{x:,.0f}")
                    display_df['Avg Order Size'] = display_df['Avg Order Size'].apply(lambda x: f"{x:.1f}")
                    
                    st.dataframe(display_df, use_container_width=True)
                    st.info(f"Large dataset detected ({total_cells:,} cells). Styling disabled for performance.")
                else:
                    st.dataframe(
                        performance_df.style.format({
                            'Overall Score': '{:.2f}',
                            'Volume Consistency': '{:.2f}',
                            'Price Stability': '{:.2f}',
                            'Lead Time Consistency': '{:.2f}',
                            'Delivery Performance': '{:.2f}',
                            'Quality Score': '{:.2f}',
                            'Total Spend': '{:,.0f}',
                            'Avg Order Size': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"Error in vendor performance analysis: {str(e)}")
        
        else:
            st.info("Please select vendors for performance analysis.")
    
    # Add the remaining tabs (tab3, tab4, tab5) here with similar error handling...
    # For brevity, I'm showing the structure but you can copy the rest from your original file
    
    with tab3:
        st.subheader("💰 Contract Savings Analysis")
        
        if 'contract_opportunities' in st.session_state:
            opportunities_df = st.session_state['contract_opportunities']
            
            # Select opportunity for detailed analysis
            opportunity_options = [f"{row['Vendor Name']} - Item {row['Item']}" 
                                 for _, row in opportunities_df.iterrows()]
            
            selected_opportunity = st.selectbox(
                "Select Contract Opportunity for Savings Analysis",
                options=opportunity_options,
                key="savings_opportunity"
            )
            
            if selected_opportunity:
                # Parse selection
                vendor_name = selected_opportunity.split(' - Item ')[0]
                item_id = selected_opportunity.split(' - Item ')[1]
                
                # Get historical data
                try:
                    # Convert item_id to appropriate type
                    try:
                        item_id_converted = int(item_id)
                    except ValueError:
                        item_id_converted = item_id
                    
                    historical_data = df_clean[
                        (df_clean['Vendor Name'] == vendor_name) & 
                        (df_clean['Item'] == item_id_converted)
                    ]
                    
                    if historical_data.empty:
                        st.warning(f"No historical data found for {vendor_name} - Item {item_id}")
                    else:
                        # Contract terms configuration
                        st.subheader("⚙️ Contract Terms Configuration")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Contract Scenario 1: Short-term**")
                            short_discount = st.slider("Volume Discount (%)", 0, 20, 3, key="short_discount") / 100
                            short_admin_savings = st.number_input("Admin Savings per Order", 0, 200, 25, key="short_admin")
                            short_orders_per_year = st.number_input("Contract Orders/Year", 1, 52, 12, key="short_orders")
                        
                        with col2:
                            st.write("**Contract Scenario 2: Long-term**")
                            long_discount = st.slider("Volume Discount (%)", 0, 30, 8, key="long_discount") / 100
                            long_admin_savings = st.number_input("Admin Savings per Order", 0, 200, 50, key="long_admin")
                            long_orders_per_year = st.number_input("Contract Orders/Year", 1, 24, 6, key="long_orders")
                        
                        # Define contract terms
                        try:
                            date_range_days = (historical_data['Creation Date'].max() - historical_data['Creation Date'].min()).days
                            current_orders_per_year = len(historical_data) / (date_range_days / 365) if date_range_days > 0 else len(historical_data)
                        except Exception:
                            current_orders_per_year = len(historical_data)
                        
                        contract_terms = [
                            {
                                'name': 'Current (Spot Buy)',
                                'volume_discount': 0,
                                'admin_savings': 0,
                                'orders_per_year': current_orders_per_year
                            },
                            {
                                'name': 'Short-term Contract',
                                'volume_discount': short_discount,
                                'admin_savings': short_admin_savings,
                                'orders_per_year': short_orders_per_year
                            },
                            {
                                'name': 'Long-term Contract',
                                'volume_discount': long_discount,
                                'admin_savings': long_admin_savings,
                                'orders_per_year': long_orders_per_year
                            }
                        ]
                        
                        # Calculate savings
                        savings_analysis = calculate_contract_savings_potential(historical_data, contract_terms)
                        
                        if savings_analysis:
                            savings_df = pd.DataFrame(savings_analysis)
                            
                            # Display savings comparison
                            st.subheader("💵 Savings Comparison")
                            
                            # Calculate current annual cost
                            try:
                                current_annual_cost = (historical_data['Unit Price'] * historical_data['Qty Delivered']).sum()
                                if date_range_days > 0:
                                    current_annual_cost = current_annual_cost / (date_range_days / 365)
                            except Exception:
                                current_annual_cost = (historical_data['Unit Price'] * historical_data['Qty Delivered']).sum()
                            
                            col1, col2, col3 = st.columns(3)
                            
                            try:
                                short_term_savings = savings_df[savings_df['contract_term'] == 'Short-term Contract']['total_savings'].iloc[0]
                                long_term_savings = savings_df[savings_df['contract_term'] == 'Long-term Contract']['total_savings'].iloc[0]
                                
                                with col1:
                                    st.metric("Current Annual Cost", f"{current_annual_cost:,.0f}")
                                with col2:
                                    short_percent = (short_term_savings / current_annual_cost) * 100 if current_annual_cost > 0 else 0
                                    st.metric("Short-term Savings", f"{short_term_savings:,.0f}", f"{short_percent:.1f}%")
                                with col3:
                                    long_percent = (long_term_savings / current_annual_cost) * 100 if current_annual_cost > 0 else 0
                                    st.metric("Long-term Savings", f"{long_term_savings:,.0f}", f"{long_percent:.1f}%")
                                
                                # Detailed savings breakdown
                                st.subheader("📊 Detailed Savings Breakdown")
                                
                                # Format the dataframe for display
                                display_savings_df = savings_df.copy()
                                display_savings_df['contract_price'] = display_savings_df['contract_price'].apply(lambda x: f"{x:.2f}")
                                display_savings_df['price_savings'] = display_savings_df['price_savings'].apply(lambda x: f"{x:,.0f}")
                                display_savings_df['admin_savings'] = display_savings_df['admin_savings'].apply(lambda x: f"{x:,.0f}")
                                display_savings_df['total_savings'] = display_savings_df['total_savings'].apply(lambda x: f"{x:,.0f}")
                                display_savings_df['savings_percent'] = display_savings_df['savings_percent'].apply(lambda x: f"{x:.1f}%")
                                
                                st.dataframe(display_savings_df, use_container_width=True)
                                
                                # Savings visualization
                                fig = go.Figure()
                                
                                categories = ['Price Savings', 'Admin Savings']
                                
                                for _, row in savings_df.iterrows():
                                    if row['contract_term'] != 'Current (Spot Buy)':
                                        fig.add_trace(go.Bar(
                                            name=row['contract_term'],
                                            x=categories,
                                            y=[row['price_savings'], row['admin_savings']]
                                        ))
                                
                                fig.update_layout(
                                    title="Savings Breakdown by Contract Type",
                                    xaxis_title="Savings Category",
                                    yaxis_title="Annual Savings",
                                    barmode='group',
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # ROI analysis
                                st.subheader("📈 Contract ROI Analysis")
                                
                                contract_setup_cost = st.number_input("Estimated Contract Setup Cost", min_value=0, value=5000)
                                
                                roi_data = []
                                for _, row in savings_df.iterrows():
                                    if row['contract_term'] != 'Current (Spot Buy)':
                                        payback_months = (contract_setup_cost / row['total_savings'] * 12) if row['total_savings'] > 0 else float('inf')
                                        three_year_roi = ((row['total_savings'] * 3 - contract_setup_cost) / contract_setup_cost * 100) if contract_setup_cost > 0 else 0
                                        
                                        roi_data.append({
                                            'Contract Type': row['contract_term'],
                                            'Annual Savings': f"{row['total_savings']:,.0f}",
                                            'Setup Cost': f"{contract_setup_cost:,.0f}",
                                            'Payback (Months)': f"{payback_months:.1f}" if payback_months != float('inf') else "N/A",
                                            '3-Year ROI (%)': f"{three_year_roi:.0f}%"
                                        })
                                
                                if roi_data:
                                    roi_df = pd.DataFrame(roi_data)
                                    st.dataframe(roi_df, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error in savings calculation display: {str(e)}")
                                st.dataframe(savings_df, use_container_width=True)
                        
                        else:
                            st.warning("Unable to calculate savings for this opportunity.")
                
                except Exception as e:
                    st.error(f"Error processing savings analysis: {str(e)}")
        
        else:
            st.info("Run contract opportunity identification first to see savings analysis.")
    
    with tab4:
        st.subheader("📋 Contract Portfolio Management")
        
        if 'contract_opportunities' in st.session_state:
            opportunities_df = st.session_state['contract_opportunities']
            
            # Portfolio overview
            st.write("**Contract Portfolio Overview:**")
            
            # Prioritize contracts by value and suitability
            high_priority = opportunities_df[opportunities_df['Contract Priority'] == 'High Priority']
            medium_priority = opportunities_df[opportunities_df['Contract Priority'] == 'Medium Priority']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("High Priority Contracts", len(high_priority))
                st.metric("Total High Priority Spend", f"{high_priority['Annual Spend'].sum():,.0f}")
            
            with col2:
                st.metric("Medium Priority Contracts", len(medium_priority))
                st.metric("Total Medium Priority Spend", f"{medium_priority['Annual Spend'].sum():,.0f}")
            
            # Contract implementation roadmap
            st.subheader("🗺️ Implementation Roadmap")
            
            # Sort by priority and spend for implementation sequence
            implementation_order = opportunities_df.sort_values(
                ['Suitability Score', 'Annual Spend'], 
                ascending=[False, False]
            ).head(20)  # Top 20 for roadmap
            
            # Add implementation phases
            implementation_order = implementation_order.copy()
            implementation_order['Implementation Phase'] = pd.cut(
                range(len(implementation_order)),
                bins=4,
                labels=['Phase 1 (0-3 months)', 'Phase 2 (3-6 months)', 
                       'Phase 3 (6-9 months)', 'Phase 4 (9-12 months)']
            )
            
            # Implementation timeline
            phase_summary = implementation_order.groupby('Implementation Phase').agg({
                'Annual Spend': 'sum',
                'Vendor Name': 'count'
            }).round(0)
            phase_summary.columns = ['Total Spend', 'Number of Contracts']
            
            st.dataframe(phase_summary, use_container_width=True)
            
            # Detailed roadmap
            st.subheader("📅 Detailed Roadmap")
            
            roadmap_display = implementation_order[[
                'Vendor Name', 'Item', 'Annual Spend', 'Contract Priority', 
                'Suitability Score', 'Implementation Phase'
            ]].copy()
            
            # Format for display
            roadmap_display['Annual Spend'] = roadmap_display['Annual Spend'].apply(lambda x: f"{x:,.0f}")
            roadmap_display['Suitability Score'] = roadmap_display['Suitability Score'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(roadmap_display, use_container_width=True)
            
            # Resource planning
            st.subheader("📊 Resource Planning")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Contracts by vendor
                vendor_contract_count = opportunities_df['Vendor Name'].value_counts().head(10)
                
                fig = px.bar(
                    x=vendor_contract_count.index,
                    y=vendor_contract_count.values,
                    title="Contracts by Vendor (Top 10)",
                    labels={'x': 'Vendor', 'y': 'Number of Contracts'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Implementation effort estimation
                effort_by_phase = implementation_order.groupby('Implementation Phase').size()
                
                fig = px.pie(
                    values=effort_by_phase.values,
                    names=effort_by_phase.index,
                    title="Contract Distribution by Phase"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            st.subheader("⚠️ Portfolio Risk Assessment")
            
            risk_factors = []
            
            # Vendor concentration risk
            vendor_spend_concentration = opportunities_df.groupby('Vendor Name')['Annual Spend'].sum()
            total_portfolio_spend = vendor_spend_concentration.sum()
            top_vendor_concentration = (vendor_spend_concentration.max() / total_portfolio_spend) * 100
            
            if top_vendor_concentration > 30:
                risk_factors.append({
                    'Risk Type': 'Vendor Concentration',
                    'Risk Level': 'High' if top_vendor_concentration > 50 else 'Medium',
                    'Description': f"Top vendor represents {top_vendor_concentration:.1f}% of contract spend",
                    'Mitigation': 'Diversify supplier base, develop backup suppliers'
                })
            
            # Performance risk
            low_performance_contracts = opportunities_df[opportunities_df['Vendor Performance'] < 0.6]
            if len(low_performance_contracts) > 0:
                risk_factors.append({
                    'Risk Type': 'Vendor Performance',
                    'Risk Level': 'Medium',
                    'Description': f"{len(low_performance_contracts)} contracts with vendors scoring <0.6",
                    'Mitigation': 'Implement performance improvement plans, consider alternative suppliers'
                })
            
            # Market risk for high spend items
            high_spend_items = opportunities_df[opportunities_df['Annual Spend'] > opportunities_df['Annual Spend'].quantile(0.8)]
            if len(high_spend_items) > 0:
                risk_factors.append({
                    'Risk Type': 'Market Risk',
                    'Risk Level': 'Medium',
                    'Description': f"{len(high_spend_items)} high-spend items may be subject to market volatility",
                    'Mitigation': 'Include price adjustment clauses, monitor market conditions'
                })
            
            if risk_factors:
                risk_df = pd.DataFrame(risk_factors)
                st.dataframe(risk_df, use_container_width=True)
            else:
                st.success("No significant portfolio risks identified.")
        
        else:
            st.info("Run contract opportunity identification first to see portfolio management.")
    
    with tab5:
        st.subheader("⚙️ Contract Strategy & Best Practices")
        
        # Strategic guidelines
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 **Contract Selection Criteria**")
            st.write("**High Priority Items:**")
            st.write("• Annual spend > $50,000")
            st.write("• Regular, predictable demand")
            st.write("• Stable supplier relationship")
            st.write("• Limited supplier options")
            st.write("")
            st.write("**Medium Priority Items:**")
            st.write("• Annual spend $10,000 - $50,000")
            st.write("• Moderate demand variability")
            st.write("• Good supplier performance")
            st.write("• Some price volatility")
        
        with col2:
            st.markdown("#### 📋 **Contract Terms Framework**")
            st.write("**Price Terms:**")
            st.write("• Volume-based discounts")
            st.write("• Price adjustment mechanisms")
            st.write("• Market price protections")
            st.write("")
            st.write("**Performance Terms:**")
            st.write("• Delivery requirements")
            st.write("• Quality specifications")
            st.write("• Service level agreements")
            st.write("• Performance penalties/incentives")
        
        # Contract types
        st.subheader("📄 Contract Types & Applications")
        
        contract_types = [
            {
                "Type": "Fixed Price Contract",
                "Best For": "Stable demand, predictable costs",
                "Duration": "6-18 months",
                "Risk Level": "Low",
                "Savings Potential": "3-8%"
            },
            {
                "Type": "Volume Commitment",
                "Best For": "High volume, regular demand",
                "Duration": "12-36 months",
                "Risk Level": "Medium",
                "Savings Potential": "5-15%"
            },
            {
                "Type": "Blanket Purchase Order",
                "Best For": "Multiple items, same supplier",
                "Duration": "12-24 months",
                "Risk Level": "Low",
                "Savings Potential": "2-6%"
            },
            {
                "Type": "Requirements Contract",
                "Best For": "Uncertain volumes, guaranteed supply",
                "Duration": "12-36 months",
                "Risk Level": "Medium",
                "Savings Potential": "4-10%"
            }
        ]
        
        contract_types_df = pd.DataFrame(contract_types)
        st.dataframe(contract_types_df, use_container_width=True)
        
        # Implementation best practices
        st.subheader("💡 Implementation Best Practices")
        
        practices_col1, practices_col2 = st.columns(2)
        
        with practices_col1:
            st.markdown("#### ✅ **Do's:**")
            st.write("• Conduct thorough supplier due diligence")
            st.write("• Define clear performance metrics")
            st.write("• Include termination clauses")
            st.write("• Regular contract reviews")
            st.write("• Document all changes")
            st.write("• Maintain supplier relationships")
        
        with practices_col2:
            st.markdown("#### ❌ **Don'ts:**")
            st.write("• Lock in without market research")
            st.write("• Ignore performance monitoring")
            st.write("• Overlook legal compliance")
            st.write("• Neglect backup suppliers")
            st.write("• Skip regular price benchmarking")
            st.write("• Ignore contract renewal planning")
        
        # KPIs and metrics
        st.subheader("📊 Key Performance Indicators")
        
        kpi_categories = {
            "Financial KPIs": [
                "Cost savings achieved vs target",
                "Price variance from market",
                "Total cost of ownership reduction",
                "Contract compliance rate"
            ],
            "Operational KPIs": [
                "On-time delivery rate",
                "Quality performance",
                "Order cycle time",
                "Supplier responsiveness"
            ],
            "Strategic KPIs": [
                "Supplier relationship score",
                "Innovation contribution",
                "Risk mitigation effectiveness",
                "Market intelligence value"
            ]
        }
        
        for category, kpis in kpi_categories.items():
            st.write(f"**{category}:**")
            for kpi in kpis:
                st.write(f"• {kpi}")
            st.write("")
        
        # Contract lifecycle
        st.subheader("🔄 Contract Lifecycle Management")
        
        lifecycle_stages = [
            {"Stage": "Planning", "Duration": "2-4 weeks", "Key Activities": "Market analysis, supplier evaluation, term negotiation"},
            {"Stage": "Execution", "Duration": "2-6 weeks", "Key Activities": "Legal review, approvals, contract signing"},
            {"Stage": "Management", "Duration": "Contract term", "Key Activities": "Performance monitoring, relationship management"},
            {"Stage": "Renewal/Exit", "Duration": "4-8 weeks", "Key Activities": "Performance review, renegotiation, transition planning"}
        ]
        
        lifecycle_df = pd.DataFrame(lifecycle_stages)
        st.dataframe(lifecycle_df, use_container_width=True)
        
        # Export strategy guide
        if st.button("📥 Export Contract Strategy Guide"):
            strategy_guide = {
                'section': ['Selection Criteria', 'Contract Types', 'Best Practices', 'KPIs', 'Lifecycle'],
                'content': [
                    'High Priority: >$50K annual spend, predictable demand, stable supplier',
                    'Fixed Price (6-18mo), Volume Commitment (12-36mo), BPO (12-24mo)',
                    'Due diligence, clear metrics, termination clauses, regular reviews',
                    'Cost savings, delivery rate, quality performance, supplier score',
                    'Planning→Execution→Management→Renewal (2-52 weeks per stage)'
                ],
                'timeline': ['Pre-contract', 'Contract Design', 'Implementation', 'Ongoing', 'End of Term'],
                'responsibility': ['Procurement', 'Legal/Procurement', 'Operations', 'All Functions', 'Procurement']
            }
            
            guide_df = pd.DataFrame(strategy_guide)
            csv = guide_df.to_csv(index=False)
            
            st.download_button(
                label="Download Strategy Guide",
                data=csv,
                file_name=f"contract_strategy_guide_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
