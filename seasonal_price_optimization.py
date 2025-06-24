import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def enhanced_display(df):
    """Enhanced Seasonal Price Optimization Module"""
    st.header("🌟 Enhanced Seasonal Price Optimization")
    st.markdown("Advanced procurement intelligence with predictive analytics and multi-dimensional optimization.")
    
    # Enhanced data validation
    required_columns = ['Creation Date', 'Unit Price', 'Item']
    optional_columns = ['Vendor', 'Region', 'Qty Delivered', 'Lead Time Days']
    
    missing_required = [col for col in required_columns if col not in df.columns]
    available_optional = [col for col in optional_columns if col in df.columns]
    
    if missing_required:
        st.error(f"Missing required columns: {', '.join(missing_required)}")
        return
    
    # Enhanced data preparation
    df_clean = prepare_enhanced_data(df, available_optional)
    
    if len(df_clean) == 0:
        st.warning("No valid data found for analysis.")
        return
    
    # Show data coverage summary
    show_data_coverage(df_clean, available_optional)
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Advanced Patterns", 
        "🔮 Predictive Analytics", 
        "🌍 Multi-Dimensional Analysis",
        "💰 Optimization Engine",
        "⚡ Action Center"
    ])
    
    with tab1:
        advanced_pattern_analysis(df_clean)
    
    with tab2:
        predictive_analytics(df_clean)
    
    with tab3:
        multi_dimensional_analysis(df_clean, available_optional)
    
    with tab4:
        optimization_engine(df_clean, available_optional)
    
    with tab5:
        action_center(df_clean, available_optional)

def prepare_enhanced_data(df, available_optional):
    """Enhanced data preparation with multiple dimensions"""
    df_clean = df.copy()
    
    # Convert date and clean basic data
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date', 'Unit Price', 'Item'])
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    
    # Add time components
    df_clean['Year'] = df_clean['Creation Date'].dt.year
    df_clean['Month'] = df_clean['Creation Date'].dt.month
    df_clean['Quarter'] = df_clean['Creation Date'].dt.quarter
    df_clean['Week'] = df_clean['Creation Date'].dt.isocalendar().week
    df_clean['DayOfYear'] = df_clean['Creation Date'].dt.dayofyear
    df_clean['Month_Name'] = df_clean['Creation Date'].dt.month_name()
    df_clean['Season'] = df_clean['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Handle optional dimensions
    if 'Vendor' not in df_clean.columns:
        df_clean['Vendor'] = 'Unknown'
    if 'Region' not in df_clean.columns:
        df_clean['Region'] = 'Unknown'
    if 'Lead Time Days' not in df_clean.columns:
        df_clean['Lead Time Days'] = 30  # Default lead time
    
    # Calculate additional metrics
    if 'Qty Delivered' in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    else:
        df_clean['Qty Delivered'] = 1
        df_clean['Line Total'] = df_clean['Unit Price']
    
    return df_clean

def show_data_coverage(df_clean, available_optional):
    """Display data coverage summary"""
    st.subheader("📈 Data Coverage Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df_clean):,}")
        st.metric("Unique Items", f"{df_clean['Item'].nunique():,}")
    
    with col2:
        date_range = df_clean['Creation Date'].max() - df_clean['Creation Date'].min()
        st.metric("Date Range", f"{date_range.days} days")
        st.metric("Years Covered", f"{df_clean['Year'].nunique()}")
    
    with col3:
        if 'Vendor' in available_optional:
            st.metric("Vendors", f"{df_clean['Vendor'].nunique()}")
        if 'Region' in available_optional:
            st.metric("Regions", f"{df_clean['Region'].nunique()}")
    
    with col4:
        total_spend = df_clean['Line Total'].sum()
        st.metric("Total Spend", f"${total_spend:,.0f}")
        avg_price = df_clean['Unit Price'].mean()
        st.metric("Avg Unit Price", f"${avg_price:.2f}")

def advanced_pattern_analysis(df_clean):
    """Advanced seasonal pattern analysis with statistical rigor"""
    st.subheader("📊 Advanced Seasonal Pattern Analysis")
    
    # Item selection with statistics
    items = sorted(df_clean['Item'].unique())
    
    # Create item summary for selection
    item_stats = []
    for item in items:
        item_data = df_clean[df_clean['Item'] == item]
        item_stats.append({
            'Item': item,
            'Records': len(item_data),
            'Date Range': (item_data['Creation Date'].max() - item_data['Creation Date'].min()).days,
            'Avg Price': item_data['Unit Price'].mean(),
            'Price Volatility': item_data['Unit Price'].std() / item_data['Unit Price'].mean() * 100
        })
    
    item_stats_df = pd.DataFrame(item_stats)
    item_stats_df = item_stats_df.sort_values('Records', ascending=False)
    
    # Show top items by data availability
    st.write("**Items by Data Availability:**")
    st.dataframe(
        item_stats_df.head(10).style.format({
            'Avg Price': '${:.2f}',
            'Price Volatility': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    selected_item = st.selectbox("Select Item for Detailed Analysis", items)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item].copy()
        
        if len(item_data) < 24:  # Need at least 2 years of monthly data
            st.warning(f"Limited data for {selected_item}. Analysis may be less reliable.")
        
        # Create time series
        monthly_series = item_data.groupby(item_data['Creation Date'].dt.to_period('M'))['Unit Price'].mean()
        monthly_series.index = monthly_series.index.to_timestamp()
        
        # Seasonal decomposition
        if len(monthly_series) >= 24:
            try:
                decomposition = seasonal_decompose(monthly_series, model='additive', period=12)
                
                # Plot decomposition
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=monthly_series.index, y=monthly_series.values,
                    name='Original', line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=monthly_series.index, y=decomposition.trend,
                    name='Trend', line=dict(color='red')
                ))
                fig.add_trace(go.Scatter(
                    x=monthly_series.index, y=decomposition.seasonal,
                    name='Seasonal', line=dict(color='green')
                ))
                
                fig.update_layout(
                    title=f'Time Series Decomposition - {selected_item}',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal strength analysis
                seasonal_strength = np.var(decomposition.seasonal) / np.var(decomposition.seasonal + decomposition.resid)
                trend_strength = np.var(decomposition.trend) / np.var(decomposition.trend + decomposition.resid)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Seasonal Strength", f"{seasonal_strength:.3f}")
                with col2:
                    st.metric("Trend Strength", f"{trend_strength:.3f}")
                
            except Exception as e:
                st.error(f"Could not perform seasonal decomposition: {str(e)}")
        
        # Advanced seasonal analysis
        perform_advanced_seasonal_analysis(item_data, selected_item)

def perform_advanced_seasonal_analysis(item_data, item_name):
    """Perform advanced seasonal analysis with confidence intervals"""
    
    # Monthly analysis with confidence intervals
    monthly_stats = item_data.groupby('Month')['Unit Price'].agg([
        'mean', 'std', 'count', 
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 75)
    ]).reset_index()
    
    monthly_stats.columns = ['Month', 'Mean', 'Std', 'Count', 'Q25', 'Q75']
    monthly_stats['Month_Name'] = monthly_stats['Month'].apply(
        lambda x: pd.to_datetime(f'2024-{x:02d}-01').strftime('%B')
    )
    
    # Calculate confidence intervals
    monthly_stats['CI_Lower'] = monthly_stats['Mean'] - 1.96 * (monthly_stats['Std'] / np.sqrt(monthly_stats['Count']))
    monthly_stats['CI_Upper'] = monthly_stats['Mean'] + 1.96 * (monthly_stats['Std'] / np.sqrt(monthly_stats['Count']))
    
    # Plot with confidence intervals
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_stats['Month_Name'], y=monthly_stats['Mean'],
        mode='lines+markers', name='Average Price',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_stats['Month_Name'], y=monthly_stats['CI_Upper'],
        mode='lines', line=dict(width=0), showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_stats['Month_Name'], y=monthly_stats['CI_Lower'],
        mode='lines', line=dict(width=0), showlegend=False,
        fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title=f'Monthly Price Patterns with Confidence Intervals - {item_name}',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def predictive_analytics(df_clean):
    """Advanced predictive analytics for price forecasting"""
    st.subheader("🔮 Predictive Price Analytics")
    
    # Item selection for prediction
    items_with_sufficient_data = []
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        if len(item_data) >= 12:  # Need at least 12 data points
            items_with_sufficient_data.append(item)
    
    if not items_with_sufficient_data:
        st.warning("Not enough historical data for predictive analytics.")
        return
    
    selected_item = st.selectbox("Select Item for Price Prediction", items_with_sufficient_data)
    
    if selected_item:
        item_data = df_clean[df_clean['Item'] == selected_item].copy()
        
        # Create monthly time series
        monthly_data = item_data.groupby(item_data['Creation Date'].dt.to_period('M')).agg({
            'Unit Price': 'mean',
            'Qty Delivered': 'sum',
            'Line Total': 'sum'
        }).reset_index()
        
        monthly_data['Date'] = monthly_data['Creation Date'].dt.to_timestamp()
        monthly_data = monthly_data.sort_values('Date')
        
        # Prepare features for ML model
        monthly_data['Month'] = monthly_data['Date'].dt.month
        monthly_data['Quarter'] = monthly_data['Date'].dt.quarter
        monthly_data['Year'] = monthly_data['Date'].dt.year
        monthly_data['Days_Since_Start'] = (monthly_data['Date'] - monthly_data['Date'].min()).dt.days
        
        # Create lag features
        for lag in [1, 2, 3, 6, 12]:
            monthly_data[f'Price_Lag_{lag}'] = monthly_data['Unit Price'].shift(lag)
        
        # Remove rows with NaN values
        model_data = monthly_data.dropna()
        
        if len(model_data) >= 8:
            # Split data for training and testing
            train_size = int(len(model_data) * 0.8)
            train_data = model_data[:train_size]
            test_data = model_data[train_size:]
            
            # Prepare features
            feature_cols = ['Month', 'Quarter', 'Days_Since_Start'] + [col for col in model_data.columns if 'Lag' in col]
            
            X_train = train_data[feature_cols]
            y_train = train_data['Unit Price']
            X_test = test_data[feature_cols]
            y_test = test_data['Unit Price']
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = rf_model.predict(X_train)
            test_pred = rf_model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred) if len(y_test) > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training R²", f"{train_r2:.3f}")
            with col2:
                st.metric("Testing R²", f"{test_r2:.3f}")
            with col3:
                train_mae = mean_absolute_error(y_train, train_pred)
                st.metric("Training MAE", f"${train_mae:.2f}")
            
            # Plot predictions
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=train_data['Date'], y=y_train,
                mode='lines', name='Historical (Train)',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=train_data['Date'], y=train_pred,
                mode='lines', name='Predicted (Train)',
                line=dict(color='red', dash='dash')
            ))
            
            if len(test_data) > 0:
                fig.add_trace(go.Scatter(
                    x=test_data['Date'], y=y_test,
                    mode='lines', name='Historical (Test)',
                    line=dict(color='green')
                ))
                
                fig.add_trace(go.Scatter(
                    x=test_data['Date'], y=test_pred,
                    mode='lines', name='Predicted (Test)',
                    line=dict(color='orange', dash='dash')
                ))
            
            fig.update_layout(
                title=f'Price Prediction Model - {selected_item}',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Future predictions
            st.subheader("📅 Future Price Predictions")
            
            # Generate future predictions
            future_months = 6
            last_date = monthly_data['Date'].max()
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, future_months + 1)]
            
            future_predictions = []
            current_data = model_data.iloc[-1:].copy()
            
            for future_date in future_dates:
                # Prepare features for future prediction
                future_features = {
                    'Month': future_date.month,
                    'Quarter': future_date.quarter,
                    'Days_Since_Start': (future_date - monthly_data['Date'].min()).days
                }
                
                # Use last known lag values (simplified approach)
                for lag in [1, 2, 3, 6, 12]:
                    col_name = f'Price_Lag_{lag}'
                    if col_name in current_data.columns:
                        future_features[col_name] = current_data[col_name].iloc[0]
                    else:
                        future_features[col_name] = monthly_data['Unit Price'].iloc[-1]
                
                # Make prediction
                pred = rf_model.predict([list(future_features.values())])[0]
                future_predictions.append({
                    'Date': future_date,
                    'Predicted_Price': pred
                })
            
            future_df = pd.DataFrame(future_predictions)
            
            # Display future predictions
            st.dataframe(
                future_df.style.format({'Predicted_Price': '${:.2f}'}),
                use_container_width=True
            )
            
            # Feature importance
            st.subheader("🎯 Model Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), 
                        x='Importance', y='Feature',
                        orientation='h',
                        title="Top 10 Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

def multi_dimensional_analysis(df_clean, available_optional):
    """Multi-dimensional analysis across vendors and regions"""
    st.subheader("🌍 Multi-Dimensional Price Analysis")
    
    # Vendor Analysis
    if 'Vendor' in available_optional and df_clean['Vendor'].nunique() > 1:
        st.subheader("🏢 Vendor Comparison Analysis")
        
        vendor_analysis = df_clean.groupby(['Item', 'Vendor']).agg({
            'Unit Price': ['mean', 'std', 'count'],
            'Line Total': 'sum'
        }).reset_index()
        
        vendor_analysis.columns = ['Item', 'Vendor', 'Avg_Price', 'Price_Std', 'Count', 'Total_Spend']
        vendor_analysis['Price_CV'] = vendor_analysis['Price_Std'] / vendor_analysis['Avg_Price']
        
        # Show vendor comparison for selected item
        items_multi_vendor = vendor_analysis[vendor_analysis['Count'] >= 3].groupby('Item').size()
        items_multi_vendor = items_multi_vendor[items_multi_vendor >= 2].index.tolist()
        
        if items_multi_vendor:
            selected_item_vendor = st.selectbox("Select Item for Vendor Analysis", items_multi_vendor)
            
            item_vendor_data = vendor_analysis[vendor_analysis['Item'] == selected_item_vendor]
            item_vendor_data = item_vendor_data.sort_values('Avg_Price')
            
            st.dataframe(
                item_vendor_data[['Vendor', 'Avg_Price', 'Price_Std', 'Count', 'Total_Spend']].style.format({
                    'Avg_Price': '${:.2f}',
                    'Price_Std': '${:.2f}',
                    'Total_Spend': '${:,.0f}'
                }),
                use_container_width=True
            )
            
            # Vendor price comparison chart
            fig = px.box(df_clean[df_clean['Item'] == selected_item_vendor], 
                        x='Vendor', y='Unit Price',
                        title=f"Price Distribution by Vendor - {selected_item_vendor}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Regional Analysis
    if 'Region' in available_optional and df_clean['Region'].nunique() > 1:
        st.subheader("🗺️ Regional Price Analysis")
        
        regional_analysis = df_clean.groupby(['Item', 'Region']).agg({
            'Unit Price': ['mean', 'std', 'count'],
            'Line Total': 'sum'
        }).reset_index()
        
        regional_analysis.columns = ['Item', 'Region', 'Avg_Price', 'Price_Std', 'Count', 'Total_Spend']
        
        # Regional arbitrage opportunities
        arbitrage_opportunities = []
        for item in df_clean['Item'].unique():
            item_regional = regional_analysis[regional_analysis['Item'] == item]
            if len(item_regional) >= 2:
                min_price = item_regional['Avg_Price'].min()
                max_price = item_regional['Avg_Price'].max()
                price_spread = max_price - min_price
                spread_pct = (price_spread / min_price) * 100
                
                if spread_pct > 5:  # Significant arbitrage opportunity
                    arbitrage_opportunities.append({
                        'Item': item,
                        'Min_Price': min_price,
                        'Max_Price': max_price,
                        'Spread_$': price_spread,
                        'Spread_%': spread_pct,
                        'Best_Region': item_regional.loc[item_regional['Avg_Price'].idxmin(), 'Region'],
                        'Worst_Region': item_regional.loc[item_regional['Avg_Price'].idxmax(), 'Region']
                    })
        
        if arbitrage_opportunities:
            arbitrage_df = pd.DataFrame(arbitrage_opportunities)
            arbitrage_df = arbitrage_df.sort_values('Spread_%', ascending=False)
            
            st.subheader("💱 Regional Arbitrage Opportunities")
            st.dataframe(
                arbitrage_df.head(10).style.format({
                    'Min_Price': '${:.2f}',
                    'Max_Price': '${:.2f}',
                    'Spread_$': '${:.2f}',
                    'Spread_%': '{:.1f}%'
                }),
                use_container_width=True
            )

def optimization_engine(df_clean, available_optional):
    """Advanced optimization engine for procurement decisions"""
    st.subheader("💰 Procurement Optimization Engine")
    
    # Calculate comprehensive optimization metrics
    optimization_results = []
    
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        if len(item_data) >= 6:
            # Seasonal optimization
            monthly_avg = item_data.groupby('Month')['Unit Price'].mean()
            overall_avg = item_data['Unit Price'].mean()
            
            best_month = monthly_avg.idxmin()
            worst_month = monthly_avg.idxmax()
            seasonal_savings = ((monthly_avg[worst_month] - monthly_avg[best_month]) / monthly_avg[worst_month]) * 100
            
            # Volume optimization
            if item_data['Qty Delivered'].var() > 0:
                volume_price_corr = item_data['Qty Delivered'].corr(item_data['Unit Price'])
            else:
                volume_price_corr = 0
            
            # Lead time analysis
            if 'Lead Time Days' in item_data.columns:
                avg_lead_time = item_data['Lead Time Days'].mean()
            else:
                avg_lead_time = 30
            
            # Annual spend and potential savings
            annual_spend = item_data['Line Total'].sum()
            potential_seasonal_savings = annual_spend * (seasonal_savings / 100)
            
            # Risk analysis
            price_volatility = item_data['Unit Price'].std() / item_data['Unit Price'].mean() * 100
            
            optimization_results.append({
                'Item': item,
                'Annual_Spend': annual_spend,
                'Current_Avg_Price': overall_avg,
                'Best_Month': pd.to_datetime(f'2024-{best_month:02d}-01').strftime('%B'),
                'Worst_Month': pd.to_datetime(f'2024-{worst_month:02d}-01').strftime('%B'),
                'Seasonal_Savings_%': seasonal_savings,
                'Potential_Savings_$': potential_seasonal_savings,
                'Price_Volatility_%': price_volatility,
                'Volume_Price_Correlation': volume_price_corr,
                'Avg_Lead_Time': avg_lead_time,
                'Optimization_Score': calculate_optimization_score(seasonal_savings, annual_spend, price_volatility)
            })
    
    optimization_df = pd.DataFrame(optimization_results)
    optimization_df = optimization_df.sort_values('Optimization_Score', ascending=False)
    
    # Display top optimization opportunities
    st.subheader("🎯 Top Optimization Opportunities")
    
    display_cols = ['Item', 'Annual_Spend', 'Seasonal_Savings_%', 'Potential_Savings_$', 
                   'Best_Month', 'Price_Volatility_%', 'Optimization_Score']
    
    st.dataframe(
        optimization_df.head(15)[display_cols].style.format({
            'Annual_Spend': '${:,.0f}',
            'Seasonal_Savings_%': '{:.1f}%',
            'Potential_Savings_$': '${:,.0f}',
            'Price_Volatility_%': '{:.1f}%',
            'Optimization_Score': '{:.1f}'
        }),
        use_container_width=True
    )
    
    # Optimization summary
    total_potential_savings = optimization_df['Potential_Savings_$'].sum()
    total_spend = optimization_df['Annual_Spend'].sum()
    avg_optimization_score = optimization_df['Optimization_Score'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Potential Savings", f"${total_potential_savings:,.0f}")
    with col2:
        st.metric("Savings Rate", f"{(total_potential_savings/total_spend*100):.1f}%")
    with col3:
        st.metric("Avg Optimization Score", f"{avg_optimization_score:.1f}")

def calculate_optimization_score(seasonal_savings, annual_spend, price_volatility):
    """Calculate composite optimization score"""
    # Weight factors
    savings_weight = 0.4
    spend_weight = 0.3
    volatility_weight = 0.3
    
    # Normalize components (0-100 scale)
    savings_score = min(seasonal_savings * 2, 100)  # Cap at 50% savings
    spend_score = min(np.log10(annual_spend + 1) * 10, 100)  # Log scale for spend
    volatility_score = min(price_volatility, 100)  # Higher volatility = higher opportunity
    
    total_score = (savings_score * savings_weight + 
                  spend_score * spend_weight + 
                  volatility_score * volatility_weight)
    
    return total_score

def action_center(df_clean, available_optional):
    """Action center with specific recommendations and alerts"""
    st.subheader("⚡ Procurement Action Center")
    
    # Current month analysis
    current_month = datetime.now().month
    current_month_name = datetime.now().strftime('%B')
    
    # Immediate actions
    st.subheader("🚨 Immediate Actions Required")
    
    immediate_actions = []
    
    for item in df_clean['Item'].unique():
        item_data = df_clean[df_clean['Item'] == item]
        
        if len(item_data) >= 6:
            monthly_avg = item_data.groupby('Month')['Unit Price'].mean()
            current_month_price = monthly_avg.get(current_month, monthly_avg.mean())
            min_price = monthly_avg.min()
            
            # Check if current month is optimal
            price_diff = (current_month_price - min_price) / min_price * 100
            
            if price_diff <= 5:  # Within 5% of minimum
                action_type = "🟢 BUY NOW"
                priority = "High"
            elif price_diff <= 15:  # Within 15% of minimum
                action_type = "🟡 CONSIDER"
                priority = "Medium"
            else:
                action_type = "🔴 WAIT"
                priority = "Low"
            
            immediate_actions.append({
                'Item': item,
                'Action': action_type,
                'Priority': priority,
                'Current_Month_Price': current_month_price,
                'Min_Price': min_price,
                'Price_Difference_%': price_diff,
                'Best_Month': pd.to_datetime(f'2024-{monthly_avg.idxmin():02d}-01').strftime('%B')
            })
    
    action_df = pd.DataFrame(immediate_actions)
    action_df = action_df.sort_values(['Priority', 'Price_Difference_%'])
    
    # Show high priority actions
    high_priority = action_df[action_df['Priority'] == 'High']
    if len(high_priority) > 0:
        st.success(f"🎉 {len(high_priority)} items are at optimal prices this month!")
        st.dataframe(
            high_priority[['Item', 'Action', 'Current_Month_Price', 'Price_Difference_%']].style.format({
                'Current_Month_Price': '${:.2f}',
                'Price_Difference_%': '{:.1f}%'
            }),
            use_container_width=True
        )
    
    # Procurement calendar
    st.subheader("📅 Procurement Calendar")
    
    calendar_data = []
    for month in range(1, 13):
        month_name = pd.to_datetime(f'2024-{month:02d}-01').strftime('%B')
        month_items = []
        
        for item in df_clean['Item'].unique():
            item_data = df_clean[df_clean['Item'] == item]
            if len(item_data) >= 6:
                monthly_avg = item_data.groupby('Month')['Unit Price'].mean()
                if month in monthly_avg.index:
                    best_month = monthly_avg.idxmin()
                    if month == best_month:
                        month_items.append(item)
        
        calendar_data.append({
            'Month': month_name,
            'Optimal_Items': len(month_items),
            'Items': ', '.join(month_items[:3]) + ('...' if len(month_items) > 3 else '')
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    
    fig = px.bar(calendar_df, x='Month', y='Optimal_Items',
                title="Optimal Purchase Months by Item Count",
                labels={'Optimal_Items': 'Number of Items at Optimal Price'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Export all recommendations
    if st.button("📥 Export Complete Analysis"):
        # Combine all analysis results
        export_data = action_df
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="Download Procurement Recommendations",
            data=csv,
            file_name=f"procurement_optimization_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Example usage
if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced Seasonal Price Optimization", layout="wide")
    
    # Sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='W')
    
    sample_data = []
    items = ['Widget A', 'Widget B', 'Widget C', 'Component X', 'Material Y']
    vendors = ['Vendor 1', 'Vendor 2', 'Vendor 3']
    regions = ['North', 'South', 'East', 'West']
    
    for date in dates:
        for item in np.random.choice(items, size=np.random.randint(1, 4), replace=False):
            # Add seasonal pattern
            month = date.month
            seasonal_multiplier = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
            
            base_price = {'Widget A': 50, 'Widget B': 75, 'Widget C': 30, 
                         'Component X': 120, 'Material Y': 25}[item]
            
            price = base_price * seasonal_multiplier * np.random.uniform(0.8, 1.2)
            
            sample_data.append({
                'Creation Date': date,
                'Item': item,
                'Unit Price': price,
                'Qty Delivered': np.random.randint(1, 50),
                'Vendor': np.random.choice(vendors),
                'Region': np.random.choice(regions),
                'Lead Time Days': np.random.randint(7, 60)
            })
    
    df = pd.DataFrame(sample_data)
    df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    
    enhanced_display(df)
