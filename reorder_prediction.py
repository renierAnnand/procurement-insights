# reorder_prediction.py

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import plotly.express as px
import io


def calculate_forecasted_rop(item_df, forecast_days=10, z_score=1.65):
    if len(item_df) == 0:
        return None

    item_df = item_df.copy()
    item_df['Creation Date'] = pd.to_datetime(item_df['Creation Date'])
    item_df = item_df[item_df['Qty Delivered'] > 0]
    item_df = item_df.sort_values('Creation Date')

    full_date_range = pd.date_range(
        start=item_df['Creation Date'].min(),
        end=item_df['Creation Date'].max(),
        freq='D'
    )
    daily_demand = item_df.groupby('Creation Date')['Qty Delivered'].sum()
    daily_demand = daily_demand.reindex(full_date_range, fill_value=0)

    if len(daily_demand) < 10:
        return None

    model = ExponentialSmoothing(daily_demand, trend='add', seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(forecast_days)
    forecast_total = forecast.sum()

    lead_time_std = item_df['Creation Date'].diff().dt.days.std()
    avg_daily_demand = daily_demand.mean()
    demand_std = daily_demand.std()
    safety_stock = z_score * avg_daily_demand * (lead_time_std if not np.isnan(lead_time_std) else 1)
    demand_cv = demand_std / avg_daily_demand if avg_daily_demand > 0 else 0

    return {
        'forecasted_rop': forecast_total + safety_stock,
        'forecast_total': forecast_total,
        'safety_stock': safety_stock,
        'avg_daily_demand': avg_daily_demand,
        'lead_time_std': lead_time_std,
        'demand_cv': demand_cv,
        'daily_demand_series': daily_demand,
        'forecast_series': forecast
    }


def bulk_forecast_analysis(df):
    results = []
    items = df['Item'].dropna().unique()
    for item in items:
        item_df = df[df['Item'] == item]
        metrics = calculate_forecasted_rop(item_df)
        if metrics:
            lead_time_risk = "High" if metrics['lead_time_std'] > 10 else "Medium" if metrics['lead_time_std'] > 5 else "Low"
            priority_score = (
                metrics['avg_daily_demand'] * 0.4 +
                metrics['forecast_total'] * 0.3 +
                metrics['demand_cv'] * 100 * 0.3
            )
            if priority_score >= 80:
                priority = "Critical"
            elif priority_score >= 50:
                priority = "High"
            elif priority_score >= 25:
                priority = "Medium"
            else:
                priority = "Low"

            results.append({
                'Item': item,
                'Avg Daily Demand': round(metrics['avg_daily_demand'], 2),
                'Lead Time Std': round(metrics['lead_time_std'], 2),
                'Demand CV': round(metrics['demand_cv'], 2),
                'Lead Time Risk': lead_time_risk,
                'Priority': priority,
                'Priority Score': round(priority_score, 1),
                'Safety Stock': round(metrics['safety_stock'], 2),
                'Forecast Total (Next 10 days)': round(metrics['forecast_total'], 2),
                'Forecasted ROP': round(metrics['forecasted_rop'], 2)
            })
    return pd.DataFrame(results)


def display(df):
    st.title("📦 Forecast-Based Reorder Point Analysis")

    if 'Item' not in df.columns or 'Creation Date' not in df.columns or 'Qty Delivered' not in df.columns:
        st.error("Required columns: Item, Creation Date, Qty Delivered")
        return

    mode = st.radio("Select Analysis Mode", ["Single Item", "Bulk Analysis"])

    if mode == "Single Item":
        item = st.selectbox("Select Item", df['Item'].dropna().unique())
        item_df = df[df['Item'] == item].copy()

        if item_df.empty:
            st.warning("No data for selected item.")
            return

        forecast_data = calculate_forecasted_rop(item_df)

        if not forecast_data:
            st.warning("Not enough data to perform forecast.")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Daily Demand", f"{forecast_data['avg_daily_demand']:.2f}")
        with col2:
            st.metric("Safety Stock", f"{forecast_data['safety_stock']:.2f}")
        with col3:
            st.metric("Forecasted ROP", f"{forecast_data['forecasted_rop']:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_data['daily_demand_series'].index,
            y=forecast_data['daily_demand_series'].values,
            name="Historical Demand",
            line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast_series'].index,
            y=forecast_data['forecast_series'].values,
            name="Forecasted Demand",
            line=dict(color="orange", dash='dash')
        ))
        fig.update_layout(
            title=f"📈 Demand Forecast for {item}",
            xaxis_title="Date",
            yaxis_title="Qty Delivered",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    elif mode == "Bulk Analysis":
        st.info("This will analyze all items and generate an Excel report")
        if st.button("🚀 Run Bulk Analysis", type="primary"):
            with st.spinner("Analyzing items..."):
                results_df = bulk_forecast_analysis(df)
                st.success(f"Completed analysis for {len(results_df)} items.")
                st.dataframe(results_df, use_container_width=True)

                st.subheader("📊 Dashboard Overview")
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Total Items", len(results_df))
                with k2:
                    st.metric("Critical Items", len(results_df[results_df['Priority'] == 'Critical']))
                with k3:
                    st.metric("Avg Safety Stock", f"{results_df['Safety Stock'].mean():.2f}")
                with k4:
                    st.metric("High Lead Time Risk", len(results_df[results_df['Lead Time Risk'] == 'High']))

                st.subheader("🎯 Priority Distribution")
                fig_pie = px.pie(results_df, names='Priority', title='Item Priority Breakdown')
                st.plotly_chart(fig_pie, use_container_width=True)

                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='ROP Forecast')

                    summary_df = pd.DataFrame({
                        'Metric': ['Total Items Analyzed', 'Total Forecasted ROP', 'Average Safety Stock'],
                        'Value': [
                            len(results_df),
                            results_df['Forecasted ROP'].sum(),
                            results_df['Safety Stock'].mean()
                        ]
                    })
                    summary_df.to_excel(writer, index=False, sheet_name='Summary')

                    high_risk_df = results_df[(results_df['Lead Time Risk'] == 'High') | (results_df['Demand CV'] > 0.5)]
                    if not high_risk_df.empty:
                        high_risk_df.to_excel(writer, index=False, sheet_name='Risk Assessment')

                    action_plan = results_df.copy()
                    action_plan['Recommended Action'] = np.where(
                        action_plan['Priority'].isin(['Critical', 'High']),
                        'Order immediately',
                        'Monitor demand closely'
                    )
                    action_plan.to_excel(writer, index=False, sheet_name='Action Plan')

                excel_buffer.seek(0)

                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_buffer,
                    file_name=f"forecast_rop_bulk_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


if __name__ == '__main__':
    st.set_page_config(page_title="Forecast-Based ROP", layout="wide")

    sample_data = {
        'Creation Date': pd.date_range('2023-01-01', periods=120, freq='D'),
        'Item': ['Sample Item'] * 120,
        'Qty Delivered': np.random.poisson(5, 120)
    }
    test_df = pd.DataFrame(sample_data)

    display(test_df)
