import streamlit as st
import pandas as pd

def display(df):
    st.header("Smart Reorder Point Prediction")

    # User selects an item to analyze
    item = st.selectbox("Select Item", df["Item"].dropna().unique())

    # Filter the dataset for that item
    item_df = df[df["Item"] == item].copy()

    # Group by month and sum quantity delivered
    item_df["Month"] = pd.to_datetime(item_df["Creation Date"]).dt.to_period("M")
    demand_by_month = item_df.groupby("Month")["Qty Delivered"].sum().fillna(0)

    # Calculate reorder point using basic statistical method
    reorder_point = demand_by_month.mean() + demand_by_month.std()

    # Display metrics
    st.metric("Suggested Reorder Point", round(reorder_point, 2))
    st.line_chart(demand_by_month)
