import streamlit as st

def display(df):
    st.header("Cross-Region Vendor Optimization")

    # Select item to analyze across regions/vendors
    item = st.selectbox("Select Item", df["Item"].dropna().unique())
    filtered = df[df["Item"] == item]

    # Group by vendor and warehouse to compare pricing
    result = (
        filtered.groupby(["Vendor Name", "W/H"])["Unit Price"]
        .mean()
        .reset_index()
        .sort_values(by="Unit Price")
    )

    # Show results
    st.write("Average Unit Price by Vendor and Warehouse:")
    st.dataframe(result)
