import streamlit as st
from fuzzywuzzy import fuzz
import pandas as pd

def get_fuzzy_matches(series, threshold=90):
    pairs = []
    values = series.dropna().unique()
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            score = fuzz.token_sort_ratio(values[i], values[j])
            if score > threshold:
                pairs.append((values[i], values[j], score))
    return pd.DataFrame(pairs, columns=["Name 1", "Name 2", "Similarity Score"])

def display(df):
    st.header("Duplicate Vendor/Item Detection")

    option = st.radio("Check Duplicates For:", ["Vendor Name", "Item Description"])

    matches = get_fuzzy_matches(df[option])

    st.write(f"Potential Duplicates in {option}:")
    st.dataframe(matches)
