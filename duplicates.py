import streamlit as st
from fuzzywuzzy import fuzz
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def clean_text(text):
    """Clean text for better matching"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_fuzzy_matches(series, threshold=80):
    """Find fuzzy matches using multiple algorithms"""
    pairs = []
    values = series.dropna().unique()
    
    # Progress bar
    progress_bar = st.progress(0)
    total_comparisons = len(values) * (len(values) - 1) // 2
    current_comparison = 0
    
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            current_comparison += 1
            progress_bar.progress(current_comparison / total_comparisons)
            
            val1, val2 = values[i], values[j]
            
            # Multiple fuzzy matching algorithms
            token_sort = fuzz.token_sort_ratio(val1, val2)
            token_set = fuzz.token_set_ratio(val1, val2)
            partial = fuzz.partial_ratio(val1, val2)
            ratio = fuzz.ratio(val1, val2)
            
            # Use the maximum score
            max_score = max(token_sort, token_set, partial, ratio)
            
            if max_score >= threshold:
                pairs.append({
                    'Name 1': val1,
                    'Name 2': val2,
                    'Similarity Score': max_score,
                    'Token Sort': token_sort,
                    'Token Set': token_set,
                    'Partial': partial,
                    'Ratio': ratio,
                    'Method': 'Fuzzy String'
                })
    
    progress_bar.empty()
    return pd.DataFrame(pairs)

def get_semantic_matches(series, threshold=0.7):
    """Find semantic matches using TF-IDF and cosine similarity"""
    values = series.dropna().unique()
    cleaned_values = [clean_text(val) for val in values]
    
    # Remove empty strings
    non_empty_indices = [i for i, val in enumerate(cleaned_values) if val]
    if len(non_empty_indices) < 2:
        return pd.DataFrame()
    
    filtered_values = [values[i] for i in non_empty_indices]
    filtered_cleaned = [cleaned_values[i] for i in non_empty_indices]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(filtered_cleaned)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    pairs = []
    for i in range(len(filtered_values)):
        for j in range(i+1, len(filtered_values)):
            similarity = cosine_sim[i, j]
            if similarity >= threshold:
                pairs.append({
                    'Name 1': filtered_values[i],
                    'Name 2': filtered_values[j],
                    'Similarity Score': round(similarity * 100, 2),
                    'Method': 'Semantic'
                })
    
    return pd.DataFrame(pairs)

def analyze_duplicate_impact(df, duplicates_df, column_name):
    """Analyze the business impact of duplicates"""
    impact_data = []
    
    for _, row in duplicates_df.iterrows():
        name1, name2 = row['Name 1'], row['Name 2']
        
        # Get data for both potential duplicates
        data1 = df[df[column_name] == name1]
        data2 = df[df[column_name] == name2]
        
        # Calculate metrics
        total_pos = len(data1) + len(data2)
        total_value = data1['Line Total'].sum() + data2['Line Total'].sum() if 'Line Total' in df.columns else 0
        
        impact_data.append({
            'Name 1': name1,
            'Name 2': name2,
            'Similarity Score': row['Similarity Score'],
            'Total POs': total_pos,
            'Total Value': total_value,
            'POs Name 1': len(data1),
            'POs Name 2': len(data2),
            'Value Name 1': data1['Line Total'].sum() if 'Line Total' in df.columns else 0,
            'Value Name 2': data2['Line Total'].sum() if 'Line Total' in df.columns else 0
        })
    
    return pd.DataFrame(impact_data)

def display(df):
    st.header("🔍 Duplicate Vendor/Item Detection")
    st.markdown("Identify and analyze potential duplicates in master data using advanced fuzzy matching and semantic analysis.")
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_type = st.selectbox(
            "Detection Method",
            ["Fuzzy String Matching", "Semantic Analysis", "Combined Analysis"],
            key="detection_type"
        )
    
    with col2:
        threshold = st.slider(
            "Similarity Threshold",
            min_value=50,
            max_value=95,
            value=80,
            step=5,
            help="Higher values = more strict matching"
        )
    
    with col3:
        analysis_column = st.selectbox(
            "Analyze Duplicates In",
            ["Vendor Name", "Item Description", "Supplier Site"],
            key="analysis_column"
        )
    
    if analysis_column not in df.columns:
        st.error(f"Column '{analysis_column}' not found in the dataset.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Duplicate Detection", "📊 Impact Analysis", "🛠️ Data Quality", "📋 Recommendations"])
    
    with tab1:
        st.subheader(f"Duplicate Detection in {analysis_column}")
        
        # Run detection based on selected method
        matches_df = pd.DataFrame()
        
        if st.button(f"🔍 Find Duplicates in {analysis_column}", type="primary"):
            with st.spinner("Analyzing for duplicates..."):
                if detection_type == "Fuzzy String Matching":
                    matches_df = get_fuzzy_matches(df[analysis_column], threshold)
                elif detection_type == "Semantic Analysis":
                    matches_df = get_semantic_matches(df[analysis_column], threshold/100)
                else:  # Combined Analysis
                    fuzzy_matches = get_fuzzy_matches(df[analysis_column], threshold)
                    semantic_matches = get_semantic_matches(df[analysis_column], threshold/100)
                    matches_df = pd.concat([fuzzy_matches, semantic_matches], ignore_index=True)
                    matches_df = matches_df.drop_duplicates(subset=['Name 1', 'Name 2'])
            
            if len(matches_df) > 0:
                st.success(f"Found {len(matches_df)} potential duplicate pairs!")
                
                # Display results with formatting
                display_df = matches_df.copy()
                if 'Similarity Score' in display_df.columns:
                    display_df = display_df.sort_values('Similarity Score', ascending=False)
                
                # Color code by similarity score
                def color_score(val):
                    if val >= 90:
                        return 'background-color: #ff9999'  # Red for high similarity
                    elif val >= 80:
                        return 'background-color: #ffcc99'  # Orange for medium
                    else:
                        return 'background-color: #ffffcc'  # Yellow for lower
                
                styled_df = display_df.style.applymap(
                    color_score, 
                    subset=['Similarity Score'] if 'Similarity Score' in display_df.columns else []
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Visualization
                if len(matches_df) > 0:
                    fig = px.histogram(
                        matches_df,
                        x='Similarity Score',
                        nbins=20,
                        title=f"Distribution of Similarity Scores for {analysis_column}",
                        labels={'count': 'Number of Pairs', 'Similarity Score': 'Similarity Score'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Export option
                csv = matches_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Duplicates Report",
                    data=csv,
                    file_name=f"duplicates_{analysis_column}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Store results in session state for other tabs
                st.session_state['matches_df'] = matches_df
                st.session_state['analysis_column'] = analysis_column
                
            else:
                st.info("No potential duplicates found with the current threshold.")
    
    with tab2:
        st.subheader("📊 Business Impact Analysis")
        
        if 'matches_df' in st.session_state and len(st.session_state['matches_df']) > 0:
            matches_df = st.session_state['matches_df']
            analysis_column = st.session_state['analysis_column']
            
            # Calculate business impact
            with st.spinner("Calculating business impact..."):
                impact_df = analyze_duplicate_impact(df, matches_df, analysis_column)
            
            if len(impact_df) > 0:
                # Summary metrics
                total_affected_pos = impact_df['Total POs'].sum()
                total_affected_value = impact_df['Total Value'].sum()
                high_impact_pairs = len(impact_df[impact_df['Total Value'] > impact_df['Total Value'].quantile(0.8)])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Affected POs", f"{total_affected_pos:,}")
                with col2:
                    st.metric("Affected Value", f"{total_affected_value:,.2f}")
                with col3:
                    st.metric("High Impact Pairs", high_impact_pairs)
                
                # Top impact duplicates
                st.subheader("🎯 Highest Impact Duplicate Pairs")
                top_impact = impact_df.nlargest(20, 'Total Value')
                
                st.dataframe(
                    top_impact.style.format({
                        'Total Value': '{:,.2f}',
                        'Value Name 1': '{:,.2f}',
                        'Value Name 2': '{:,.2f}',
                        'Similarity Score': '{:.1f}'
                    }),
                    use_container_width=True
                )
                
                # Impact visualization
                fig = px.scatter(
                    impact_df,
                    x='Similarity Score',
                    y='Total Value',
                    size='Total POs',
                    color='Total POs',
                    hover_data=['Name 1', 'Name 2'],
                    title="Duplicate Impact: Similarity vs Business Value",
                    labels={'Total Value': 'Total Business Value', 'Total POs': 'Number of POs'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run duplicate detection first to see impact analysis.")
    
    with tab3:
        st.subheader("🛠️ Data Quality Assessment")
        
        # Overall data quality metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Completeness:**")
            completeness = {}
            key_columns = ['Vendor Name', 'Item Description', 'Supplier Site']
            
            for col in key_columns:
                if col in df.columns:
                    non_null_pct = (df[col].notna().sum() / len(df)) * 100
                    completeness[col] = non_null_pct
            
            completeness_df = pd.DataFrame(list(completeness.items()), columns=['Column', 'Completeness %'])
            
            fig = px.bar(
                completeness_df,
                x='Column',
                y='Completeness %',
                title="Data Completeness by Column",
                color='Completeness %',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Data Uniqueness:**")
            uniqueness = {}
            
            for col in key_columns:
                if col in df.columns:
                    total_records = df[col].notna().sum()
                    unique_records = df[col].nunique()
                    uniqueness_pct = (unique_records / total_records) * 100 if total_records > 0 else 0
                    uniqueness[col] = uniqueness_pct
            
            uniqueness_df = pd.DataFrame(list(uniqueness.items()), columns=['Column', 'Uniqueness %'])
            
            fig = px.bar(
                uniqueness_df,
                x='Column',
                y='Uniqueness %',
                title="Data Uniqueness by Column",
                color='Uniqueness %',
                color_continuous_scale='RdYlBu'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data quality issues
        st.subheader("🚨 Data Quality Issues")
        
        issues = []
        
        # Check for common data quality issues
        for col in key_columns:
            if col in df.columns:
                # Missing values
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    issues.append({
                        'Column': col,
                        'Issue': 'Missing Values',
                        'Count': missing_count,
                        'Percentage': f"{(missing_count/len(df)*100):.1f}%"
                    })
                
                # Very short values (likely incomplete)
                short_values = df[col].astype(str).str.len() < 3
                short_count = short_values.sum()
                if short_count > 0:
                    issues.append({
                        'Column': col,
                        'Issue': 'Very Short Values',
                        'Count': short_count,
                        'Percentage': f"{(short_count/len(df)*100):.1f}%"
                    })
        
        if issues:
            issues_df = pd.DataFrame(issues)
            st.dataframe(issues_df, use_container_width=True)
        else:
            st.success("No major data quality issues detected!")
    
    with tab4:
        st.subheader("📋 Data Cleanup Recommendations")
        
        if 'matches_df' in st.session_state and len(st.session_state['matches_df']) > 0:
            matches_df = st.session_state['matches_df']
            
            st.write("**Recommended Actions:**")
            
            # Categorize recommendations by similarity score
            high_confidence = matches_df[matches_df['Similarity Score'] >= 90]
            medium_confidence = matches_df[(matches_df['Similarity Score'] >= 80) & (matches_df['Similarity Score'] < 90)]
            low_confidence = matches_df[matches_df['Similarity Score'] < 80]
            
            if len(high_confidence) > 0:
                st.markdown("#### 🔴 High Confidence Duplicates (90%+ similarity)")
                st.markdown("**Action:** Merge immediately after manual verification")
                st.dataframe(high_confidence[['Name 1', 'Name 2', 'Similarity Score']], use_container_width=True)
            
            if len(medium_confidence) > 0:
                st.markdown("#### 🟡 Medium Confidence Duplicates (80-89% similarity)")
                st.markdown("**Action:** Review and merge if confirmed as duplicates")
                st.dataframe(medium_confidence[['Name 1', 'Name 2', 'Similarity Score']], use_container_width=True)
            
            if len(low_confidence) > 0:
                st.markdown("#### 🟢 Low Confidence Matches (<80% similarity)")
                st.markdown("**Action:** Investigate for data standardization opportunities")
                st.dataframe(low_confidence[['Name 1', 'Name 2', 'Similarity Score']], use_container_width=True)
            
            # Generate master data cleanup template
            if st.button("📋 Generate Cleanup Template"):
                cleanup_template = []
                
                for _, row in matches_df.iterrows():
                    cleanup_template.append({
                        'Original Name 1': row['Name 1'],
                        'Original Name 2': row['Name 2'],
                        'Similarity Score': row['Similarity Score'],
                        'Recommended Action': 'MERGE' if row['Similarity Score'] >= 85 else 'REVIEW',
                        'Master Name': '',  # To be filled by user
                        'Status': 'PENDING',
                        'Comments': ''
                    })
                
                template_df = pd.DataFrame(cleanup_template)
                csv = template_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Cleanup Template",
                    data=csv,
                    file_name=f"cleanup_template_{analysis_column}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("Run duplicate detection first to see recommendations.")
        
        # General data quality recommendations
        st.markdown("#### 💡 General Data Quality Recommendations")
        
        recommendations = [
            "**Standardize naming conventions** - Establish clear guidelines for vendor and item naming",
            "**Implement data validation** - Add validation rules during data entry",
            "**Regular cleanup cycles** - Schedule monthly duplicate detection and cleanup",
            "**Master data governance** - Assign data stewards for different categories",
            "**Automated matching** - Implement real-time duplicate detection during data entry"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
