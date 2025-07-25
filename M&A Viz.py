import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MAAI as train
import tempfile

st.set_page_config(page_title="M&A Synergy Live Pipeline Dashboard", layout="wide")
st.title("📊 M&A Synergy Monte Carlo Analysis - Live Upload with Visualization")

uploaded_file = st.file_uploader("Upload your acquisition CSV file for live processing:", type=["csv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing uploaded file with Monte Carlo simulation..."):
        df, model_df, _ = train.run_mna_pipeline(tmp_path, save_trace=False, train_model=False)

    st.success("✅ File processed successfully!")

    # Sidebar filters
    st.sidebar.header("Filters")
    parent_filter = st.sidebar.selectbox("Parent Company", ["All"] + sorted(df['Parent Company'].unique()))
    category_filter = st.sidebar.selectbox("Category", ["All"] + sorted(df['Category'].unique()))

    filtered_df = df.copy()
    filtered_model_df = model_df.copy()
    if parent_filter != "All":
        filtered_df = filtered_df[filtered_df['Parent Company'] == parent_filter]
        filtered_model_df = filtered_model_df[filtered_model_df['Parent Company'] == parent_filter]
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
        filtered_model_df = filtered_model_df[filtered_model_df['Category'] == category_filter]

    st.subheader("Top Synergy Deals")
    st.dataframe(
        filtered_df[['Parent Company', 'Acquired Company', 'Business', 'Synergy Score']]
        .sort_values(['Parent Company', 'Synergy Score'], ascending=[True, False])
        .groupby('Parent Company')
        .head(3),
        use_container_width=True
    )

    st.subheader("Top Deals by Simulated Fair Value")
    st.dataframe(
        filtered_model_df[['Acquired Company', 'Acquisition Price', 'Simulated Fair Value Mean',
                  'Simulated Fair Value VaR_5', 'Simulated Fair Value VaR_95', 'P(Fair Value > Price)']]
        .sort_values(by='Simulated Fair Value Mean', ascending=False)
        .head(10),
        use_container_width=True
    )

    st.subheader("📈 Fair Value vs Acquisition Price")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=filtered_model_df,
        x='Simulated Fair Value Mean',
        y='Acquisition Price',
        hue='P(Fair Value > Price)',
        size='Synergy Score',
        palette='viridis',
        ax=ax1
    )
    ax1.plot([0, filtered_model_df['Acquisition Price'].max()], [0, filtered_model_df['Acquisition Price'].max()], 'r--')
    ax1.set_title("Simulated Fair Value vs Acquisition Price")
    st.pyplot(fig1)

    st.subheader("📊 Distribution of Simulated Fair Values")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_model_df['Simulated Fair Value Mean'], kde=True, ax=ax2)
    ax2.set_title("Distribution of Simulated Fair Values")
    st.pyplot(fig2)

    st.download_button(
        "Download Processed Results",
        filtered_model_df.to_csv(index=False).encode('utf-8'),
        file_name="processed_mna_results.csv",
        mime="text/csv"
    )

    st.caption("Built with ❤️ using Streamlit and Monte Carlo for live M&A synergy valuation with company-specific filters and visual insights.")
else:
    st.info("Upload a CSV file to start the live M&A synergy analysis.")