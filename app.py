import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Clinical Data Analyzer", layout="wide")

st.title("Clinical Data Analyzer 🩺📊")
st.markdown("""
Upload your Excel or CSV file below. This app will help you analyze, visualize, and interpret your data step by step.
""")

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.subheader("Data Preview")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(20))
        st.markdown("**Columns:**")
        st.write(list(df.columns))

        # --- Single-Column Analysis Section ---
        st.markdown("---")
        st.header("🔎 Analyze a Column")
        col_selected = st.selectbox("Select a column to analyze", df.columns, key="single_col")

        col_data = df[col_selected]
        col_data_clean = col_data.dropna()
        is_numeric = pd.api.types.is_numeric_dtype(col_data_clean)
        n_unique = col_data_clean.nunique()
        if not is_numeric and n_unique <= 15:
            is_categorical = True
        elif is_numeric and n_unique <= 10:
            is_categorical = True
        else:
            is_categorical = not is_numeric

        # Single-column graph options
        single_graph_options = []
        if is_numeric and not is_categorical:
            single_graph_options = ["Histogram", "Boxplot", "Violin Plot", "KDE Plot", "Summary Stats"]
        else:
            single_graph_options = ["Bar Chart", "Pie Chart", "Donut Chart", "Value Counts Table"]
        graph_type = st.radio("Select graph type", single_graph_options, key="single_graph")

        # Layout: Graph and Interpretation side by side
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            if is_numeric and not is_categorical:
                if graph_type == "Histogram":
                    ax.hist(col_data_clean, bins=20, color="#3498db", edgecolor="black")
                    ax.set_xlabel(col_selected)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Histogram of {col_selected}")
                    st.pyplot(fig)
                elif graph_type == "Boxplot":
                    sns.boxplot(x=col_data_clean, ax=ax, color="#e67e22")
                    ax.set_title(f"Boxplot of {col_selected}")
                    st.pyplot(fig)
                elif graph_type == "Violin Plot":
                    sns.violinplot(x=col_data_clean, ax=ax, color="#8e44ad")
                    ax.set_title(f"Violin Plot of {col_selected}")
                    st.pyplot(fig)
                elif graph_type == "KDE Plot":
                    sns.kdeplot(col_data_clean, ax=ax, fill=True, color="#16a085")
                    ax.set_title(f"KDE Plot of {col_selected}")
                    st.pyplot(fig)
            else:
                value_counts = col_data_clean.value_counts()
                if graph_type == "Bar Chart":
                    orientation = st.radio("Bar Chart Orientation", ["Vertical", "Horizontal"], horizontal=True, key="bar_orientation")
                    if orientation == "Vertical":
                        value_counts.plot(kind="bar", ax=ax, color="#2ecc71")
                        ax.set_xlabel(col_selected)
                        ax.set_ylabel("Count")
                        ax.set_title(f"Bar Chart of {col_selected}")
                        st.pyplot(fig)
                    else:
                        value_counts.plot(kind="barh", ax=ax, color="#2ecc71")
                        ax.set_ylabel(col_selected)
                        ax.set_xlabel("Count")
                        ax.set_title(f"Horizontal Bar Chart of {col_selected}")
                        st.pyplot(fig)
                elif graph_type == "Pie Chart":
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    value_counts.plot(kind="pie", ax=ax2, autopct="%1.1f%%", startangle=90, legend=False)
                    ax2.set_ylabel("")
                    ax2.set_title(f"Pie Chart of {col_selected}")
                    st.pyplot(fig2)
                elif graph_type == "Donut Chart":
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    wedges, texts, autotexts = ax2.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4))
                    ax2.set_ylabel("")
                    ax2.set_title(f"Donut Chart of {col_selected}")
                    st.pyplot(fig2)
        with col2:
            st.subheader("Summary / Interpretation")
            if is_numeric and not is_categorical and graph_type == "Summary Stats":
                st.write(col_data_clean.describe().to_frame())
            elif not (is_numeric and not is_categorical) and graph_type == "Value Counts Table":
                st.write(col_data_clean.value_counts().to_frame(name="Count"))
            # Text area for user notes/interpretation
            user_notes = st.text_area("Add your interpretation or notes (optional)", key="notes")
            if user_notes:
                st.info(user_notes)

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload an Excel or CSV file to get started.") 