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

def wrap_labels(labels, width=30):
    # Helper to wrap long labels for matplotlib
    import textwrap
    return ['\n'.join(textwrap.wrap(str(l), width)) for l in labels]

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

        # --- Categorical ordering and top N options ---
        if not (is_numeric and not is_categorical):
            st.markdown("**Category Display Options:**")
            order_by = st.selectbox(
                "Order categories by",
                ["Frequency (Descending)", "Frequency (Ascending)", "Label (A-Z)", "Label (Z-A)"],
                key="order_by"
            )
            max_n = st.number_input(
                "Show Top N Categories (rest grouped as 'Other')",
                min_value=2, max_value=n_unique, value=min(10, n_unique), step=1, key="top_n"
            )
            # Compute value counts
            value_counts = col_data_clean.value_counts()
            # Apply ordering
            if order_by == "Frequency (Descending)":
                value_counts = value_counts.sort_values(ascending=False)
            elif order_by == "Frequency (Ascending)":
                value_counts = value_counts.sort_values(ascending=True)
            elif order_by == "Label (A-Z)":
                value_counts = value_counts.sort_index(ascending=True)
            elif order_by == "Label (Z-A)":
                value_counts = value_counts.sort_index(ascending=False)
            # Apply top N
            if len(value_counts) > max_n:
                top = value_counts.iloc[:max_n]
                other = value_counts.iloc[max_n:].sum()
                value_counts = pd.concat([top, pd.Series({'Other': other})])
            # Wrap long labels
            plot_labels = wrap_labels(value_counts.index)
        else:
            value_counts = None
            plot_labels = None

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
                if graph_type == "Bar Chart":
                    orientation = st.radio("Bar Chart Orientation", ["Vertical", "Horizontal"], horizontal=True, key="bar_orientation")
                    percentages = (value_counts.values / value_counts.values.sum() * 100).round(1)
                    if orientation == "Vertical":
                        bars = ax.bar(plot_labels, value_counts.values, color="#2ecc71", alpha=0.7)
                        ax.set_xlabel(col_selected)
                        ax.set_ylabel("Count")
                        ax.set_title(f"Bar Chart of {col_selected}")
                        plt.xticks(rotation=30, ha='right')
                        # Add percentage labels on top of bars
                        for bar, pct in zip(bars, percentages):
                            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct}%", ha='center', va='bottom', fontsize=10)
                        st.pyplot(fig)
                    else:
                        bars = ax.barh(plot_labels, value_counts.values, color="#2ecc71", alpha=0.7)
                        ax.set_ylabel(col_selected)
                        ax.set_xlabel("Count")
                        ax.set_title(f"Horizontal Bar Chart of {col_selected}")
                        # Add percentage labels to the right of bars
                        for bar, pct in zip(bars, percentages):
                            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{pct}%", ha='left', va='center', fontsize=10)
                        st.pyplot(fig)
                elif graph_type == "Pie Chart":
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    ax2.pie(value_counts.vtalues, labels=plot_labels, autopct="%1.1f%%", startangle=90)
                    ax2.set_ylabel("")
                    ax2.set_title(f"Pie Chart of {col_selected}")
                    st.pyplot(fig2)
                elif graph_type == "Donut Chart":
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    wedges, texts, autotexts = ax2.pie(value_counts.values, labels=plot_labels, autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4))
                    ax2.set_ylabel("")
                    ax2.set_title(f"Donut Chart of {col_selected}")
                    st.pyplot(fig2)
        with col2:
            st.subheader("Summary / Interpretation")
            # Show summary table for categorical columns
            if not (is_numeric and not is_categorical):
                # Prepare summary table
                summary_df = value_counts.reset_index()
                summary_df.columns = ["Option", "Responses"]
                summary_df["Description"] = summary_df["Option"]  # Placeholder, can be customized
                summary_df["Percentage (%)"] = (summary_df["Responses"] / summary_df["Responses"].sum() * 100).round(2)
                summary_df = summary_df[["Option", "Description", "Responses", "Percentage (%)"]]
                st.markdown("**Response Table:**")
                st.dataframe(summary_df, use_container_width=True)
            if is_numeric and not is_categorical and graph_type == "Summary Stats":
                st.write(col_data_clean.describe().to_frame())
            elif not (is_numeric and not is_categorical) and graph_type == "Value Counts Table":
                st.write(value_counts.to_frame(name="Count"))
            # Text area for user notes/interpretation
            user_notes = st.text_area("Add your interpretation or notes (optional)", key="notes")
            if user_notes:
                st.info(user_notes)

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload an Excel or CSV file to get started.") 