import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import textwrap

def wrap_labels(labels, width=30):
    # Helper to wrap long labels for matplotlib
    return ['\n'.join(textwrap.wrap(str(l), width)) for l in labels]

def sanitize_dataframe_for_streamlit(df):
    for col in df.columns:
        if df[col].dtype == 'O' and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

# --- Logo/banner ---
def get_logo():
    logo_url = "https://img.icons8.com/fluency/96/medical-doctor.png"  # Placeholder medical icon
    return f'<img src="{logo_url}" width="60" style="margin-bottom:10px;">'

st.set_page_config(page_title="Clinical Data Analyzer 🩺📊", page_icon="🩺", layout="wide")

with st.sidebar:
    st.markdown(get_logo(), unsafe_allow_html=True)
    st.markdown("<h2 style='margin-bottom:0;'>Clinical Data Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("<small>by Rahul | Powered by Streamlit</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Upload your data and follow the steps in each tab to analyze and interpret your clinical dataset.")

# --- Main Tabs ---
tabs = st.tabs([
    "📁 Data Upload",
    "🔍 Explore",
    "📊 Visualize",
    "🧪 Statistical Tests",
    "📝 Interpretation/Report"
])

# --- Data Upload Tab ---
with tabs[0]:
    st.header("📁 Data Upload")
    st.markdown("Upload your Excel or CSV file. Supported formats: .xlsx, .xls, .csv")
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"], help="Supported formats: .xlsx, .xls, .csv")
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(sanitize_dataframe_for_streamlit(df.head(20)), use_container_width=True)
            st.markdown("**Columns:**")
            st.write(list(df.columns))
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Please upload an Excel or CSV file to get started.")

# --- Early exit if no data ---
if 'df' not in locals() or df is None:
    st.stop()

# --- Explore Tab ---
with tabs[1]:
    st.header("🔍 Data Exploration")
    st.markdown("Get a quick overview of your data. Select a column to see its summary.")
    col_selected = st.selectbox("Select a column to explore", df.columns, key="explore_col")
    st.write(sanitize_dataframe_for_streamlit(df[col_selected].describe(include='all').to_frame()))
    st.write("**Unique values:**", df[col_selected].unique())
    st.write("**Missing values:**", df[col_selected].isna().sum())

# --- Visualize Tab ---
with tabs[2]:
    st.header("📊 Visualization")
    st.markdown("Visualize distributions and relationships. Choose a column and graph type.")
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

        # --- User customization for color and display name ---
        st.sidebar.markdown(f"### Customize Categories for '{col_selected}'")
        custom_colors = {}
        custom_labels = {}
        default_palette = sns.color_palette("Set2", n_colors=len(value_counts))
        for i, cat in enumerate(value_counts.index):
            color_key = f"color_{col_selected}_{cat}"
            label_key = f"label_{col_selected}_{cat}"
            default_color = '#%02x%02x%02x' % tuple(int(255*x) for x in default_palette[i])
            custom_colors[cat] = st.sidebar.color_picker(f"Color for '{cat}'", value=default_color, key=color_key)
            custom_labels[cat] = st.sidebar.text_input(f"Label for '{cat}'", value=str(cat), key=label_key)
        # Use custom labels for plotting and tables
        plot_labels = [custom_labels[cat] for cat in value_counts.index]
        category_colors = [custom_colors[cat] for cat in value_counts.index]

        # --- Manual grouping for clinical interpretation ---
        st.sidebar.markdown(f"### Clinical Interpretation Groups for '{col_selected}'")
        max_n_groups = len(value_counts)
        default_n_groups = min(3, max_n_groups)
        group_count = st.sidebar.number_input(
            "Number of groups",
            min_value=1,
            max_value=max_n_groups,
            value=default_n_groups,
            key="group_count"
        )
        groupings = []
        used_cats = set()
        for i in range(group_count):
            group_label = st.sidebar.text_input(f"Group {i+1} label", value=f"Group {i+1}", key=f"group_label_{i}")
            group_cats = st.sidebar.multiselect(f"Categories in {group_label}", [cat for cat in value_counts.index if cat not in used_cats], key=f"group_cats_{i}")
            group_interp = st.sidebar.text_input(f"Interpretation for {group_label}", value="", key=f"group_interp_{i}")
            groupings.append({"label": group_label, "cats": group_cats, "interp": group_interp})
            used_cats.update(group_cats)
    else:
        value_counts = None
        plot_labels = None
        category_colors = None

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
                threshold = 0.1 * max(value_counts.values)  # 10% of max bar length
                if orientation == "Vertical":
                    bars = ax.bar(plot_labels, value_counts.values, color=category_colors, alpha=0.7)
                    ax.set_xlabel(col_selected)
                    ax.set_ylabel("Count")
                    ax.set_title(f"Bar Chart of {col_selected}")
                    plt.xticks(rotation=30, ha='right')
                    # Add percentage labels inside or outside bars
                    for bar, pct, val in zip(bars, percentages, value_counts.values):
                        if val > threshold:
                            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5, f"{pct}%", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
                        else:
                            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct}%", ha='center', va='bottom', fontsize=10, color='black')
                    st.pyplot(fig)
                else:
                    bars = ax.barh(plot_labels, value_counts.values, color=category_colors, alpha=0.7)
                    ax.set_ylabel(col_selected)
                    ax.set_xlabel("Count")
                    ax.set_title(f"Horizontal Bar Chart of {col_selected}")
                    # Add percentage labels inside or outside bars
                    for bar, pct, val in zip(bars, percentages, value_counts.values):
                        if val > threshold:
                            ax.text(bar.get_width() * 0.5, bar.get_y() + bar.get_height() / 2, f"{pct}%", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
                        else:
                            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{pct}%", ha='left', va='center', fontsize=10, color='black')
                    st.pyplot(fig)
            elif graph_type == "Pie Chart":
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.pie(value_counts.values, labels=plot_labels, autopct="%1.1f%%", startangle=90, colors=category_colors)
                ax2.set_ylabel("")
                ax2.set_title(f"Pie Chart of {col_selected}")
                st.pyplot(fig2)
            elif graph_type == "Donut Chart":
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                wedges, texts, autotexts = ax2.pie(value_counts.values, labels=plot_labels, autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4), colors=category_colors)
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
            summary_df["Description"] = summary_df["Option"].map(custom_labels)  # Use custom label
            summary_df["Percentage (%)"] = (summary_df["Responses"] / summary_df["Responses"].sum() * 100).round(2)
            summary_df = summary_df[["Option", "Description", "Responses", "Percentage (%)"]]
            # Add total row
            total_row = pd.DataFrame({
                "Option": ["Total"],
                "Description": [""],
                "Responses": [summary_df["Responses"].sum()],
                "Percentage (%)": [100.0]
            })
            summary_df = pd.concat([summary_df, total_row], ignore_index=True)
            st.markdown("**Response Table:**")
            st.dataframe(sanitize_dataframe_for_streamlit(summary_df), use_container_width=True)
            summary_df["Option"] = summary_df["Option"].astype(str)

            # --- Clinical interpretation summary ---
            st.markdown("**Clinical Interpretation:**")
            total_respondents = int(summary_df["Responses"].sum() - summary_df.iloc[-1]["Responses"]) if len(summary_df) > 1 else 0
            st.write(f"• Total Respondents: {total_respondents}")
            for group in groupings:
                if group["cats"]:
                    group_count = summary_df[summary_df["Option"].isin(group["cats"])]["Responses"].sum()
                    group_pct = (group_count / total_respondents * 100) if total_respondents > 0 else 0
                    group_cats_disp = ', '.join([custom_labels[cat] for cat in group["cats"]])
                    interp_text = group["interp"] if group["interp"] else ""
                    st.write(f"• {group['label']} ({group_cats_disp}): {group_pct:.1f}% - {interp_text}")
        if is_numeric and not is_categorical and graph_type == "Summary Stats":
            st.write(col_data_clean.describe().to_frame())
        elif not (is_numeric and not is_categorical) and graph_type == "Value Counts Table":
            st.write(value_counts.to_frame(name="Count"))
        # Text area for user notes/interpretation
        user_notes = st.text_area("Add your interpretation or notes (optional)", key="notes")
        if user_notes:
            st.info(user_notes)

        # --- Chi-Square Test Tab ---
        if not (is_numeric and not is_categorical):
            import scipy.stats as stats
            st.markdown("---")
            chi_tab = st.expander("🔬 Chi-Square Goodness-of-Fit Test", expanded=False)
            with chi_tab:
                st.subheader(f"Chi-Square Test for '{col_selected}' Distribution")
                st.write("Select categories to include in the test (at least 2):")
                chi_categories = st.multiselect(
                    "Categories for Chi-Square Test",
                    options=list(value_counts.index),
                    default=list(value_counts.index),
                    key="chi_categories"
                )
                if len(chi_categories) >= 2:
                    observed_counts = [value_counts[cat] for cat in chi_categories]
                    total = sum(observed_counts)
                    expected_counts = [total / len(chi_categories)] * len(chi_categories)
                    chi_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
                    # Show observed vs expected table
                    chi_df = pd.DataFrame({
                        "Category": [custom_labels[cat] for cat in chi_categories],
                        "Observed": observed_counts,
                        "Expected": [round(x, 2) for x in expected_counts]
                    })
                    st.markdown("**Observed vs. Expected Counts:**")
                    st.dataframe(sanitize_dataframe_for_streamlit(chi_df), use_container_width=True)
                    # Show results
                    st.markdown("**Test Result:**")
                    st.write(f"Chi-Square Statistic: **{chi_stat:.4f}**")
                    # Format p-value up to 30 decimals, trim trailing zeros
                    p_str = f"{p_value:.30f}".rstrip('0').rstrip('.') if '.' in f"{p_value:.30f}" else f"{p_value:.30f}"
                    st.write(f"P-Value: **{p_str}**")
                    alpha = 0.05
                    if p_value < alpha:
                        st.success("👉 p < 0.05 → The distribution is significantly different from uniform.")
                    else:
                        st.info("👉 p ≥ 0.05 → The distribution is NOT significantly different from uniform.")
                else:
                    st.warning("Please select at least 2 categories for the test.")

# --- Statistical Tests Tab ---
with tabs[3]:
    st.header("🧪 Statistical Tests")
    st.markdown("Run biostatistics tests on your data. Select a test and relevant columns.")
    test_options = [
        "Chi-Square Goodness-of-Fit Test",
        "Chi-Square Test of Independence",
        "Fisher's Exact Test (2x2)",
        "T-Test (Independent)",
        "T-Test (Paired)",
        "ANOVA",
        "Mann-Whitney U Test",
        "Wilcoxon Signed-Rank Test",
        "Pearson Correlation",
        "Spearman Correlation",
        "Shapiro-Wilk Test"
    ]
    selected_test = st.selectbox("Select Statistical Test", test_options, key="stat_test_select")
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() < 20]
    import scipy.stats as stats
    result = None
    interpretation = None
    if selected_test == "Chi-Square Goodness-of-Fit Test":
        st.markdown("**Chi-Square Goodness-of-Fit Test** (for one categorical column)")
        col_a = st.selectbox("Categorical column", cat_cols, key="chisq_gof_col")
        if col_a:
            value_counts = df[col_a].value_counts()
            st.write("Select categories to include in the test (at least 2):")
            chi_categories = st.multiselect(
                "Categories for Chi-Square Test",
                options=list(value_counts.index),
                default=list(value_counts.index),
                key="chisq_gof_categories"
            )
            if len(chi_categories) >= 2:
                observed_counts = [value_counts[cat] for cat in chi_categories]
                total = sum(observed_counts)
                expected_counts = [total / len(chi_categories)] * len(chi_categories)
                chi_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
                # Show observed vs expected table
                chi_df = pd.DataFrame({
                    "Category": chi_categories,
                    "Observed": observed_counts,
                    "Expected": [round(x, 2) for x in expected_counts]
                })
                st.markdown("**Observed vs. Expected Counts:**")
                st.dataframe(sanitize_dataframe_for_streamlit(chi_df), use_container_width=True)
                # Show results
                st.markdown("**Test Result:**")
                st.write(f"Chi-Square Statistic: **{chi_stat:.4f}**")
                p_str = f"{p_value:.30f}".rstrip('0').rstrip('.') if '.' in f"{p_value:.30f}" else f"{p_value:.30f}"
                st.write(f"P-Value: **{p_str}**")
                alpha = 0.05
                if p_value < alpha:
                    st.success("👉 p < 0.05 → The distribution is significantly different from uniform.")
                else:
                    st.info("👉 p ≥ 0.05 → The distribution is NOT significantly different from uniform.")
            else:
                st.warning("Please select at least 2 categories for the test.")
    elif selected_test == "Chi-Square Test of Independence":
        st.markdown("**Chi-Square Test of Independence** (for two categorical columns)")
        col_a = st.selectbox("First categorical column", cat_cols, key="chi2_col_a")
        col_b = st.selectbox("Second categorical column", [c for c in cat_cols if c != col_a], key="chi2_col_b")
        if col_a and col_b:
            contingency = pd.crosstab(df[col_a], df[col_b])
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            st.write("Contingency Table:")
            st.dataframe(sanitize_dataframe_for_streamlit(contingency))
            st.write(f"Chi-Square Statistic: **{chi2:.4f}**")
            p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
            st.write(f"P-Value: **{p_str}**")
            if p < 0.05:
                st.success("👉 p < 0.05 → The variables are NOT independent.")
            else:
                st.info("👉 p ≥ 0.05 → The variables are independent.")
    elif selected_test == "Fisher's Exact Test (2x2)":
        st.markdown("**Fisher's Exact Test** (for two categorical columns, 2x2 table)")
        col_a = st.selectbox("First categorical column", cat_cols, key="fisher_col_a")
        col_b = st.selectbox("Second categorical column", [c for c in cat_cols if c != col_a], key="fisher_col_b")
        if col_a and col_b:
            contingency = pd.crosstab(df[col_a], df[col_b])
            if contingency.shape == (2, 2):
                oddsratio, p = stats.fisher_exact(contingency)
                st.write("Contingency Table:")
                st.dataframe(sanitize_dataframe_for_streamlit(contingency))
                st.write(f"Odds Ratio: **{oddsratio:.4f}**")
                p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
                st.write(f"P-Value: **{p_str}**")
                if p < 0.05:
                    st.success("👉 p < 0.05 → The variables are NOT independent.")
                else:
                    st.info("👉 p ≥ 0.05 → The variables are independent.")
            else:
                st.warning("Both columns must have exactly 2 unique values for Fisher's Exact Test.")
    elif selected_test == "T-Test (Independent)":
        st.markdown("**T-Test (Independent)** (compare means of two groups)")
        group_col = st.selectbox("Grouping (categorical) column", cat_cols, key="ttest_ind_group")
        value_col = st.selectbox("Numeric column", numeric_cols, key="ttest_ind_value")
        if group_col and value_col:
            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                data1 = df[df[group_col] == groups[0]][value_col].dropna()
                data2 = df[df[group_col] == groups[1]][value_col].dropna()
                tstat, p = stats.ttest_ind(data1, data2)
                st.write(f"Groups: {groups[0]} (n={len(data1)}), {groups[1]} (n={len(data2)})")
                st.write(f"T-Statistic: **{tstat:.4f}**")
                p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
                st.write(f"P-Value: **{p_str}**")
                if p < 0.05:
                    st.success("👉 p < 0.05 → The means are significantly different.")
                else:
                    st.info("👉 p ≥ 0.05 → No significant difference in means.")
            else:
                st.warning("Grouping column must have exactly 2 unique values.")
    elif selected_test == "T-Test (Paired)":
        st.markdown("**T-Test (Paired)** (compare means of two paired numeric columns)")
        col1 = st.selectbox("First numeric column", numeric_cols, key="ttest_rel_1")
        col2 = st.selectbox("Second numeric column", [c for c in numeric_cols if c != col1], key="ttest_rel_2")
        if col1 and col2:
            data1 = df[col1].dropna()
            data2 = df[col2].dropna()
            min_len = min(len(data1), len(data2))
            tstat, p = stats.ttest_rel(data1[:min_len], data2[:min_len])
            st.write(f"T-Statistic: **{tstat:.4f}**")
            p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
            st.write(f"P-Value: **{p_str}**")
            if p < 0.05:
                st.success("👉 p < 0.05 → The means are significantly different.")
            else:
                st.info("👉 p ≥ 0.05 → No significant difference in means.")
    elif selected_test == "ANOVA":
        st.markdown("**ANOVA** (compare means across more than two groups)")
        group_col = st.selectbox("Grouping (categorical) column", cat_cols, key="anova_group")
        value_col = st.selectbox("Numeric column", numeric_cols, key="anova_value")
        if group_col and value_col:
            groups = [df[df[group_col] == g][value_col].dropna() for g in df[group_col].dropna().unique()]
            if len(groups) > 2:
                fstat, p = stats.f_oneway(*groups)
                st.write(f"F-Statistic: **{fstat:.4f}**")
                p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
                st.write(f"P-Value: **{p_str}**")
                if p < 0.05:
                    st.success("👉 p < 0.05 → At least one group mean is significantly different.")
                else:
                    st.info("👉 p ≥ 0.05 → No significant difference in group means.")
            else:
                st.warning("Grouping column must have more than 2 unique values.")
    elif selected_test == "Mann-Whitney U Test":
        st.markdown("**Mann-Whitney U Test** (non-parametric, two independent groups)")
        group_col = st.selectbox("Grouping (categorical) column", cat_cols, key="mw_group")
        value_col = st.selectbox("Numeric column", numeric_cols, key="mw_value")
        if group_col and value_col:
            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                data1 = df[df[group_col] == groups[0]][value_col].dropna()
                data2 = df[df[group_col] == groups[1]][value_col].dropna()
                ustat, p = stats.mannwhitneyu(data1, data2)
                st.write(f"Groups: {groups[0]} (n={len(data1)}), {groups[1]} (n={len(data2)})")
                st.write(f"U-Statistic: **{ustat:.4f}**")
                p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
                st.write(f"P-Value: **{p_str}**")
                if p < 0.05:
                    st.success("👉 p < 0.05 → The distributions are significantly different.")
                else:
                    st.info("👉 p ≥ 0.05 → No significant difference in distributions.")
            else:
                st.warning("Grouping column must have exactly 2 unique values.")
    elif selected_test == "Wilcoxon Signed-Rank Test":
        st.markdown("**Wilcoxon Signed-Rank Test** (non-parametric, paired samples)")
        col1 = st.selectbox("First numeric column", numeric_cols, key="wilcoxon_1")
        col2 = st.selectbox("Second numeric column", [c for c in numeric_cols if c != col1], key="wilcoxon_2")
        if col1 and col2:
            data1 = df[col1].dropna()
            data2 = df[col2].dropna()
            min_len = min(len(data1), len(data2))
            try:
                wstat, p = stats.wilcoxon(data1[:min_len], data2[:min_len])
                st.write(f"W-Statistic: **{wstat:.4f}**")
                p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
                st.write(f"P-Value: **{p_str}**")
                if p < 0.05:
                    st.success("👉 p < 0.05 → The distributions are significantly different.")
                else:
                    st.info("👉 p ≥ 0.05 → No significant difference in distributions.")
            except Exception as e:
                st.warning(f"Wilcoxon test error: {e}")
    elif selected_test == "Pearson Correlation":
        st.markdown("**Pearson Correlation** (linear relationship between two numeric columns)")
        col1 = st.selectbox("First numeric column", numeric_cols, key="pearson_1")
        col2 = st.selectbox("Second numeric column", [c for c in numeric_cols if c != col1], key="pearson_2")
        if col1 and col2:
            data1 = df[col1].dropna()
            data2 = df[col2].dropna()
            min_len = min(len(data1), len(data2))
            r, p = stats.pearsonr(data1[:min_len], data2[:min_len])
            st.write(f"Pearson r: **{r:.4f}**")
            p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
            st.write(f"P-Value: **{p_str}**")
            if p < 0.05:
                st.success("👉 p < 0.05 → Significant linear correlation.")
            else:
                st.info("👉 p ≥ 0.05 → No significant linear correlation.")
    elif selected_test == "Spearman Correlation":
        st.markdown("**Spearman Correlation** (monotonic relationship between two numeric columns)")
        col1 = st.selectbox("First numeric column", numeric_cols, key="spearman_1")
        col2 = st.selectbox("Second numeric column", [c for c in numeric_cols if c != col1], key="spearman_2")
        if col1 and col2:
            data1 = df[col1].dropna()
            data2 = df[col2].dropna()
            min_len = min(len(data1), len(data2))
            r, p = stats.spearmanr(data1[:min_len], data2[:min_len])
            st.write(f"Spearman r: **{r:.4f}**")
            p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
            st.write(f"P-Value: **{p_str}**")
            if p < 0.05:
                st.success("👉 p < 0.05 → Significant monotonic correlation.")
            else:
                st.info("👉 p ≥ 0.05 → No significant monotonic correlation.")
    elif selected_test == "Shapiro-Wilk Test":
        st.markdown("**Shapiro-Wilk Test** (test for normality of a numeric column)")
        col1 = st.selectbox("Numeric column", numeric_cols, key="shapiro_1")
        if col1:
            data = df[col1].dropna()
            stat, p = stats.shapiro(data)
            st.write(f"W-Statistic: **{stat:.4f}**")
            p_str = f"{p:.30f}".rstrip('0').rstrip('.') if '.' in f"{p:.30f}" else f"{p:.30f}"
            st.write(f"P-Value: **{p_str}**")
            if p < 0.05:
                st.success("👉 p < 0.05 → The data is NOT normally distributed.")
            else:
                st.info("👉 p ≥ 0.05 → The data is likely normal.")

# --- Interpretation/Report Tab ---
with tabs[4]:
    st.header("📝 Interpretation & Report")
    st.markdown("Summarize your findings, add clinical interpretation, and export results.")
    # (Insert your clinical interpretation summary and notes section here)
    # ...

# --- Add tooltips, info boxes, and polish throughout the app as needed --- 
