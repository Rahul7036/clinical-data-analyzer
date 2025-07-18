# Clinical Data Analyzer

A **generic, user-friendly web application** for analyzing clinical, survey, or any structured data from Excel files. This tool automates data cleaning, visualization, statistical analysis, and report generation, making it easy for users to generate publication-ready outputs without coding.

---

## 🚀 Features

- **Excel/CSV Upload:** Upload your own data files.
- **Data Preview:** Instantly see columns and sample rows.
- **Dynamic Analysis:**
  - Select columns and graph types (bar, pie, scatter, etc.).
  - Multi-column analyses (e.g., correlation, group comparison).
  - Statistical tests (chi-square, Spearman, Mann-Whitney, etc.).
- **Automated Graphs & Tables:**
  - Generate professional plots and summary tables.
- **Interpretation:**
  - Automated, human-readable explanations for each analysis.
- **Report Builder:**
  - Add multiple analyses to a session.
  - Download a comprehensive report (PDF/HTML) with all results.
- **User-Friendly Interface:**
  - No coding required. Intuitive, step-by-step workflow.

---

## 🖥️ User Workflow

1. **Upload Data:**
   - Drag and drop your Excel/CSV file.
2. **Preview Data:**
   - View columns and sample data.
3. **Select Analysis:**
   - Choose columns and desired graph/statistical test.
   - Optionally, ask questions in natural language (future feature).
4. **View Results:**
   - See generated graphs, tables, and interpretations.
   - Add to your report.
5. **Download Report:**
   - Export all results as a PDF/HTML report.

---

## 🛠️ Tech Stack

- **Frontend & Backend:** [Streamlit](https://streamlit.io/) (Python)
- **Data Processing:** pandas
- **Visualization:** matplotlib, seaborn, plotly
- **Statistics:** scipy, statsmodels
- **Report Generation:** jinja2 (HTML), weasyprint/reportlab (PDF)

---

## ⚡ Quickstart

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd clinical-data-analyzer
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
4. **Open in your browser:**
   - Go to [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

- `app.py` — Main Streamlit app
- `requirements.txt` — Python dependencies
- `README.md` — This file
- `src/` — (Optional) Custom modules for analysis, plotting, etc.

---

## 📝 Example Use Cases

- Clinical survey analysis (e.g., mMRC, Borg, CAT, pain scales)
- General survey or questionnaire data
- Any structured data requiring quick, flexible analysis

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for new features, bug fixes, or suggestions.

---

## 📄 License

MIT License (see `LICENSE` file)
