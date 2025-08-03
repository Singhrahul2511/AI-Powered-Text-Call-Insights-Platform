# utils/helpers.py

import pandas as pd
import streamlit as st
import base64
from io import BytesIO

@st.cache_data
def load_data(file):
    """
    Load data from an uploaded file (CSV, Excel, or JSON).
    """
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        elif file.name.endswith('.json'):
            return pd.read_json(file, lines=True)
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_table_download_link(df):
    """
    Generates a download link for a DataFrame as an Excel file.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='AnalysisResults')
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="analysis_results.xlsx">Download analysis results as Excel file</a>'
