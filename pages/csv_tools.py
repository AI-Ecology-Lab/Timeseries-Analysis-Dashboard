"""
CSV Tools Page

This page provides various tools for manipulating and analyzing CSV files. The features include:

1. **Data Subsampling**: Create a subsample of the data based on a specified time range.
2. **Drop Classes/Columns**: Drop selected columns from the dataset.
3. **Change Time Bins**: Resample the data to different time bins (hourly, daily, weekly).
4. **Filter Rows by Column Value**: Filter rows based on specific column values.
5. **Rename Columns**: Rename selected columns.
6. **Fill Missing Values**: Fill missing values using different methods (forward fill, backward fill, fill with zero).
7. **Preview Data Statistics**: Show basic statistics of the dataset.
8. **Output New CSV**: Download the processed data as a new CSV file.
9. **Sort Data**: Sort data based on selected columns.
10. **Merge CSVs**: Merge multiple CSV files into one.
11. **Visualize Data**: Create basic visualizations of the data.

The page also includes previews of the data before and after each operation.
"""

import streamlit as st
import pandas as pd
import os
import glob
from scripts.utils import load_local_files, load_uploaded_files
from scripts.utils_pagebuttons import inject_custom_css, create_page_title
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="CSV Tools", layout="wide")

# --- Inject shared CSS ---
inject_custom_css()

# --- Create page title ---
create_page_title("CSV Tools")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Data Selection")
    uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

# --- Load Data ---
dfs = []
if uploaded_files:
    dfs.extend(load_uploaded_files(uploaded_files))

if not dfs:
    st.warning("Please upload CSV files to analyze.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

# Ensure required columns are present
required_columns = ['File', 'Timestamp']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

# Preview original data
st.subheader("Original Data Preview")
st.dataframe(data.head(), use_container_width=True)

# --- Tabbed Interface ---
tabs = st.tabs(["Data Subsampling", "Drop Columns", "Change Time Bins", "Filter Rows", "Rename Columns", "Fill Missing Values", "Sort Data", "Merge CSVs", "Visualize Data", "Data Statistics", "Download CSV"])

with tabs[0]:
    st.subheader("Data Subsampling")
    min_date = pd.to_datetime(data['Timestamp']).min().date()
    max_date = pd.to_datetime(data['Timestamp']).max().date()
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="%Y-%m-%d"
    )
    if st.button("Create Subsample"):
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        data = data[(data['Timestamp'] >= start_date) & (data['Timestamp'] <= end_date)]
        st.write(data)
        st.subheader("Subsampled Data Preview")
        st.dataframe(data.head(), use_container_width=True)

with tabs[1]:
    st.subheader("Drop Classes/Columns")
    columns_to_drop = st.multiselect("Select Columns to Drop", options=data.columns)
    if st.button("Drop Columns"):
        data = data.drop(columns=columns_to_drop)
        st.write(data)
        st.subheader("Data After Dropping Columns Preview")
        st.dataframe(data.head(), use_container_width=True)

with tabs[2]:
    st.subheader("Change Time Bins")
    time_bin = st.selectbox("Select Time Bin", ["Hourly", "Daily", "Weekly"])
    if st.button("Resample Data"):
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        numeric_data = data.select_dtypes(include=['number'])
        numeric_data['Timestamp'] = data['Timestamp']
        if time_bin == "Hourly":
            data = numeric_data.resample('H', on='Timestamp').mean().reset_index()
        elif time_bin == "Daily":
            data = numeric_data.resample('D', on='Timestamp').mean().reset_index()
        elif time_bin == "Weekly":
            data = numeric_data.resample('W', on='Timestamp').mean().reset_index()
        st.write(data)
        st.subheader("Resampled Data Preview")
        st.dataframe(data.head(), use_container_width=True)

with tabs[3]:
    st.subheader("Filter Rows by Column Value")
    filter_column = st.selectbox("Select Column to Filter", options=data.columns)
    unique_values = data[filter_column].unique()
    filter_value = st.selectbox("Select Value to Filter", options=unique_values)
    if st.button("Filter Rows"):
        data = data[data[filter_column] == filter_value]
        st.write(data)
        st.subheader("Filtered Data Preview")
        st.dataframe(data.head(), use_container_width=True)

with tabs[4]:
    st.subheader("Rename Columns")
    columns_to_rename = st.multiselect("Select Columns to Rename", options=data.columns)
    new_column_names = {col: st.text_input(f"New name for {col}") for col in columns_to_rename}
    if st.button("Rename Columns"):
        data = data.rename(columns=new_column_names)
        st.write(data)
        st.subheader("Data After Renaming Columns Preview")
        st.dataframe(data.head(), use_container_width=True)

with tabs[5]:
    st.subheader("Fill Missing Values")
    fill_method = st.selectbox("Select Fill Method", ["Forward Fill", "Backward Fill", "Fill with Zero"])
    if st.button("Fill Missing Values"):
        if fill_method == "Forward Fill":
            data = data.fillna(method='ffill')
        elif fill_method == "Backward Fill":
            data = data.fillna(method='bfill')
        elif fill_method == "Fill with Zero":
            data = data.fillna(0)
        st.write(data)
        st.subheader("Data After Filling Missing Values Preview")
        st.dataframe(data.head(), use_container_width=True)

with tabs[6]:
    st.subheader("Sort Data")
    sort_column = st.selectbox("Select Column to Sort By", options=data.columns)
    sort_order = st.selectbox("Select Sort Order", ["Ascending", "Descending"])
    if st.button("Sort Data"):
        ascending = True if sort_order == "Ascending" else False
        data = data.sort_values(by=sort_column, ascending=ascending)
        st.write(data)
        st.subheader("Sorted Data Preview")
        st.dataframe(data.head(), use_container_width=True)

with tabs[7]:
    st.subheader("Merge CSVs")
    merge_files = st.file_uploader("Upload CSV Files to Merge", type=["csv"], accept_multiple_files=True)
    if merge_files:
        merge_dfs = load_uploaded_files(merge_files)
        data = pd.concat([data] + merge_dfs, ignore_index=True)
        st.write(data)
        st.subheader("Merged Data Preview")
        st.dataframe(data.head(), use_container_width=True)

with tabs[8]:
    st.subheader("Visualize Data Histograms")
    visualize_column = st.selectbox("Select Column to Visualize", options=data.columns)
    if st.button("Visualize Data"):
        fig = px.histogram(data, x=visualize_column)
        st.plotly_chart(fig)

with tabs[9]:
    st.subheader("Data Statistics")
    st.write(data.describe())

with tabs[10]:
    st.subheader("Output New CSV")
    if st.button("Download CSV"):
        csv = data.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name="processed_data.csv", mime="text/csv")