import streamlit as st
import pandas as pd

def app():
    st.title("CSV Data Format Documentation")
    
    st.markdown("""
    ### Overview
    The dashboard accepts CSV files containing time series data for environmental and biological analysis.
    This document details the required structure, column formats, and provides an example of a properly formatted CSV.
    """)
    
    st.markdown("### CSV File Structure")
    st.markdown("""
    Your CSV files should follow this structure:
    
    1. **File:** The name of the source data file.
    2. **Timestamp:** The date and time of the observation in `MM/DD/YYYY HH:MM` format.
    3. **Species Data:** One or more columns with numeric values (e.g., counts or 0/1 presence indicators) for each observed species.
    4. **Cluster Data:** Columns named following the pattern `Cluster X` (e.g., `Cluster 0`, `Cluster 1`) that contain numeric data from image segmentation. *Optional:* If not available, include a dummy column to separate species from environmental variables.
    5. **Environmental Variables:** Columns with numeric measurements such as Temperature, Salinity, etc. These are optional but required for correlation analyses.
    """)
    
    st.subheader("Detailed Column Descriptions")
    col_data = [
        ["File", "String", "CAMDSB103_10.33.7.5_20220810T154659_UTC.txt", "Unique identifier for the source data file."],
        ["Timestamp", "Date & Time", "8/10/2022 15:46", "Observation date and time in MM/DD/YYYY HH:MM format."],
        ["Species Columns", "Numeric", "0 or 1 (or counts)", "Data for each species (e.g., Anoplopoma, Asteroidea). Use clear, descriptive column names."],
        ["Cluster Columns", "Numeric", "40.76, 7.19", "Data from k-means image segmentation. Should follow the naming convention `Cluster X`. Optional but recommended."],
        ["Environmental Variables", "Numeric", "15.3, 39.482", "Environmental measurements (e.g., Temperature, Salinity). Optional; required for environmental correlation tests."]
    ]
    
    df_cols = pd.DataFrame(col_data, columns=["Column Name", "Type", "Example", "Description"])
    st.table(df_cols)
    
    st.subheader("Species Data Columns")
    st.markdown("""
    **Species Data:**
    
    - **Purpose:** Record counts or binary presence (0/1) of various species.
    - **Flexibility:** The system auto-detects species from the column names. No fixed list is required.
    - **Tip:** Use descriptive names (e.g., `Anoplopoma`, `Asteroidea`) to ensure clarity.
    """)
    
    st.subheader("Cluster Data Columns")
    st.markdown("""
    **Cluster Data:**
    
    - **Naming Convention:** Use `Cluster X` where X is an integer (e.g., `Cluster 0`, `Cluster 1`).
    - **Data:** Numeric values representing proportions or intensities derived from image segmentation.
    - **Usage:** Commonly used for analyzing features such as bacterial mat coverage, sea ice ratios, sediment distribution, or habitat classifications.
    - **Optional:** If cluster data is unavailable, include a dummy column to mark the start of environmental variables.
    """)
    
    st.subheader("Environmental Variable Columns")
    st.markdown("""
    **Environmental Variables:**
    
    - **Measurements:** Include variables such as Temperature, Conductivity, Pressure, Salinity, etc.
    - **Format:** Numeric values for each measurement.
    - **Purpose:** Enable the analysis of correlations between environmental conditions and both species and cluster data.
    - **Examples:** Temperature, Salinity, Oxygen Phase (usec), Oxygen Temperature Voltage, PressurePSI.
    """)
    
    st.subheader("Example CSV File")
    st.markdown("Displaying the first 10 rows of the example CSV file:")
    
    try:
        example_df = pd.read_csv("timeseries/MJ01B_CAMDSB103/SHR_2022_2023_fauna_ctd_pressure.csv", nrows=10)
        st.dataframe(example_df)
    except FileNotFoundError:
        st.error("Example CSV file not found. Please ensure the file exists in the specified directory.")
    except Exception as e:
        st.error(f"An error occurred while reading the CSV file: {e}")
    
    st.subheader("Data Processing Workflow")
    st.markdown("""
    **How the Data is Processed:**
    
    1. **Loading:** CSV files are loaded from local storage or uploaded by the user.
    2. **Parsing:** The system extracts:
       - The file identifier (`File` column)
       - Observation timestamps (`Timestamp` column)
       - Species data for biological analysis
       - Cluster data from image segmentation (if provided)
       - Environmental variables for further analysis
    3. **Analysis:** Now its in your hands to make any of the graphs you need!
    """)
    
    st.info("Note: While some columns (Species, Cluster, Environmental) are optional, the `File` and `Timestamp` columns are required. Variations in column names are acceptable as long as the overall structure is maintained.")

if __name__ == "__main__":
    app()
