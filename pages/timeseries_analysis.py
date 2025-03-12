import streamlit as st 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly
import numpy as np
import statsmodels.api as sm
from scripts.utils import load_local_files, load_uploaded_files, extract_data_columns, melt_species_data
from scripts.utils_pagebuttons import (
    inject_custom_css, create_page_title, create_analysis_card, 
    create_add_analysis_button, show_matplotlib_plot
)
import os
import glob
import time
import platform

# --- Page Configuration ---
st.set_page_config(page_title="Time Series Analysis", layout="wide")

# --- Inject shared CSS ---
inject_custom_css()

# --- Create page title with version info ---
version_info = f"Running on Python {'.'.join(platform.python_version_tuple())} | Plotly {plotly.__version__} | Pandas {pd.__version__} | Numpy {np.__version__} | StatsModels {sm.__version__}"
create_page_title("Time Series Analysis", version_info)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Data Selection")
    camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]
    selected_camera = st.selectbox("Select Camera", camera_names)
    
    uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)
    
    base_dir = os.path.join("timeseries", selected_camera)
    csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    csv_files = [os.path.relpath(f, base_dir) for f in csv_files]
    selected_csvs = st.multiselect("Select CSV Files", csv_files)

# --- Load and Process Data ---
dfs = []
if selected_csvs:
    dfs.extend(load_local_files(base_dir, selected_csvs))
if uploaded_files:
    dfs.extend(load_uploaded_files(uploaded_files))

if not dfs:
    st.warning("Please select or upload CSV files to analyze.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

if 'Timestamp' in data.columns:
    # Convert Timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Extract columns (species, clusters, environmental variables)
    class_names, cluster_cols, env_vars = extract_data_columns(data)
    
    # --- Option to include/deselect Bubble as a species ---
    if "Bubble" in class_names:
        include_bubble = st.sidebar.checkbox("Include 'Bubble' as a species", value=False)
        if not include_bubble:
            class_names = [sp for sp in class_names if sp != "Bubble"]
    
    # Convert to long format for species data
    melted_data = melt_species_data(data, class_names)
    
    # --- Analysis Options ---
    analysis_info = {
        "Class Distribution": {
            "description": "Visualize the total counts for each species.",
            "time": "Fast (< 10s)",
            "function": "class_distribution",
            "can_duplicate": False,
            "library": f"Plotly {plotly.__version__} (px.bar)",
            "interpretation": """
                The bar chart shows the total count for each species across all data. 
                Higher bars indicate more abundant species in the dataset.
                Compare the relative heights to understand which species are dominant in the ecosystem.
            """,
            "parameters": {}
        },
        "Cluster Analysis": {
            "description": "Analyze cluster coverage with trend lines over time.",
            "time": "Long (90-120s)",
            "function": "cluster_analysis",
            "can_duplicate": False,
            "library": f"Plotly {plotly.__version__} (go.Bar, go.Scatter) | StatsModels {sm.__version__}",
            "interpretation": """
                The stacked bar chart shows cluster coverage over time, with trend lines indicating overall direction.
                Each cluster has a regression line with its equation and R² value indicating goodness of fit.
                Higher R² values (closer to 1.0) indicate stronger trends, while values closer to 0 suggest random variation.
                Upward trends suggest increasing coverage, while downward trends indicate decline.
            """,
            "parameters": {}
        },
        "Environmental Variable Analysis": {
            "description": "Track changes in environmental variables over time.",
            "time": "Fast (< 10s)",
            "function": "environmental_analysis",
            "can_duplicate": True,
            "library": f"Plotly {plotly.__version__} (px.line)",
            "interpretation": """
                The line chart tracks changes in the selected environmental variable over time.
                Look for patterns such as daily/seasonal cycles, sudden changes, or long-term trends.
                Consider how these patterns might correlate with changes in species abundance.
            """,
            "parameters": {
                "env_var": ("Environmental Variable", env_vars)
            }
        },
        "Stacked Visualizations": {
            "description": "Compare multiple species abundances over time with stacked charts.",
            "time": "Long (60-90s)",
            "function": "stacked_visualizations",
            "can_duplicate": True,
            "library": f"Plotly {plotly.__version__} (go.Bar, go.Scatter)",
            "interpretation": """
                The stacked chart shows how multiple species contribute to total abundance over time.
                Stacked bar charts emphasize discrete time points, while area charts highlight continuous trends.
                Look for patterns in relative abundance - whether certain species consistently dominate
                or if there are temporal shifts in community composition.
            """,
            "parameters": {
                "chart_type": ("Chart Type", ["Stacked Bar", "Stacked Area"]),
                "selected_classes": ("Species to Include", class_names)
            }
        },
        "Multi-class Timeline": {
            "description": "View individual time series for multiple species.",
            "time": "Medium (10-30s)",
            "function": "multi_class_timeline",
            "can_duplicate": True,
            "library": f"Plotly {plotly.__version__} (go.Scatter)",
            "interpretation": """
                Each subplot shows the abundance of a single species over time.
                This visualization makes it easier to see patterns specific to each species.
                Look for cyclical patterns, sudden changes, or correlations between different species.
                Similar patterns across species might suggest responses to the same environmental drivers.
            """,
            "parameters": {
                "selected_classes": ("Species to Display", class_names)
            }
        }
    }
    
    # Initialize session state
    if 'analysis_configs' not in st.session_state:
        st.session_state.analysis_configs = {}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Create analysis selection interface
    st.subheader("Configure Analyses")
    
    # Add new analysis button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""<style>
            div[data-testid="stSelectbox"] > div:first-child {
                margin-bottom: -1em;
            }
            div[data-testid="stSelectbox"] > div > div[data-baseweb="select"] {
                background: var(--background-color);
            }
            </style>""", unsafe_allow_html=True)
        new_analysis = st.selectbox("Select Analysis Type", [""] + list(analysis_info.keys()))
    with col2:
        if create_add_analysis_button():
            if new_analysis:
                analysis_count = sum(1 for k in st.session_state.analysis_configs.keys() if k.startswith(new_analysis))
                if not analysis_info[new_analysis]["can_duplicate"] and analysis_count > 0:
                    st.warning(f"{new_analysis} cannot be added multiple times.")
                else:
                    config_key = f"{new_analysis}_{analysis_count + 1}" if analysis_count > 0 else new_analysis
                    # Initialize parameters with default values from analysis_info
                    default_parameters = {}
                    for param_key, param_info in analysis_info[new_analysis]["parameters"].items():
                        _, *param_range = param_info
                        # Get default value based on parameter type
                        if isinstance(param_range[0], list):
                            default_parameters[param_key] = param_range[0][0]  # First option as default
                        elif isinstance(param_range[0], bool):
                            default_parameters[param_key] = param_range[0]  # Default boolean value
                        elif isinstance(param_range[0], (int, float)):
                            default_parameters[param_key] = param_range[2]  # Default value from slider
                    
                    st.session_state.analysis_configs[config_key] = {
                        "type": new_analysis,
                        "parameters": default_parameters,
                        "run_status": "not_run"
                    }
            else:
                st.warning("Please select an analysis type first.")
    
    # Display and configure added analyses
    analyses_to_remove = []
    
    for config_key, config in st.session_state.analysis_configs.items():
        analysis_type = config["type"]
        info = analysis_info[analysis_type]
        
        create_analysis_card(
            title=config_key,
            description=info['description'],
            execution_time=info['time'],
            library_info=info['library'],
            interpretation=info['interpretation']
        )
        
        # Parameter configuration
        with st.expander("Configure Parameters"):
            for param_key, param_info in info["parameters"].items():
                label, *param_range = param_info
                if isinstance(param_range[0], list):
                    if param_key == "selected_classes":
                        # For multi-select parameters like species lists
                        config["parameters"][param_key] = st.multiselect(
                            label,
                            param_range[0],
                            default=param_range[0][:min(5, len(param_range[0]))],  # Default to first 5
                            key=f"{config_key}_{param_key}"
                        )
                    else:
                        config["parameters"][param_key] = st.selectbox(
                            label,
                            param_range[0],
                            key=f"{config_key}_{param_key}"
                        )
                elif isinstance(param_range[0], bool):
                    config["parameters"][param_key] = st.checkbox(
                        label,
                        param_range[0],
                        key=f"{config_key}_{param_key}"
                    )
                elif isinstance(param_range[0], (int, float)):
                    config["parameters"][param_key] = st.slider(
                        label,
                        param_range[0],
                        param_range[1],
                        param_range[2],
                        key=f"{config_key}_{param_key}"
                    )
            
            if st.button("Run Analysis", key=f"run_{config_key}"):
                config["run_status"] = "running"
                
                # Run the analysis based on plot type selection
                if analysis_type == "Class Distribution":
                    class_counts = melted_data.groupby('class_name')['animal_count'].sum().reset_index()
                    fig = px.bar(class_counts, x='class_name', y='animal_count', 
                             title="Total Counts by Class")
                    fig.update_layout(xaxis_title="Species", yaxis_title="Count")
                    
                    st.session_state.analysis_results[config_key] = {"plotly_fig": fig}
                
                elif analysis_type == "Cluster Analysis":
                    if not any("Cluster" in col for col in data.columns):
                        st.warning("No cluster data available in the selected files.")
                    else:
                        # Get cluster columns
                        cluster_cols = [col for col in data.columns if "Cluster" in col]
                        
                        # Create DataFrame with clusters
                        cluster_data = data[['timestamp'] + cluster_cols].copy()
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        # Plot stacked bars for each cluster
                        for column in cluster_cols:
                            fig.add_trace(go.Bar(
                                x=cluster_data['timestamp'],
                                y=cluster_data[column],
                                name=column
                            ))
                            
                            # Add regression line
                            X = np.arange(len(cluster_data)).reshape(-1, 1)
                            y = cluster_data[column].values
                            X_sm = sm.add_constant(X)
                            model = sm.OLS(y, X_sm).fit()
                            predictions = model.predict(X_sm)
                            
                            # Calculate equation and R-squared
                            slope = model.params[1]
                            intercept = model.params[0]
                            equation = f'y = {slope:.2e}x + {intercept:.2f}'
                            r_squared = model.rsquared
                            
                            # Add regression line
                            fig.add_trace(go.Scatter(
                                x=cluster_data['timestamp'],
                                y=predictions,
                                mode='lines',
                                name=f'{column} Trend',
                                line=dict(dash='dash')
                            ))
                            
                            # Add annotation
                            fig.add_annotation(
                                x=cluster_data['timestamp'].iloc[-1],
                                y=predictions[-1],
                                text=f'{column}:<br>{equation}<br>R² = {r_squared:.3f}',
                                showarrow=False,
                                yshift=10
                            )
                        
                        fig.update_layout(
                            barmode='stack',
                            title='Cluster Coverage Analysis Over Time',
                            xaxis_title='Time',
                            yaxis_title='Coverage',
                            template='plotly_white',
                            hovermode='x unified',
                            legend_title='Clusters'
                        )
                        
                        st.session_state.analysis_results[config_key] = {"plotly_fig": fig}
                
                elif analysis_type == "Environmental Variable Analysis":
                    if env_vars:
                        env_var = config["parameters"]["env_var"]
                        fig = px.line(data, x='timestamp', y=env_var,
                                  title=f"{env_var} Over Time")
                        fig.update_layout(xaxis_title="Time", yaxis_title=env_var)
                        st.session_state.analysis_results[config_key] = {"plotly_fig": fig}
                    else:
                        st.warning("No environmental variables found in the data.")
                
                elif analysis_type == "Stacked Visualizations":
                    selected_classes = config["parameters"]["selected_classes"]
                    chart_type = config["parameters"]["chart_type"]
                    
                    if not selected_classes:
                        st.warning("Please select at least one class.")
                    else:
                        # Filter data for selected classes
                        filtered_data = melted_data[melted_data['class_name'].isin(selected_classes)]
                        
                        # Pivot data for stacking
                        pivot_data = filtered_data.pivot_table(
                            index='timestamp',
                            columns='class_name',
                            values='animal_count',
                            fill_value=0
                        ).reset_index()

                        if chart_type == "Stacked Bar":
                            fig = go.Figure()
                            for class_name in selected_classes:
                                fig.add_trace(go.Bar(
                                    name=class_name,
                                    x=pivot_data['timestamp'],
                                    y=pivot_data[class_name]
                                ))
                            fig.update_layout(
                                barmode='stack',
                                title="Stacked Species Distribution Over Time",
                                xaxis_title="Time",
                                yaxis_title="Count"
                            )

                        else:  # Stacked Area
                            fig = go.Figure()
                            for class_name in selected_classes:
                                fig.add_trace(go.Scatter(
                                    name=class_name,
                                    x=pivot_data['timestamp'],
                                    y=pivot_data[class_name],
                                    mode='lines',
                                    stackgroup='one'
                                ))
                            fig.update_layout(
                                title="Species Distribution Area Over Time",
                                xaxis_title="Time",
                                yaxis_title="Count"
                            )

                        st.session_state.analysis_results[config_key] = {"plotly_fig": fig}
                
                elif analysis_type == "Multi-class Timeline":
                    selected_classes = config["parameters"]["selected_classes"]

                    if not selected_classes:
                        st.warning("Please select at least one class.")
                    else:
                        # Create subplots, one for each class
                        fig = go.Figure()
                        
                        for i, class_name in enumerate(selected_classes):
                            class_data = melted_data[melted_data['class_name'] == class_name]
                            
                            fig.add_trace(go.Scatter(
                                x=class_data['timestamp'],
                                y=class_data['animal_count'],
                                name=class_name,
                                yaxis=f'y{i+1}' if i > 0 else 'y'
                            ))

                        # Update layout with multiple y-axes
                        layout_updates = {
                            'title': 'Multi-class Timeline Analysis',
                            'xaxis': {'title': 'Time'},
                            'height': 100 + (len(selected_classes) * 200),  # Adjust height based on number of classes
                            'showlegend': True,
                            'grid': {'rows': len(selected_classes), 'columns': 1, 'pattern': 'independent'},
                        }

                        # Add separate y-axes for each class
                        for i, class_name in enumerate(selected_classes):
                            if i == 0:
                                layout_updates['yaxis'] = {
                                    'title': class_name,
                                    'domain': [(len(selected_classes)-1-i)/len(selected_classes), 
                                              (len(selected_classes)-i)/len(selected_classes)]
                                }
                            else:
                                layout_updates[f'yaxis{i+1}'] = {
                                    'title': class_name,
                                    'domain': [(len(selected_classes)-1-i)/len(selected_classes), 
                                              (len(selected_classes)-i)/len(selected_classes)]
                                }

                        fig.update_layout(**layout_updates)
                        st.session_state.analysis_results[config_key] = {"plotly_fig": fig}
                
                config["run_status"] = "completed"

            if st.button("Remove Analysis", key=f"remove_{config_key}"):
                analyses_to_remove.append(config_key)

        # Display results if available
        if config_key in st.session_state.analysis_results:
            results = st.session_state.analysis_results[config_key]
            if "plotly_fig" in results:
                st.plotly_chart(results["plotly_fig"], use_container_width=True)
    
    # Remove analyses marked for removal
    for key in analyses_to_remove:
        if key in st.session_state.analysis_configs:
            del st.session_state.analysis_configs[key]
        if key in st.session_state.analysis_results:
            del st.session_state.analysis_results[key]
    
    # --- Download Options ---
    with st.expander("Download Data"):
        if st.button("Download Analysis Data"):
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"analysis_{selected_camera}.csv",
                mime="text/csv"
            )
else:
    st.warning("Please select or upload CSV files with 'Timestamp' to analyze.")
    st.stop()