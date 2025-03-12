import streamlit as st 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly
import numpy as np
from scipy import stats
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils import load_local_files, load_uploaded_files, extract_data_columns
from scripts.utils_pagebuttons import (
    inject_custom_css, create_page_title, create_analysis_card, 
    create_add_analysis_button, show_matplotlib_plot
)
import os
import glob
import time
import platform

# --- Page Configuration ---
st.set_page_config(page_title="Environmental Correlation Analysis", layout="wide")

# --- Inject shared CSS ---
inject_custom_css()

# --- Create page title with version info ---
version_info = f"Running on Python {'.'.join(platform.python_version_tuple())} | Plotly {plotly.__version__} | Pandas {pd.__version__} | Numpy {np.__version__} | Scipy {scipy.__version__} | Scikit-learn {sklearn.__version__}"
create_page_title("Environmental Correlation Analysis", version_info)

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
            
    # --- Analysis Options ---
    analysis_info = {
        "Correlation Matrix": {
            "description": "Understand relationships between environmental variables and species.",
            "time": "Fast (< 30s)",
            "function": "correlation_matrix",
            "can_duplicate": False,
            "library": f"Plotly {plotly.__version__} (px.imshow) | SciPy {scipy.__version__} (stats.pearsonr)",
            "interpretation": """
                The correlation matrix shows relationships between variables from -1 (perfect negative correlation) to +1 (perfect positive correlation). 
                Red colors indicate negative correlations (as one variable increases, the other decreases), while blue colors indicate positive correlations. 
                The intensity of the color shows the strength of the relationship. P-values < 0.05 suggest statistically significant correlations.
                Focus on strong correlations (|r| > 0.5) and their significance levels to identify meaningful relationships.
            """,
            "parameters": {
                "significance_level": ("Significance Level", 0.01, 0.10, 0.05),
                "min_correlation": ("Minimum Correlation", 0.0, 1.0, 0.3),
                "color_scale": ("Color Scale", ["RdBu", "Viridis", "Plasma", "Magma"])
            }
        },
        "Environmental Response": {
            "description": "Visualize how species respond to environmental changes.",
            "time": "Medium (30s - 1min)",
            "function": "environmental_response",
            "can_duplicate": True,
            "library": f"Plotly {plotly.__version__} (px.scatter, go.Figure)",
            "interpretation": """
                The scatter plot shows the direct relationship between an environmental variable and species abundance. 
                The trend line indicates the overall pattern - an upward slope suggests the species prefers higher values of the environmental variable. 
                The density plot shows where most observations cluster, helping identify optimal environmental conditions for the species.
                Look for clear patterns and any potential thresholds where species abundance changes dramatically.
            """,
            "parameters": {
                "env_var": ("Environmental Variable", env_vars),
                "species": ("Species", class_names),
                "trendline_type": ("Trendline Type", ["lowess", "ols", "none"]),
                "density_type": ("Density Plot Type", ["2d", "contour", "none"])
            }
        },
        "PCA Analysis": {
            "description": "Reduce data dimensionality and identify key variables.",
            "time": "Fast (< 30s)",
            "function": "pca_analysis",
            "can_duplicate": False,
            "library": f"Scikit-learn {sklearn.__version__} (PCA) | Plotly {plotly.__version__}",
            "interpretation": """
                The scree plot shows how much variance each principal component explains - look for the "elbow" to determine important components. 
                In the biplot, longer arrows indicate more influential variables, and their directions show relationships between variables. 
                Closely aligned arrows suggest correlated variables, while opposing arrows suggest inverse relationships.
                The clustering of points shows similar environmental conditions.
            """,
            "parameters": {
                "n_components": ("Number of Components", 2, min(len(env_vars), 10), min(len(env_vars), 4)),
                "standardize": ("Standardize Data", [True, False]),
                "plot_type_pca": ("PCA Plot Type", ["scree", "biplot", "both"])
            }
        },
        "Time-lagged Correlations": {
            "description": "Identify time-delayed effects on species.",
            "time": "Slow (1-2min)",
            "function": "time_lagged",
            "can_duplicate": True,
            "library": f"Pandas {pd.__version__} (shift) | Plotly {plotly.__version__} (px.line)",
            "interpretation": """
                The plot shows how correlations change with different time lags. Peaks in the correlation indicate optimal time delays. 
                A peak at lag=0 suggests immediate responses, while peaks at other lags suggest delayed responses. 
                The red lines show the significance threshold - correlations beyond these lines are considered meaningful.
                Consider both positive and negative correlations, as species may respond positively or negatively to environmental changes.
            """,
            "parameters": {
                "env_var": ("Environmental Variable", env_vars),
                "species": ("Species", class_names),
                "max_lag": ("Maximum Lag (hours)", 1, 72, 24),
                "min_correlation": ("Minimum Correlation", 0.0, 1.0, 0.3)
            }
        },
        "Threshold Analysis": {
            "description": "Find environmental thresholds affecting species.",
            "time": "Medium (30s - 1min)",
            "function": "threshold_analysis",
            "can_duplicate": True,
            "library": f"NumPy {np.__version__} (percentile) | Plotly {plotly.__version__} (px.bar)",
            "interpretation": """
                The bar plot shows average species abundance across different environmental ranges. 
                Error bars indicate variability within each range - longer bars suggest less consistent patterns. 
                Look for significant differences between adjacent bars to identify potential threshold values where species behavior changes.
                Consider both the means and the spread of the data when interpreting thresholds.
            """,
            "parameters": {
                "env_var": ("Environmental Variable", env_vars),
                "species": ("Species", class_names),
                "n_thresholds": ("Number of Thresholds", 2, 5, 3),
                "threshold_type": ("Threshold Type", ["percentile", "fixed"])
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
                if analysis_type == "Correlation Matrix":
                    corr_df = data[env_vars + class_names].corr()
                    env_species_corr = corr_df.loc[env_vars, class_names]
                    p_values = pd.DataFrame(index=env_vars, columns=class_names)
                    
                    for env in env_vars:
                        for species in class_names:
                            _, p_val = stats.pearsonr(data[env], data[species])
                            p_values.loc[env, species] = p_val
                    
                    results = {"corr_matrix": env_species_corr, "p_values": p_values}
                    
                    # Create Plotly figure for correlation matrix
                    fig = px.imshow(
                        env_species_corr,
                        title=f"Environment-Species Correlation Matrix",
                        labels=dict(x="Species", y="Environmental Variables", color="Correlation"),
                        aspect="auto",
                        color_continuous_scale=config["parameters"]["color_scale"]
                    )
                    results["plotly_fig"] = fig
                    
                    st.session_state.analysis_results[config_key] = results
                
                elif analysis_type == "Environmental Response":
                    results = {}
                    
                    # Create scatter plot with Plotly
                    fig = px.scatter(
                        data, 
                        x=config["parameters"]["env_var"], 
                        y=config["parameters"]["species"],
                        trendline=config["parameters"]["trendline_type"] if config["parameters"]["trendline_type"] != "none" else None,
                        title=f"{config['parameters']['species']} Response to {config['parameters']['env_var']}"
                    )
                    results["plotly_scatter"] = fig
                    
                    if config["parameters"]["density_type"] != "none":
                        fig2 = go.Figure()
                        if config["parameters"]["density_type"] == "2d":
                            fig2.add_histogram2d(
                                x=data[config["parameters"]["env_var"]],
                                y=data[config["parameters"]["species"]],
                                colorscale="Viridis",
                                nbinsx=30,
                                nbinsy=30
                            )
                        else:  # contour
                            fig2.add_histogram2dcontour(
                                x=data[config["parameters"]["env_var"]],
                                y=data[config["parameters"]["species"]],
                                colorscale="Viridis",
                                nbinsx=30,
                                nbinsy=30
                            )
                        fig2.update_layout(title=f"Density Distribution: {config['parameters']['species']} vs {config['parameters']['env_var']}")
                        results["plotly_density"] = fig2
                    
                    st.session_state.analysis_results[config_key] = results
                
                elif analysis_type == "PCA Analysis":
                    if config["parameters"]["standardize"]:
                        scaler = StandardScaler()
                        env_scaled = scaler.fit_transform(data[env_vars])
                    else:
                        env_scaled = data[env_vars].values
                    
                    pca = PCA(n_components=config["parameters"]["n_components"])
                    env_pca = pca.fit_transform(env_scaled)
                    explained_variance = pca.explained_variance_ratio_ * 100
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                    
                    results = {}
                    
                    if config["parameters"]["plot_type_pca"] in ["scree", "both"]:
                        fig_scree = px.line(
                            x=range(1, len(explained_variance) + 1),
                            y=explained_variance,
                            markers=True,
                            title="PCA Scree Plot",
                            labels={"x": "Principal Component", "y": "Explained Variance (%)"}
                        )
                        results["plotly_scree"] = fig_scree
                    
                    if config["parameters"]["plot_type_pca"] in ["biplot", "both"]:
                        fig_biplot = go.Figure()
                        fig_biplot.add_scatter(
                            x=env_pca[:, 0], y=env_pca[:, 1],
                            mode='markers', name='Samples',
                            marker=dict(size=8, opacity=0.6)
                        )
                        
                        for i, var in enumerate(env_vars):
                            fig_biplot.add_scatter(
                                x=[0, loadings[i, 0]], y=[0, loadings[i, 1]],
                                mode='lines+text', name=var,
                                text=[var], textposition='top right'
                            )
                        
                        fig_biplot.update_layout(
                            title="PCA Biplot",
                            xaxis_title=f"PC1 ({explained_variance[0]:.1f}%)",
                            yaxis_title=f"PC2 ({explained_variance[1]:.1f}%)"
                        )
                        results["plotly_biplot"] = fig_biplot
                    
                    st.session_state.analysis_results[config_key] = results
                
                elif analysis_type == "Time-lagged Correlations":
                    correlations = []
                    for lag in range(config["parameters"]["max_lag"] + 1):
                        corr = data[config["parameters"]["env_var"]].shift(lag).corr(data[config["parameters"]["species"]])
                        correlations.append({"lag": lag, "correlation": corr})
                    
                    lag_df = pd.DataFrame(correlations)
                    results = {"correlations": lag_df}
                    
                    fig = px.line(
                        lag_df,
                        x="lag",
                        y="correlation",
                        title=f"Time-lagged Correlation: {config['parameters']['species']} vs {config['parameters']['env_var']}",
                        labels={"lag": "Lag (hours)", "correlation": "Correlation Coefficient"}
                    )
                    fig.add_hline(y=config["parameters"]["min_correlation"], line_dash="dash", line_color="red")
                    fig.add_hline(y=-config["parameters"]["min_correlation"], line_dash="dash", line_color="red")
                    results["plotly_fig"] = fig
                    
                    st.session_state.analysis_results[config_key] = results
                
                elif analysis_type == "Threshold Analysis":
                    if config["parameters"]["threshold_type"] == "percentile":
                        percentiles = np.linspace(0, 100, config["parameters"]["n_thresholds"] + 1)[1:-1]
                        thresholds = np.percentile(data[config["parameters"]["env_var"]], percentiles)
                    else:  # fixed
                        env_range = data[config["parameters"]["env_var"]].max() - data[config["parameters"]["env_var"]].min()
                        thresholds = np.linspace(
                            data[config["parameters"]["env_var"]].min() + env_range / (config["parameters"]["n_thresholds"] + 1),
                            data[config["parameters"]["env_var"]].max() - env_range / (config["parameters"]["n_thresholds"] + 1),
                            config["parameters"]["n_thresholds"]
                        )
                    
                    # Create threshold categories
                    categories = pd.cut(
                        data[config["parameters"]["env_var"]],
                        bins=[-np.inf] + list(thresholds) + [np.inf],
                        labels=[f"< {thresholds[0]:.2f}"] + 
                               [f"{thresholds[i]:.2f} - {thresholds[i+1]:.2f}" for i in range(len(thresholds)-1)] +
                               [f"> {thresholds[-1]:.2f}"]
                    )
                    
                    # Calculate statistics for each threshold
                    threshold_stats = []
                    for cat in categories.unique():
                        mask = categories == cat
                        stats_dict = {
                            "threshold": cat,
                            "mean": data.loc[mask, config["parameters"]["species"]].mean(),
                            "std": data.loc[mask, config["parameters"]["species"]].std(),
                            "count": mask.sum()
                        }
                        threshold_stats.append(stats_dict)
                    
                    stats_df = pd.DataFrame(threshold_stats)
                    results = {"stats": stats_df}
                    
                    fig = px.bar(
                        stats_df,
                        x="threshold",
                        y="mean",
                        error_y="std",
                        title=f"{config['parameters']['species']} Abundance by {config['parameters']['env_var']} Thresholds",
                        labels={"threshold": f"{config['parameters']['env_var']} Level", "mean": f"Mean {config['parameters']['species']} Count"}
                    )
                    results["plotly_fig"] = fig
                    
                    st.session_state.analysis_results[config_key] = results
                
                config["run_status"] = "completed"

            if st.button("Remove Analysis", key=f"remove_{config_key}"):
                analyses_to_remove.append(config_key)

        # Display results if available
        if config_key in st.session_state.analysis_results:
            results = st.session_state.analysis_results[config_key]
            
            if analysis_type == "Correlation Matrix":
                if "plotly_fig" in results:
                    st.plotly_chart(results["plotly_fig"], use_container_width=True)
                
                # Display significant correlations
                significant_corr = (results["p_values"] < config["parameters"]["significance_level"]) & (
                    abs(results["corr_matrix"]) > config["parameters"]["min_correlation"]
                )
                if significant_corr.any().any():
                    st.write(f"Significant correlations (p < {config['parameters']['significance_level']:.2f}, |r| > {config['parameters']['min_correlation']:.2f}):")
                    for env in env_vars:
                        for species in class_names:
                            if significant_corr.loc[env, species]:
                                st.write(f"{env} vs {species}: r = {results['corr_matrix'].loc[env, species]:.3f}")
            
            elif analysis_type == "Environmental Response":
                if "plotly_scatter" in results:
                    st.plotly_chart(results["plotly_scatter"], use_container_width=True)
                
                if "plotly_density" in results:
                    st.plotly_chart(results["plotly_density"], use_container_width=True)
            
            elif analysis_type == "PCA Analysis":
                if "plotly_scree" in results:
                    st.plotly_chart(results["plotly_scree"], use_container_width=True)
                
                if "plotly_biplot" in results:
                    st.plotly_chart(results["plotly_biplot"], use_container_width=True)
            
            elif analysis_type == "Time-lagged Correlations":
                if "plotly_fig" in results:
                    st.plotly_chart(results["plotly_fig"], use_container_width=True)
            
            elif analysis_type == "Threshold Analysis":
                if "plotly_fig" in results:
                    st.plotly_chart(results["plotly_fig"], use_container_width=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Remove analyses marked for deletion
    for key in analyses_to_remove:
        del st.session_state.analysis_configs[key]
        if key in st.session_state.analysis_results:
            del st.session_state.analysis_results[key]
                
else:
    st.warning("Please select or upload CSV files with a 'Timestamp' column to analyze.")
    st.stop()