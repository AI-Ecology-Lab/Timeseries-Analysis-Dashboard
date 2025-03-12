import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

def inject_custom_css():
    """Inject shared custom CSS styles for components"""
    st.markdown("""<style>
/* Theme variables */
:root[data-theme="dark"] {
    --title-bg: linear-gradient(165deg, 
        rgba(65,0,255,0.08) 0%,
        rgba(103,0,255,0.06) 40%,
        rgba(147,0,255,0.08) 100%
    );
    --title-border: rgba(147,0,255,0.2);
    --title-shimmer: rgba(147,0,255,0.1);
    --title-shadow: rgba(65,0,255,0.1);
    --version-bg: rgba(65,0,255,0.05);
    --text-primary: rgba(255,255,255,0.95);
    --text-secondary: rgba(255,255,255,0.7);
}

:root[data-theme="light"] {
    --title-bg: linear-gradient(165deg, 
        rgba(65,0,255,0.05) 0%,
        rgba(103,0,255,0.03) 40%,
        rgba(147,0,255,0.05) 100%
    );
    --title-border: rgba(147,0,255,0.15);
    --title-shimmer: rgba(147,0,255,0.08);
    --title-shadow: rgba(65,0,255,0.05);
    --version-bg: rgba(65,0,255,0.03);
    --text-primary: rgba(0,0,0,0.95);
    --text-secondary: rgba(0,0,0,0.7);
}

/* Modern animated title */
.title-container {
    position: relative;
    background: var(--title-bg);
    padding: 2.5rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    overflow: hidden;
    border: 1px solid var(--title-border);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px var(--title-shadow);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.title-container:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px var(--title-shadow);
    border-color: rgba(147,0,255,0.3);
}

.title-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    right: -50%;
    bottom: -50%;
    background: linear-gradient(45deg, 
        rgba(65,0,255,0) 0%, 
        var(--title-shimmer) 50%,
        rgba(65,0,255,0) 100%);
    transform: rotate(45deg) translateX(-100%);
    animation: shimmer 8s ease-in-out infinite;
    pointer-events: none;
}

@keyframes shimmer {
    0% {
        transform: rotate(45deg) translateX(-100%) scale(1);
        opacity: 0;
    }
    10% {
        opacity: 0.5;
    }
    50% {
        transform: rotate(45deg) translateX(100%) scale(1.2);
        opacity: 0.8;
    }
    90% {
        opacity: 0.5;
    }
    100% {
        transform: rotate(45deg) translateX(200%) scale(1);
        opacity: 0;
    }
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(120deg, 
        #4100ff 0%, 
        #7b00ff 30%, 
        #9300ff 70%, 
        #4100ff 100%
    );
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin: 0;
    animation: shine 4s linear infinite;
    position: relative;
    filter: drop-shadow(0 2px 4px rgba(65,0,255,0.2));
}

@keyframes shine {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.version-info {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.8rem;
    margin-top: 1rem;
    padding: 0.5rem;
    background: var(--version-bg);
    border-radius: 8px;
    backdrop-filter: blur(5px);
}

/* Analysis card styling */
.analysis-box {
    position: relative;
    width: 100%;
    min-height: 120px;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 1;
    margin: 12px 0;
    padding: 8px;
    cursor: pointer;
}

.analysis-box:hover {
    transform: translateY(-2px);
}

.analysis-box::before,
.analysis-box::after {
    content: '';
    position: absolute;
    top: 0;
    left: 50px;
    width: 50%;
    height: 100%;
    background: rgba(0, 180, 216, 0.1);
    border-radius: 8px;
    transform: skewX(15deg);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.analysis-box::after {
    filter: blur(30px);
    background: rgba(0, 119, 190, 0.1);
}

.analysis-box:hover::before,
.analysis-box:hover::after {
    transform: skewX(0deg) scaleX(1.3);
    background: linear-gradient(315deg, rgba(0, 180, 216, 0.2), rgba(0, 119, 190, 0.2));
}

.analysis-box .content {
    position: relative;
    width: 100%;
    padding: 12px 24px;
    background: rgba(20, 20, 20, 0.4);
    backdrop-filter: blur(10px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    z-index: 1;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    color: #fff;
    border: 1px solid rgba(0, 180, 216, 0.1);
}

.analysis-box:hover .content {
    background: rgba(30, 30, 30, 0.6);
    border-color: rgba(0, 180, 216, 0.3);
    box-shadow: 0 8px 25px rgba(0, 180, 216, 0.15);
}

/* Add Analysis button styling */
div[data-testid="stButton"].add-button > button {
    position: relative;
    background: var(--background-color, rgb(38, 38, 38));
    height: 64px;
    width: 256px;
    border: 1px solid rgba(65, 0, 255, 0.2);
    text-align: left !important;
    padding: 12px 20px !important;
    color: var(--text-color, rgb(249, 250, 251));
    font-size: 16px;
    font-weight: 700;
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    text-decoration: underline;
    text-underline-offset: 2px;
    text-decoration-thickness: 1px;
    display: flex !important;
    align-items: center;
    justify-content: flex-start !important;
}

div[data-testid="stButton"].add-button > button::before,
div[data-testid="stButton"].add-button > button::after {
    content: '';
    position: absolute;
    z-index: 10;
    border-radius: 9999px;
    filter: blur(16px);
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    opacity: 0.8;
}

div[data-testid="stButton"].add-button > button::before {
    width: 48px;
    height: 48px;
    right: 4px;
    top: 4px;
    background: rgb(65, 0, 255);
}

div[data-testid="stButton"].add-button > button::after {
    width: 80px;
    height: 80px;
    right: 32px;
    top: 12px;
    background: rgb(147, 0, 255);
}

div[data-testid="stButton"].add-button > button:hover {
    border-color: rgba(147, 0, 255, 0.3);
    text-decoration-thickness: 2px;
    text-underline-offset: 4px;
    color: rgb(147, 0, 255);
    transform: translateY(-1px);
}

div[data-testid="stButton"].add-button > button:hover::before {
    right: 48px;
    bottom: -32px;
    filter: blur(20px);
    box-shadow: 20px 20px 20px 30px rgba(65, 0, 255, 0.2);
    opacity: 1;
}

div[data-testid="stButton"].add-button > button:hover::after {
    right: -32px;
    opacity: 1;
}

@media (prefers-color-scheme: light) {
    div[data-testid="stButton"].add-button > button {
        background: rgba(38, 38, 38, 0.03);
        color: rgb(65, 0, 255);
        border-color: rgba(65, 0, 255, 0.15);
    }
    
    div[data-testid="stButton"].add-button > button:hover {
        background: rgba(38, 38, 38, 0.02);
        border-color: rgba(147, 0, 255, 0.3);
        color: rgb(147, 0, 255);
    }
}

/* Analysis info styling */
.analysis-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 4px;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.analysis-time {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.75em;
}

.library-info {
    color: rgba(255, 255, 255, 0.7);
    font-size: 0.75em;
    font-family: monospace;
}
</style>""", unsafe_allow_html=True)

def select_plot_type(default_key=None):
    """
    Create a dropdown for selecting between Plotly and Matplotlib visualization options
    """
    options = ["Plotly Interactive", "Matplotlib Static", "Both"]
    return st.selectbox("Plot Type", options, key=f"plot_type_{default_key}")

def show_matplotlib_plot():
    """
    Convert matplotlib plot to streamlit-displayable format and clear the current figure
    """
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return st.image(buf)

def create_page_title(title: str, version_info: str = None):
    """Create a styled page title with optional version info"""
    st.markdown(f"""
    <div class="title-container">
        <h1 class="main-title">{title}</h1>
        {f'<div class="version-info">{version_info}</div>' if version_info else '' }
    </div>
    """, unsafe_allow_html=True)

def create_analysis_card(title: str, description: str, execution_time: str, library_info: str, interpretation: str = None):
    """Create a styled analysis card"""
    st.markdown(f"""
    <div class="analysis-box">
        <div class="content">
            <h2>{title}</h2>
            <p>{description}</p>
            <div class="analysis-info">
                <span class="analysis-time">⏱ {execution_time}</span>
                <span class="library-info">🔧 {library_info}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if interpretation:
        with st.expander("How to Interpret Results"):
            st.markdown(interpretation)

def create_add_analysis_button():
    """Create a styled Add Analysis button"""
    st.markdown("""<style>
        div[data-testid="stButton"] {
            margin-top: 1.6em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    return st.button("Add Analysis", key="add_analysis_button")