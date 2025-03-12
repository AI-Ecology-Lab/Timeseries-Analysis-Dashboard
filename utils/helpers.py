import streamlit as st
import os


def set_page_config(title, layout="wide", sidebar_state="expanded"):
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title=title,
        page_icon="🦑",
        layout=layout,
        initial_sidebar_state=sidebar_state,
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    
    # Enforce dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)


def load_css():
    """Load custom CSS styles"""
    css_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "css", "style.css")
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def typed_text(text, container=None):
    """Create text with typewriter animation effect"""
    markdown_component = container.markdown if container else st.markdown
    
    # Using container width to help with responsive design
    markdown_component(
        f'<h1 class="main-heading" style="display: inline-block; text-align: center; width: 100%;">{text}</h1>',
        unsafe_allow_html=True
    )


def animated_text(text, animation_class="typing", tag="h1", css_class="main-heading"):
    """Create animated text with CSS animations"""
    if animation_class == "typing":
        st.markdown(f'<{tag} class="{css_class}">{text}</{tag}>', unsafe_allow_html=True)
    else:
        st.markdown(f'<{tag} class="{css_class}" style="animation-name: {animation_class};">{text}</{tag}>', unsafe_allow_html=True)


def styled_button(text, key=None, css_class="ui-btn", on_click=None):
    """Create a custom styled button using UI-btn styling"""
    if on_click:
        clicked = st.button(text, key=key, on_click=on_click)
    else:
        clicked = st.button(text, key=key)
    
    # Apply custom UI-btn styling to the button
    st.markdown(f"""
    <style>
    div.stButton > button:first-child {{
        box-sizing: border-box;
        padding: 15px 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font: 600 16px Menlo,Roboto Mono,monospace;
        background: rgb(41, 41, 41);
        border: none;
        cursor: pointer;
        transition: 0.3s;
        overflow: hidden;
        box-shadow: 0 2px 10px 0 rgba(0, 0, 0, 0.137);
        border-radius: 0;
    }}
    div.stButton > button:first-child:hover {{
        background: rgb(51, 51, 51);
    }}
    div.stButton > button:first-child::after {{
        content: "";
        position: relative;
        display: inline-block;
        margin-left: 7px;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    return clicked


def create_data_card(title, content):
    """Create a styled card for data display"""
    card_html = f"""
    <div class="data-card">
        <h3>{title}</h3>
        <div>{content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def center_content():
    """Center content in the page"""
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    return lambda: st.markdown('</div>', unsafe_allow_html=True)