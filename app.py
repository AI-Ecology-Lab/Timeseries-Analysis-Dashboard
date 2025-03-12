import streamlit as st
import os
import sys
import base64
import streamlit.components.v1 as components

# Add the utils directory to Python path for importing helper functions
sys.path.insert(0, os.path.dirname(__file__))
from utils.helpers import set_page_config, load_css

def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def main():
    set_page_config("SEAPEN Dashboard", "wide", "expanded")
    
    image_path = os.path.join(os.path.dirname(__file__), "assets", "image.png")
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon_image.png")
    
    try:
        st.logo(
            image=image_path,
            size="large",
            link=None,
            icon_image=icon_path
        )
    except Exception as e:
        st.error(f"Error loading logo: {e}")

    # Load custom CSS
    load_css()
    
    squid_path = os.path.join(os.path.dirname(__file__), "assets", "squidbg.png")
    squid_base64 = get_img_as_base64(squid_path)
    if squid_base64:
        st.markdown(f"""
        <style>
        body::before {{
            content: "";
            background-image: url("data:image/png;base64,{squid_base64}");
            background-size: contain;
            background-position: center 60%;
            background-repeat: no-repeat;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            width: 100vw;
            height: 100vh;
            z-index: 1;
            opacity: 0.6;
            pointer-events: none;
        }}
        </style>
        """, unsafe_allow_html=True)
    

    embedded_html = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000; color: white; font-family: sans-serif;">
      <div style="font-size: 3em; font-weight: bold;">Free,</div>
      <div style="font-size: 3em; font-weight: bold;">Open-Source,</div>
      <div style="font-size: 3em; font-weight: bold;">
         Marine <span id="dynamic-text">Imaging</span> Tools
      </div>
    </div>
    </div>
    <script>
      // Typewriter effect for the dynamic word.
      const words = ["Imaging", "Ecology", "Timeseries", "Statistical", "Data Viz", "Computer Vision", "Machine Learning", "Stock Management", "Conservation"];
      let wordIndex = 0;
      let charIndex = 0;
      let isDeleting = false;
      const dynamicText = document.getElementById("dynamic-text");
      function typeWriter() {
          const currentWord = words[wordIndex];
          if (!isDeleting) {
              dynamicText.innerHTML = currentWord.substring(0, charIndex + 1);
              charIndex++;
              if (charIndex === currentWord.length) {
                  isDeleting = true;
                  setTimeout(typeWriter, 1500);
              } else {
                  setTimeout(typeWriter, 200);
              }
          } else {
              dynamicText.innerHTML = currentWord.substring(0, charIndex - 1);
              charIndex--;
              if (charIndex === 0) {
                  isDeleting = false;
                  wordIndex = (wordIndex + 1) % words.length;
                  setTimeout(typeWriter, 500);
              } else {
                  setTimeout(typeWriter, 100);
              }
          }
      }
      typeWriter();
    </script>
    """
    # The width and height are set to ensure full display.
    components.html(embedded_html, height=300, scrolling=False)
    
    st.markdown("""
    <footer class="static-footer">
        <div class="footer-content">
            <span>Interested in learning marine CV development? Check out </span>
            <a href="https://oceancv.org" target="_blank" rel="noopener noreferrer">oceancv.org</a>
        </div>
    </footer>
    """, unsafe_allow_html=True)
    
    # Page navigation handling.
    if 'page' in st.query_params and st.query_params['page'] == 'welcome':
        st.switch_page('pages/welcome.py')

if __name__ == "__main__":
    main()
