import streamlit as st
from pages import readme_page, recommendation_page, search_visualize_page
from utils.data_loader import load_data, load_models

# Initialize session state
if 'previous_recommendations' not in st.session_state:
    st.session_state.previous_recommendations = set()
if 'all_recommendations_cache' not in st.session_state:
    st.session_state.all_recommendations_cache = None
if 'search_page' not in st.session_state:
    st.session_state['search_page'] = 0

# Load data and models
df = load_data()
models = load_models()

# Sidebar for Page Navigation
with st.sidebar.expander("Navigation", expanded=True):
    page = st.radio("Go to:", ["ReadMe 📖", "🍅🧀MyHealthMyFood🥑🥬", "🔎Search & Visualize📊"])

# Route to appropriate page
if page == "ReadMe 📖":
    readme_page.render_page()
elif page == "🍅🧀MyHealthMyFood🥑🥬":
    recommendation_page.render_page(df, models)
elif page == "🔎Search & Visualize📊":
    search_visualize_page.render_page(df)

