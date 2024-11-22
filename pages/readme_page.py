import streamlit as st

def render_page():
    st.title('ReadMe 📖')
    
    st.markdown("""
    ## How to Use ❓
    
    The app offers two powerful features:
    
    ### 1 - The Star of the Show ⭐
    Recipes are recommended through advanced machine learning techniques:
    - **KMeans Clustering**: Clusters recipes to identify similar groups
    - **Random Forest Classification**: Classifies and predicts food items
    - **Content-Based Recommendation**: Suggests recipes based on item similarity
    
    ### 2 - Recipe Search 🔎
    - Search recipes using keywords like "Fish", "Chicken", "Egg", and more
    - View detailed nutritional information
    - Access calorie details for each recipe
    """)
    
    st.markdown("---")
    st.info("Explore recipes, discover nutrition, and enjoy your culinary journey!")
