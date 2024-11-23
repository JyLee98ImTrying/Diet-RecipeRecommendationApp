import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.recipe_utils import format_recipe_instructions, combine_ingredients

class SearchVisualizePage:
    def __init__(self):
        if 'search_page' not in st.session_state:
            st.session_state['search_page'] = 0

    def display_search_recommendations(self, recommendations, start_index, num_items=5):
        """Display a subset of recommendations with ingredients."""
        if not recommendations.empty:
            page_results = recommendations.iloc[start_index:start_index + num_items]
            for idx, row in page_results.iterrows():
                with st.expander(f"📗 {row['Name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**📊 Nutritional Information**")
                        st.write(f"• Calories: {row['Calories']:.1f}")
                        st.write(f"• Protein: {row['ProteinContent']:.1f}g")
                        st.write(f"• Fat: {row['FatContent']:.1f}g")
                        st.write(f"• Carbohydrates: {row['CarbohydrateContent']:.1f}g")
                    with col2:
                        st.write("**🔍 Additional Details**")
                        st.write(f"• Sodium: {row['SodiumContent']:.1f}mg")
                        st.write(f"• Cholesterol: {row['CholesterolContent']:.1f}mg")
                        st.write(f"• Saturated Fat: {row['SaturatedFatContent']:.1f}g")
                        st.write(f"• Sugar: {row['SugarContent']:.1f}g")
                    
                    st.write("**🥗 Ingredients**")
                    ingredients = combine_ingredients(
                        row.get('RecipeIngredientQuantities', ''), 
                        row.get('RecipeIngredientParts', '')
                    )
                    if ingredients:
                        for ingredient in ingredients:
                            st.write(f"• {ingredient}")
                    else:
                        st.write("No ingredient information available")
                    
                    st.write("**👩‍🍳 Recipe Instructions**")
                    instructions = format_recipe_instructions(row['RecipeInstructions'])
                    for i, step in enumerate(instructions, 1):
                        st.write(f"{i}. {step}")
        else:
            st.warning("No recipes found. Please try a different keyword.")

    def render_visualization_section(self, df):
        """Render the visualization section of the page."""
        st.subheader("Visualizations")
        visualization_type = st.selectbox(
            "Choose a visualization:",
            ["Select an option", "Ingredient Distribution", "Nutrient Comparison"]
        )
            
        if visualization_type == "Ingredient Distribution":
            st.write("### Ingredient Distribution")
            ingredient_column = st.selectbox(
                "Select an ingredient column:",
                ["SugarContent", "ProteinContent", "FatContent", "FiberContent", "SodiumContent"]
            )
            if ingredient_column:
                try:
                    st.bar_chart(df[ingredient_column].value_counts())
                except Exception as e:
                    st.error(f"Error plotting {ingredient_column}: {str(e)}")
        
        elif visualization_type == "Nutrient Comparison":
            st.write("### Nutrient Comparison")
            nutrients = ["Calories", "ProteinContent", "FatContent", "CarbohydrateContent", "SugarContent"]
            nutrient1 = st.selectbox("Select first nutrient:", nutrients)
            nutrient2 = st.selectbox("Select second nutrient:", nutrients)
                
            if nutrient1 and nutrient2:
                try:
                    st.line_chart(df[[nutrient1, nutrient2]])
                except Exception as e:
                    st.error(f"Error comparing {nutrient1} and {nutrient2}: {str(e)}")

    def render(self, df):
        """Main render method for the search and visualize page."""
        st.title("🔎Search & Visualize📊")
        
        # Search section
        st.subheader("Search for Recipes")
        search_query = st.text_input("Enter a keyword to search for recipes:")
        
        if search_query:
            search_results = df[df['Name'].str.contains(search_query, case=False, na=False)]
            
            st.write(f"### 🍳 Recipes Matching '{search_query}'")
            start_index = st.session_state['search_page'] * 5
            self.display_search_recommendations(search_results, start_index)
            
            # Pagination controls
            col1, col2 = st.columns([1, 1])
            with col1:
                if start_index > 0:
                    if st.button("Previous"):
                        st.session_state['search_page'] -= 1
            with col2:
                if start_index + 5 < len(search_results):
                    if st.button("Next"):
                        st.session_state['search_page'] += 1
        
        # Visualization section
        self.render_visualization_section(df)
