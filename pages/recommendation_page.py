import streamlit as st
import numpy as np
from utils.recipe_utils import calculate_caloric_needs, format_recipe_instructions, combine_ingredients
from utils.recommendation_engine import recommend_food

def render_page(df, models):
    st.title('🍅🧀MyHealthMyFood🥑🥬')
    
    if df is not None and models is not None:
        # User inputs
        gender = st.selectbox("Select your gender", ["Female", "Male"])
        weight = st.number_input("Enter your weight (kg)", min_value=30, max_value=200, value=70)
        height = st.number_input("Enter your height (cm)", min_value=100, max_value=250, value=160)
        age = st.number_input("Enter your age (years)", min_value=1, max_value=100, value=30)
        health_condition = st.selectbox("Select your health condition", 
                                      ["No Non-Communicable Disease", "Diabetic", "High Blood Pressure", "High Cholesterol"])
        
        wellness_goal = None
        if health_condition == "No Non-Communicable Disease":
            wellness_goal = st.selectbox("Select your wellness goal", 
                                       ["Maintain Weight", "Lose Weight", "Muscle Gain"])

        if st.button("Get Recommendations"):
            _handle_recommendation_request(df, models, gender, weight, height, age, wellness_goal, health_condition)
            
        if st.button("Reshuffle Recommendations"):
            _handle_reshuffle_request()

def _handle_recommendation_request(df, models, gender, weight, height, age, wellness_goal, health_condition):
    """Handle initial recommendation request"""
    daily_calories = calculate_caloric_needs(gender, weight, height, age)
    input_features = _calculate_input_features(daily_calories, weight)
    
    # Store in session state
    st.session_state.current_input_features = input_features
    st.session_state.current_wellness_goal = wellness_goal
    st.session_state.current_weight = weight
    st.session_state.current_health_condition = health_condition
    
    # Reset previous recommendations
    st.session_state.previous_recommendations = set()
    
    # Get recommendations
    recommendations = recommend_food(input_features, df, models)
    
    if not recommendations.empty:
        st.session_state.all_recommendations_cache = recommendations
        st.session_state.previous_recommendations.update(recommendations.index[:5].tolist())
        _display_recommendations(recommendations.head(5))
    else:
        st.warning("No recommendations found. Please try different inputs.")

def _handle_reshuffle_request():
    """Handle reshuffle recommendation request"""
    if hasattr(st.session_state, 'all_recommendations_cache'):
        if st.session_state.all_recommendations_cache is not None:
            remaining_recommendations = st.session_state.all_recommendations_cache[
                ~st.session_state.all_recommendations_cache.index.isin(st.session_state.previous_recommendations)
            ]
            
            if not remaining_recommendations.empty:
                new_recommendations = remaining_recommendations.head(5)
                st.session_state.previous_recommendations.update(new_recommendations.index.tolist())
                _display_recommendations(new_recommendations)
            else:
                st.warning("No more recommendations available. Please try adjusting your inputs for more options.")
        else:
            st.warning("Please get initial recommendations first.")

def _calculate_input_features(daily_calories, weight):
    """Calculate input features for recommendation"""
    protein_grams = 0.8 * weight
    fat_calories = 0.25 * daily_calories
    carb_calories = 0.55 * daily_calories
    fat_grams = fat_calories / 9
    carb_grams = carb_calories / 4
    meal_fraction = 0.3
    
    return np.array([
        daily_calories * meal_fraction,
        protein_grams * meal_fraction,
        fat_grams * meal_fraction,
        carb_grams * meal_fraction,
        2000 * meal_fraction,
        200 * meal_fraction,
        (fat_grams * 0.01) * meal_fraction,
        (carb_grams * 0.03) * meal_fraction,
        (carb_grams * 0.01) * meal_fraction
    ]).reshape(1, -1)

def _display_recommendations(recommendations):
    """Display recommendations in a vertical format with expandable recipe instructions"""
    if not recommendations.empty:
        st.write("### 🍳 Recommended Food Items (Single Serving)")
        
        for idx, row in recommendations.iterrows():
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
