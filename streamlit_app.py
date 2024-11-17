import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from itertools import zip_longest

# Clear cache to ensure fresh data loading
st.cache_data.clear()

def load_data():
    try:
        url = 'https://raw.githubusercontent.com/JyLee98ImTrying/Diet-RecipeRecommendationApp/master/df_sample.csv'
        df = pd.read_csv(url, delimiter=',', encoding='utf-8', on_bad_lines='skip')
        
        if 'Cluster' not in df.columns and 'kmeans' in st.session_state.get('models', {}):
            features = df[['Calories', 'ProteinContent', 'FatContent', 
                         'CarbohydrateContent', 'SodiumContent', 
                         'CholesterolContent', 'SaturatedFatContent', 'FiberContent', 'SugarContent']]
            scaled_features = st.session_state['models']['scaler'].transform(features)
            df['Cluster'] = st.session_state['models']['kmeans'].predict(scaled_features)
        
        st.session_state['df'] = df
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def load_models():
    try:
        model_files = {
            'kmeans': 'kmeans.pkl',
            'rf_classifier': 'rf_classifier.pkl',
            'scaler': 'scaler.pkl'
        }
        
        models = {}
        for name, file in model_files.items():
            with open(file, 'rb') as f:
                models[name] = pickle.load(f)
        
        st.session_state['models'] = models
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def format_recipe_instructions(instructions):
    """Format recipe instructions from c() format to numbered list."""
    if not isinstance(instructions, str):
        return []
    # Remove c() wrapper and split by commas
    instructions = instructions.replace('c(', '').replace(')', '')
    # Split by '", ' and clean up remaining quotes
    steps = [step.strip().strip('"') for step in instructions.split('",')]
    return steps

def combine_ingredients(quantities, parts):
    """Combine ingredient quantities and parts into natural language format."""
    if pd.isna(quantities) or pd.isna(parts):
        return []
        
    try:
        # Function to parse R-style c() format
        def parse_r_vector(text):
            if not isinstance(text, str):
                return []
            # Remove c() wrapper
            text = text.replace('c(', '').replace(')', '')
            # Split by commas and clean up
            items = text.split(',')
            # Clean each item
            cleaned = []
            for item in items:
                item = item.strip().strip('"').strip("'")
                if item.upper() != 'NA':  # Skip NA values
                    cleaned.append(item)
            return cleaned

        # Parse quantities and parts
        quantities_list = parse_r_vector(quantities)
        parts_list = parse_r_vector(parts)
        
        # Combine quantities and parts, but only when there's a matching part
        ingredients = []
        for i in range(len(parts_list)):  # Iterate based on parts length
            if i < len(quantities_list) and quantities_list[i] and quantities_list[i].upper() != 'NA':
                ingredients.append(f"{quantities_list[i]} {parts_list[i]}".strip())
            else:
                ingredients.append(parts_list[i])
        
        return [ing for ing in ingredients if ing]
        
    except Exception as e:
        st.error(f"Error processing ingredients: {str(e)}")
        return []


def calculate_caloric_needs(gender, weight, height, age):
    if gender == "Female":
        BMR = 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
    else:
        BMR = 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    return BMR

def recommend_food(input_data, df, models, excluded_indices=None):
    try:
        input_data_reshaped = input_data.reshape(1, -1)
        input_data_scaled = models['scaler'].transform(input_data_reshaped)
        
        # Get current parameters from session state
        wellness_goal = st.session_state.get('current_wellness_goal')
        health_condition = st.session_state.get('current_health_condition')
        user_weight = st.session_state.get('current_weight')
        
        # First, apply health condition filtering to the entire dataset
        filtered_df = df.copy()
        if health_condition == "Diabetic":
            filtered_df = filtered_df[
                (filtered_df['SugarContent'] <= 5) &  # Low sugar content
                (filtered_df['FiberContent'] >= 3)    # Higher fiber helps manage blood sugar
            ]
        elif health_condition == "High Blood Pressure":
            filtered_df = filtered_df[
                (filtered_df['SodiumContent'] <= 500)  # Max 500mg sodium per serving
            ]
        elif health_condition == "High Cholesterol":
            filtered_df = filtered_df[
                (filtered_df['CholesterolContent'] <= 50) &     # Low cholesterol
                (filtered_df['SaturatedFatContent'] <= 3)       # Low saturated fat
            ]
        
        # If health-filtered dataset is empty, use original dataset with a warning
        if filtered_df.empty:
            st.warning(f"No foods exactly match the {health_condition} criteria. Showing best alternatives.")
            filtered_df = df.copy()
        
        # Find cluster using filtered dataset
        cluster_label = models['kmeans'].predict(input_data_scaled)[0]
        cluster_data = filtered_df[filtered_df['Cluster'] == cluster_label].copy()
        
        if cluster_data.empty:
            unique_clusters = filtered_df['Cluster'].unique()
            if len(unique_clusters) > 0:
                cluster_centers = models['kmeans'].cluster_centers_
                distances = cosine_similarity(input_data_scaled, cluster_centers)
                nearest_cluster = unique_clusters[distances.argmax()]
                cluster_data = filtered_df[filtered_df['Cluster'] == nearest_cluster].copy()
            else:
                st.warning("No clusters found in the dataset.")
                return pd.DataFrame()
        
        if excluded_indices is not None:
            cluster_data = cluster_data[~cluster_data.index.isin(excluded_indices)]
            
        # Enhanced weight loss filtering with scoring
        if wellness_goal == "Lose Weight":
            cluster_data['weight_loss_score'] = (
                -0.4 * cluster_data['Calories'] +
                0.3 * cluster_data['ProteinContent'] +
                -0.3 * cluster_data['SaturatedFatContent'] +
                0.2 * cluster_data['FiberContent'] +
                -0.2 * cluster_data['SugarContent']
            )
            max_score = cluster_data['weight_loss_score'].max()
            min_score = cluster_data['weight_loss_score'].min()
            if max_score != min_score:
                cluster_data['weight_loss_score'] = (cluster_data['weight_loss_score'] - min_score) / (max_score - min_score)
                cluster_data = cluster_data.nlargest(int(len(cluster_data) * 0.5), 'weight_loss_score')
        
        # Enhanced muscle gain filtering with scoring
        if wellness_goal == "Muscle Gain" and user_weight is not None:
            daily_protein_target = user_weight  # 1g per kg
            protein_per_meal = daily_protein_target / 3
            
            cluster_data['muscle_gain_score'] = (
                0.4 * (1 / (abs(cluster_data['ProteinContent'] - protein_per_meal) + 1)) +
                0.3 * cluster_data['ProteinContent'] +
                0.2 * cluster_data['Calories'] +
                0.1 * cluster_data['CarbohydrateContent']
            )
            max_score = cluster_data['muscle_gain_score'].max()
            min_score = cluster_data['muscle_gain_score'].min()
            if max_score != min_score:
                cluster_data['muscle_gain_score'] = (cluster_data['muscle_gain_score'] - min_score) / (max_score - min_score)
                cluster_data = cluster_data.nlargest(int(len(cluster_data) * 0.5), 'muscle_gain_score')
        
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 
                          'CarbohydrateContent', 'SodiumContent', 
                          'CholesterolContent', 'SaturatedFatContent', 'FiberContent', 'SugarContent']
        
        cluster_features = cluster_data[required_columns]
        cluster_features_scaled = models['scaler'].transform(cluster_features)
        
        similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
        
        # Adjust similarities based on health condition
        if health_condition != "No Non-Communicable Disease":
            if health_condition == "Diabetic":
                sugar_penalty = 1 - (cluster_data['SugarContent'] / cluster_data['SugarContent'].max())
                similarities = similarities * (1 + sugar_penalty)
            elif health_condition == "High Blood Pressure":
                sodium_penalty = 1 - (cluster_data['SodiumContent'] / cluster_data['SodiumContent'].max())
                similarities = similarities * (1 + sodium_penalty)
            elif health_condition == "High Cholesterol":
                cholesterol_penalty = 1 - (cluster_data['CholesterolContent'] / cluster_data['CholesterolContent'].max())
                similarities = similarities * (1 + cholesterol_penalty)
        
        cluster_data['Similarity'] = similarities
        
        rf_predictions = models['rf_classifier'].predict(cluster_features_scaled)
        cluster_data['Classification'] = rf_predictions
        
        final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(
            by='Similarity', ascending=False
        )
        
        if final_recommendations.empty:
            final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)
        
        # Include ingredient columns in the result
        result = final_recommendations[['Name', 'Calories', 'ProteinContent', 'FatContent', 
                                      'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                                      'SaturatedFatContent', 'SugarContent', 'RecipeInstructions',
                                      'RecipeIngredientQuantities', 'RecipeIngredientParts']]
        
        # Enhanced statistics display based on health condition and wellness goal
        if not result.empty:
            st.write("\nRecommendation Statistics:")
            
            # Base statistics
            st.write(f"Average Calories: {result['Calories'].head().mean():.2f} kcal")
            st.write(f"Average Protein Content: {result['ProteinContent'].head().mean():.2f}g")
            
            # Health condition specific statistics
            if health_condition == "Diabetic":
                st.write(f"Average Sugar Content: {result['SugarContent'].head().mean():.2f}g")
            elif health_condition == "High Blood Pressure":
                st.write(f"Average Sodium Content: {result['SodiumContent'].head().mean():.2f}mg")
            elif health_condition == "High Cholesterol":
                st.write(f"Average Cholesterol: {result['CholesterolContent'].head().mean():.2f}mg")
                st.write(f"Average Saturated Fat: {result['SaturatedFatContent'].head().mean():.2f}g")
            
            # Wellness goal specific statistics
            if wellness_goal == "Muscle Gain":
                st.write(f"Target Protein per Meal: {user_weight/3:.2f}g")
            elif wellness_goal == "Lose Weight":
                st.write(f"Average Fat Content: {result['FatContent'].head().mean():.2f}g")
        
        return result
                                    
    except Exception as e:
        st.error(f"Error in recommendation process: {str(e)}")
        st.write("Full error details:", e)
        return pd.DataFrame()

# Sidebar for Page Navigation
with st.sidebar.expander("Navigation", expanded=True):
    page = st.radio("Go to:", ["🍅🧀MyHealthMyFood🥑🥬", "🔎Search & Visualize📊"])

# Load data and models first
df = load_data()
models = load_models()

# Streamlit UI (Recommendation Page)
if page == "🍅🧀MyHealthMyFood🥑🥬":
    st.title('🍅🧀MyHealthMyFood🥑🥬')

    # Initialize session state for storing previous recommendations
    if 'previous_recommendations' not in st.session_state:
        st.session_state.previous_recommendations = set()
    if 'all_recommendations_cache' not in st.session_state:
        st.session_state.all_recommendations_cache = None
    
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
    
    def format_recipe_instructions(instructions):
        """Format recipe instructions from c() format to numbered list."""
        if not isinstance(instructions, str):
            return []
        # Remove c() wrapper and split by commas
        instructions = instructions.replace('c(', '').replace(')', '')
        # Split by '", ' and clean up remaining quotes
        steps = [step.strip().strip('"') for step in instructions.split('",')]
        return steps
    
    def display_recommendations(recommendations):
        """Display recommendations in a vertical format with expandable recipe instructions."""
        if not recommendations.empty:
            st.write("### 🍳 Recommended Food Items (Single Serving)")
            
            # Display each recipe in a vertical format
            for idx, row in recommendations.iterrows():
                with st.expander(f"📗 {row['Name']}"):
                    # Create three columns for better layout
                    col1, col2 = st.columns(2)
                    
                    # Nutritional Information in first column
                    with col1:
                        st.write("**📊 Nutritional Information**")
                        st.write(f"• Calories: {row['Calories']:.1f}")
                        st.write(f"• Protein: {row['ProteinContent']:.1f}g")
                        st.write(f"• Fat: {row['FatContent']:.1f}g")
                        st.write(f"• Carbohydrates: {row['CarbohydrateContent']:.1f}g")
                    
                    # Additional nutritional details in second column
                    with col2:
                        st.write("**🔍 Additional Details**")
                        st.write(f"• Sodium: {row['SodiumContent']:.1f}mg")
                        st.write(f"• Cholesterol: {row['CholesterolContent']:.1f}mg")
                        st.write(f"• Saturated Fat: {row['SaturatedFatContent']:.1f}g")
                        st.write(f"• Sugar: {row['SugarContent']:.1f}g")
                    
                    # Ingredients section
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
                    
                    # Recipe Instructions
                    st.write("**👩‍🍳 Recipe Instructions**")
                    instructions = format_recipe_instructions(row['RecipeInstructions'])
                    for i, step in enumerate(instructions, 1):
                        st.write(f"{i}. {step}")
        else:
            st.warning("No recommendations found. Please try different inputs.")

    
    # In your main code, replace the recommendation display section with this:
    if st.button("Get Recommendations"):
        daily_calories = calculate_caloric_needs(gender, weight, height, age)
        protein_grams = 0.8 * weight
        fat_calories = 0.25 * daily_calories
        carb_calories = 0.55 * daily_calories
        fat_grams = fat_calories / 9
        carb_grams = carb_calories / 4
        meal_fraction = 0.3
        
        # Reset previous recommendations when getting new recommendations
        st.session_state.previous_recommendations = set()
        
        input_features = np.array([
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
                
        # Store in session state
        st.session_state.current_input_features = input_features
        st.session_state.current_wellness_goal = wellness_goal
        st.session_state.current_weight = weight
        st.session_state.current_health_condition = health_condition
    
        
        # Get initial recommendations
        recommendations = recommend_food(input_features, df, models)
        
        # Store all recommendations in cache for reshuffling
        if not recommendations.empty:
            st.session_state.all_recommendations_cache = recommendations
            # Store the indices of shown recommendations
            st.session_state.previous_recommendations.update(recommendations.index[:5].tolist())
            # Display only top 5 recommendations
            display_recommendations(recommendations.head(5))
        else:
            st.warning("No recommendations found. Please try different inputs.")
    
    # Update the reshuffle button section similarly:
    if st.button("Reshuffle Recommendations") and hasattr(st.session_state, 'all_recommendations_cache'):
        if st.session_state.all_recommendations_cache is not None:
            # Get all recommendations excluding previously shown ones
            remaining_recommendations = st.session_state.all_recommendations_cache[
                ~st.session_state.all_recommendations_cache.index.isin(st.session_state.previous_recommendations)
            ]
            
            if not remaining_recommendations.empty:
                # Get next 5 recommendations
                new_recommendations = remaining_recommendations.head(5)
                # Update shown recommendations
                st.session_state.previous_recommendations.update(new_recommendations.index.tolist())
                # Display new recommendations
                display_recommendations(new_recommendations)
            else:
                st.warning("No more recommendations available. Please try adjusting your inputs for more options.")
        else:
            st.warning("Please get initial recommendations first.")

# Search and Visualization Page
elif page == "🔎Search & Visualize📊":
    st.title("🔎Search & Visualize📊")

    # Initialize session state for pagination
    if 'search_page' not in st.session_state:
        st.session_state['search_page'] = 0
    
    # Define the format_recipe_instructions function
    def format_recipe_instructions(instructions):
        """Format recipe instructions from c() format to numbered list."""
        if not isinstance(instructions, str):
            return []
        # Remove c() wrapper and split by commas
        instructions = instructions.replace('c(', '').replace(')', '')
        # Split by '", ' and clean up remaining quotes
        steps = [step.strip().strip('"') for step in instructions.split('",')]
        return steps
    
    # Search Function
    st.subheader("Search for Recipes")
    search_query = st.text_input("Enter a keyword to search for recipes:")
    
    if search_query:
        # Filter recipes based on the search query
        search_results = df[df['Name'].str.contains(search_query, case=False, na=False)]
    
        # Define a helper function to display a subset of recommendations
       def display_search_recommendations(recommendations, start_index, num_items=5):
    """Display a subset of recommendations with ingredients."""
    if not recommendations.empty:
        # Limit to the current page's results
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
                
                # Ingredients section
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
    
        # Display the current page of results
        st.write(f"### 🍳 Recipes Matching '{search_query}'")
        display_recommendations(search_results, start_index)
    
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
        
        # Visualization Options
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
                # Plot histogram
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
