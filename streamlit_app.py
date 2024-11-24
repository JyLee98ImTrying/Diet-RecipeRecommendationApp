import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from itertools import zip_longest
import plotly.express as px
import xgboost as xgb

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
            'xgb_classifier': 'xgb_classifier.pkl',
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
        
        xgb_predictions = models['xgb_classifier'].predict(cluster_features_scaled)
        cluster_data['Classification'] = xgb_predictions
        
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

def create_nutrient_distribution_plot(selected_recipes):
    """
    Create a distribution plot for nutritional content of selected recipes
    
    Parameters:
    selected_recipes (pd.DataFrame): DataFrame of selected recipes
    
    Returns:
    matplotlib figure
    """
    # Nutrients to visualize
    nutrients = ['ProteinContent', 'FatContent', 'CarbohydrateContent', 
                 'SodiumContent', 'CholesterolContent', 
                 'SaturatedFatContent', 'SugarContent']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(nutrients), 1, figsize=(10, 4*len(nutrients)))
    fig.suptitle('Nutritional Content Distribution of Selected Recipes', fontsize=16)
    
    # Plot distribution for each nutrient
    for i, nutrient in enumerate(nutrients):
        sns.boxplot(x=selected_recipes[nutrient], ax=axes[i])
        axes[i].set_title(f'{nutrient} Distribution')
        axes[i].set_xlabel('Content (g/serving)')
    
    plt.tight_layout()
    return fig

def create_calories_summary_plot(selected_recipes):
    """
    Create a bar plot summarizing calories of selected recipes
    
    Parameters:
    selected_recipes (pd.DataFrame): DataFrame of selected recipes
    
    Returns:
    matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    plt.bar(selected_recipes['Name'], selected_recipes['Calories'])
    plt.title('Calories in Selected Recipes', fontsize=16)
    plt.xlabel('Recipe Name')
    plt.ylabel('Calories (kcal)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()

# Sidebar for Page Navigation
with st.sidebar.expander("Navigation", expanded=True):
    page = st.radio("Go to:", ["ReadMe 📖", "🍅🧀MyHealthMyFood🥑🥬", "🔎Search for Recipes", "Recipe Data Visualization📊", "⚖️Weight Loss Prediction"])

# Load data and models first
df = load_data()
models = load_models()

def render_readme_page():
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
    
    # Optional: Add a visual separator or additional guidance
    st.markdown("---")
    st.info("Explore recipes, discover nutrition, and enjoy your culinary journey!")

# If this is part of a multi-page Streamlit app
if page == "ReadMe 📖":
    render_readme_page()

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_recipes' not in st.session_state:
    st.session_state.selected_recipes = set()
if 'previous_recommendations' not in st.session_state:
    st.session_state.previous_recommendations = set()
if 'all_recommendations_cache' not in st.session_state:
    st.session_state.all_recommendations_cache = None

# Streamlit UI (Recommendation Page)
if page == "🍅🧀MyHealthMyFood🥑🥬":
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
    
    def format_recipe_instructions(instructions):
        """Format recipe instructions from c() format to numbered list."""
        if not isinstance(instructions, str):
            return []
        # Remove c() wrapper and split by commas
        instructions = instructions.replace('c(', '').replace(')', '')
        # Split by '", ' and clean up remaining quotes
        steps = [step.strip().strip('"') for step in instructions.split('",')]
        return steps
    
    def display_recommendations_with_selection(recommendations, key_prefix=''):
        """
        Display recommendations with checkboxes for selection with enhanced key management
        
        Parameters:
        recommendations (pd.DataFrame): DataFrame of recipe recommendations
        key_prefix (str): Unique prefix for checkbox keys to avoid collision
        
        Returns:
        pd.DataFrame: Selected recipes
        """
        if 'current_recommendations' not in st.session_state:
            st.session_state.current_recommendations = None
    
        # Store or retrieve recommendations
        if recommendations is not None and not recommendations.empty:
            st.session_state.current_recommendations = recommendations
        else:
            recommendations = st.session_state.current_recommendations
    
        if recommendations is not None and not recommendations.empty:
            st.write("### 🍳 Recommended Food Items (Single Serving)")
                
            # Create a container to hold selections
            selection_container = st.container()
            
            with selection_container:
                selected_recipes = []
                for idx, row in recommendations.iterrows():
                    unique_key = f'recipe_select_{key_prefix}_{idx}'
                    
                    with st.expander(f"📗 {row['Name']}"):
                        is_selected = st.checkbox(
                            "Select this recipe",
                            key=unique_key,
                            value=idx in st.session_state.get('selected_recipe_indices', set())
                        )
                        
                        # Update selection state
                        if 'selected_recipe_indices' not in st.session_state:
                            st.session_state.selected_recipe_indices = set()
                        
                        if is_selected:
                            st.session_state.selected_recipe_indices.add(idx)
                            selected_recipes.append(row)
                        else:
                            st.session_state.selected_recipe_indices.discard(idx)
                        
                        # Display recipe details
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
                        
                        # Recipe Instructions
                        st.write("**👩‍🍳 Recipe Instructions**")
                        instructions = format_recipe_instructions(row['RecipeInstructions'])
                        for i, step in enumerate(instructions, 1):
                            st.write(f"{i}. {step}")
            
            # Prepare selected recipes
            if selected_recipes:
                st.write("### 🍽️ Selected Recipes")
                selected_df = pd.DataFrame(selected_recipes)
                for name in selected_df['Name']:
                    st.write(f"• {name}")
                
                if st.button("Visualize Selected Recipes", key=f'{key_prefix}_visualize'):
                    st.write("### 🍽️ Nutritional Content Distribution")
                    fig1 = create_nutrient_distribution_plot(selected_df)
                    st.pyplot(fig1)
                    
                    st.write("### 🔢 Calories Breakdown")
                    fig2 = create_calories_summary_plot(selected_df)
                    st.pyplot(fig2)
        
            return recommendations
        else:
            if not st.session_state.get('current_recommendations'):
                st.warning("No recommendations found. Please try different inputs.")
            return pd.DataFrame()

    
    if st.button("Get Recommendations"):
        daily_calories = calculate_caloric_needs(gender, weight, height, age)
        protein_grams = 0.8 * weight
        fat_calories = 0.25 * daily_calories
        carb_calories = 0.55 * daily_calories
        fat_grams = fat_calories / 9
        carb_grams = carb_calories / 4
        meal_fraction = 0.3
        
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
            display_recommendations_with_selection(recommendations.head(5))
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
                display_recommendations_with_selection(new_recommendations)
            else:
                st.warning("No more recommendations available. Please try adjusting your inputs for more options.")
        else:
            st.warning("Please get initial recommendations first.")

# Search and Visualization Page
elif page == "🔎Search for Recipes":
    st.title("🔎Search for Recipes")

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
        start_index = st.session_state['search_page'] * 5
        display_search_recommendations(search_results, start_index)
        
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

# Add the visualization page rendering
elif page == "Recipe Data Visualization📊":
    def visualization_page(df):
        st.title("Recipe Data Visualization📊")
        
        # Data preprocessing
        # Convert TotalTime to numeric, removing any non-numeric characters
        try:
            # First ensure TotalTime is string type before string operations
            df['TotalTime'] = df['TotalTime'].astype(str)
            # Then extract numbers and convert to numeric
            df['TotalTime'] = pd.to_numeric(df['TotalTime'].str.extract('(\d+)')[0], errors='coerce')
        except Exception as e:
            st.warning(f"Error processing TotalTime: {str(e)}")
            df['TotalTime'] = pd.NA
        
        # Ensure numeric columns are properly converted
        numeric_columns = ['Calories', 'FatContent', 'CarbohydrateContent', 'ProteinContent', 'RecipeYield']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing categories
        df['RecipeCategory'] = df['RecipeCategory'].fillna('Uncategorized')
        
        # Sidebar filters
        st.sidebar.header("Filters")
        available_categories = sorted(df['RecipeCategory'].unique())
        default_categories = ["Breakfast", "Lunch/Snacks", "Dinner"]
        # Filter default categories to include only those present in the dataset
        valid_defaults = [category for category in default_categories if category in available_categories] 
        selected_category = st.sidebar.multiselect(
            "Select Recipe Categories",
            options=available_categories,
            default=valid_defaults
        )
        
        # Filter data based on selection
        if selected_category:
            filtered_df = df[df['RecipeCategory'].isin(selected_category)]
        else:
            filtered_df = df
        
        # Only proceed with visualization if we have data
        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
            return
            
        # Interactive Chart 1: Scatter plot of Cooking Time vs Calories
        st.subheader("Cooking Time vs Calories by Category")
        # Remove rows where TotalTime or Calories is null for this visualization
        scatter_df = filtered_df.dropna(subset=['TotalTime', 'Calories'])
        if not scatter_df.empty:
            fig1 = px.scatter(
                scatter_df,
                x='TotalTime',
                y='Calories',
                color='RecipeCategory',
                hover_data=['Name'],
                title='Recipe Cooking Time vs Calories',
                labels={'TotalTime': 'Cooking Time (minutes)', 
                        'Calories': 'Calories',
                        'RecipeCategory': 'Category'}
            )
            st.plotly_chart(fig1)
        else:
            st.warning("Insufficient data for cooking time vs calories visualization")
        
        # Interactive Chart 2: Nutrient Distribution by Category
        st.subheader("Nutrient Distribution by Category")
        nutrients = {
            'ProteinContent': 'Protein (g)',
            'CarbohydrateContent': 'Carbohydrates (g)',
            'FatContent': 'Fat (g)'
        }
        selected_nutrient = st.selectbox("Select Nutrient", list(nutrients.keys()), 
                                       format_func=lambda x: nutrients[x])
        
        # Remove null values for the selected nutrient
        box_df = filtered_df.dropna(subset=[selected_nutrient])
        if not box_df.empty:
            fig2 = px.box(
                box_df,
                x='RecipeCategory',
                y=selected_nutrient,
                points='all',
                title=f'{nutrients[selected_nutrient]} Distribution by Category'
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2)
        else:
            st.warning(f"Insufficient data for {nutrients[selected_nutrient]} distribution")
        
        # Interactive Chart 3: Top Recipes by Calories
        st.subheader("Top Recipes by Calories")
        num_recipes = st.slider("Select number of recipes to display", 5, 20, 10)
        
        # Remove null calories for top recipes chart
        calories_df = filtered_df.dropna(subset=['Calories'])
        if not calories_df.empty:
            top_recipes = calories_df.nlargest(num_recipes, 'Calories')
            
            fig3 = px.bar(
                top_recipes,
                x='Name',
                y='Calories',
                color='RecipeCategory',
                title=f'Top {num_recipes} Recipes by Calories',
                labels={'Name': 'Recipe Name', 'Calories': 'Calories'}
            )
            fig3.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig3)
        else:
            st.warning("Insufficient data for top recipes visualization")
        
        # EDA Charts
        st.subheader("Exploratory Data Analysis")
        
        # EDA Chart 1: Correlation Heatmap
        st.write("Correlation between Nutritional Values")
        numeric_cols = ['Calories', 'FatContent', 'CarbohydrateContent', 'ProteinContent']
        
        # Remove rows with null values for correlation
        corr_df = filtered_df[numeric_cols].dropna()
        if not corr_df.empty:
            # Create correlation matrix
            corr_matrix = corr_df.corr()
            
            fig4 = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=numeric_cols,
                y=numeric_cols,
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig4.update_layout(title='Nutrient Correlation Matrix')
            st.plotly_chart(fig4)
        else:
            st.warning("Insufficient data for correlation analysis")
        
        # EDA Chart 2: Recipe Category Distribution
        st.write("Recipe Category Distribution")
        category_counts = filtered_df['RecipeCategory'].value_counts()
        
        fig5 = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title='Recipe Category Distribution',
            labels={'x': 'Number of Recipes', 'y': 'Category'}
        )
        st.plotly_chart(fig5)
    
    # Call the visualization page function with the loaded dataframe
    if df is not None:
        visualization_page(df)
    else:
        st.error("Unable to load data for visualization. Please check the data source.")

# Add this new elif block after your visualization page code
elif page == "⚖️Weight Loss Prediction":
    st.title("⚖️Weight Loss Prediction Calculator")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        height = st.number_input("Height (cm)", min_value=120, max_value=250, value=170)
        current_weight = st.number_input("Current Weight (kg)", min_value=40, max_value=200, value=70)
        target_weight = st.number_input("Target Weight (kg)", min_value=40, max_value=200, value=65)
        
    with col2:
        st.subheader("Activity Level")
        activity_level = st.select_slider(
            "Activity Level",
            options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"],
            value="Lightly Active"
        )
        
        # Activity level multipliers
        activity_multipliers = {
            "Sedentary": 1.2,        # Little or no exercise
            "Lightly Active": 1.375,  # Light exercise/sports 1-3 days/week
            "Moderately Active": 1.55,# Moderate exercise/sports 3-5 days/week
            "Very Active": 1.725,     # Hard exercise/sports 6-7 days/week
            "Extra Active": 1.9       # Very hard exercise & physical job or training twice per day
        }
        
        # Weight loss goal
        weekly_goal = st.select_slider(
            "Weekly Weight Loss Goal",
            options=["Slow (0.25kg)", "Moderate (0.5kg)", "Fast (0.75kg)", "Very Fast (1kg)"],
            value="Moderate (0.5kg)"
        )
        
        # Extract numeric value from weekly goal
        goal_multipliers = {
            "Slow (0.25kg)": 0.25,
            "Moderate (0.5kg)": 0.5,
            "Fast (0.75kg)": 0.75,
            "Very Fast (1kg)": 1.0
        }

    if st.button("Calculate Weight Loss Prediction"):
        # Calculate BMR using Mifflin-St Jeor Equation
        if gender == "Male":
            bmr = 10 * current_weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * current_weight + 6.25 * height - 5 * age - 161
            
        # Calculate TDEE (Total Daily Energy Expenditure)
        tdee = bmr * activity_multipliers[activity_level]
        
        # Calculate daily calorie deficit needed for selected weekly loss
        # 1 kg of fat = 7700 calories
        weekly_loss = goal_multipliers[weekly_goal]
        daily_deficit = (weekly_loss * 7700) / 7
        
        # Calculate target daily calories
        target_calories = tdee - daily_deficit
        
        # Calculate time to reach goal
        weight_to_lose = current_weight - target_weight
        weeks_to_goal = weight_to_lose / weekly_loss
        
        # Calculate target date
        target_date = datetime.datetime.now() + datetime.timedelta(weeks=weeks_to_goal)
        
        # Display Results
        st.markdown("---")
        st.subheader("📊 Your Weight Loss Prediction")
        
        # Create three columns for metrics
        metric1, metric2, metric3 = st.columns(3)
        
        with metric1:
            st.metric(
                label="Daily Calories Needed",
                value=f"{int(target_calories)} kcal",
                delta=f"-{int(daily_deficit)} kcal"
            )
            
        with metric2:
            st.metric(
                label="Weeks to Goal",
                value=f"{weeks_to_goal:.1f} weeks"
            )
            
        with metric3:
            st.metric(
                label="Target Date",
                value=target_date.strftime("%Y-%m-%d")
            )
        
        # Additional Information
        st.markdown("---")
        st.subheader("📋 Detailed Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Energy Expenditure**")
            st.write(f"• Base Metabolic Rate (BMR): {int(bmr)} kcal")
            st.write(f"• Total Daily Energy Expenditure: {int(tdee)} kcal")
            st.write(f"• Activity Multiplier: {activity_multipliers[activity_level]:.2f}x")
            
        with col2:
            st.write("**Weight Loss Plan**")
            st.write(f"• Weekly Weight Loss Goal: {weekly_loss} kg")
            st.write(f"• Daily Calorie Deficit: {int(daily_deficit)} kcal")
            st.write(f"• Total Weight to Lose: {weight_to_lose:.1f} kg")
        
        # Health Warning
        if target_calories < 1200 and gender == "Female" or target_calories < 1500 and gender == "Male":
            st.warning("""
                ⚠️ Warning: The calculated daily calories are below the recommended minimum intake. 
                Consider adjusting your weekly weight loss goal to a more sustainable rate or 
                consulting with a healthcare provider.
            """)
            
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        st.write("""
            To achieve your weight loss goals safely:
            1. Combine your calorie deficit with regular physical activity
            2. Focus on nutrient-dense, whole foods
            3. Stay hydrated by drinking plenty of water
            4. Get adequate sleep (7-9 hours per night)
            5. Track your progress regularly but don't obsess over daily fluctuations
            
            Remember: This is an estimate based on general calculations. Individual results may vary 
            based on factors such as metabolism, medical conditions, and consistency with the plan.
        """)
