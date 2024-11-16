import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity

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
        
        cluster_label = models['kmeans'].predict(input_data_scaled)[0]
        cluster_data = df[df['Cluster'] == cluster_label].copy()
        
        if cluster_data.empty:
            unique_clusters = df['Cluster'].unique()
            if len(unique_clusters) > 0:
                cluster_centers = models['kmeans'].cluster_centers_
                distances = cosine_similarity(input_data_scaled, cluster_centers)
                nearest_cluster = unique_clusters[distances.argmax()]
                cluster_data = df[df['Cluster'] == nearest_cluster].copy()
            else:
                st.warning("No clusters found in the dataset.")
                return pd.DataFrame()
        
        if excluded_indices is not None:
            cluster_data = cluster_data[~cluster_data.index.isin(excluded_indices)]

        # Get current wellness goal from session state
        wellness_goal = st.session_state.get('current_wellness_goal')
        user_weight = st.session_state.get('current_weight')
            
        # Condition 1: Filter for weight loss
        if wellness_goal == "Lose Weight":
            filtered_data = cluster_data[
                (cluster_data['SaturatedFatContent'] <= 0.5) & 
                (cluster_data['SugarContent'] <= 2)
            ]
            
            if not filtered_data.empty:
                cluster_data = filtered_data
            else:
                st.warning("No foods match the strict weight loss criteria. Showing alternatives with lowest fat and sugar content.")
                cluster_data['combined_score'] = (cluster_data['SaturatedFatContent'] + cluster_data['SugarContent'])
                cluster_data = cluster_data.nsmallest(int(len(cluster_data) * 0.2), 'combined_score')
        
        # Condition 2: Filter and adjust for muscle gain
        if wellness_goal == "Muscle Gain" and user_weight is not None:
            # Calculate target protein per meal (assuming 3 meals per day)
            daily_protein_target = user_weight  # 1g per kg
            protein_per_meal = daily_protein_target / 3
            
            # Create a margin of ±20% around the target
            protein_lower_bound = protein_per_meal * 0.8
            protein_upper_bound = protein_per_meal * 1.2
            
            # Filter foods within the protein range
            protein_filtered_data = cluster_data[
                (cluster_data['ProteinContent'] >= protein_lower_bound) &
                (cluster_data['ProteinContent'] <= protein_upper_bound)
            ]
            
            if not protein_filtered_data.empty:
                cluster_data = protein_filtered_data
            else:
                st.warning("No foods exactly match the protein target. Showing closest alternatives.")
                cluster_data['protein_distance'] = abs(cluster_data['ProteinContent'] - protein_per_meal)
                cluster_data = cluster_data.nsmallest(int(len(cluster_data) * 0.2), 'protein_distance')
        
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 
                          'CarbohydrateContent', 'SodiumContent', 
                          'CholesterolContent', 'SaturatedFatContent', 'FiberContent', 'SugarContent']
        
        cluster_features = cluster_data[required_columns]
        cluster_features_scaled = models['scaler'].transform(cluster_features)
        
        similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
        
        if wellness_goal == "Muscle Gain":
            protein_per_meal = user_weight / 3  # Target protein per meal
            protein_distances = abs(cluster_data['ProteinContent'] - protein_per_meal)
            max_distance = protein_distances.max()
            if max_distance > 0:
                protein_scores = 1 - (protein_distances / max_distance)
                similarities = similarities * (1 + protein_scores)
        
        cluster_data['Similarity'] = similarities
        
        rf_predictions = models['rf_classifier'].predict(cluster_features_scaled)
        cluster_data['Classification'] = rf_predictions
        
        final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(
            by='Similarity', ascending=False
        )
        
        if final_recommendations.empty:
            final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)
        
        result = final_recommendations[['Name', 'Calories', 'ProteinContent', 'FatContent', 
                                    'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                                    'SaturatedFatContent', 'SugarContent', 'RecipeInstructions']].head(5)
        
        # Add statistics about the recommendations
        st.write("\nRecommendation Statistics:")
        st.write(f"Average Protein Content: {result['ProteinContent'].mean():.2f}g")
        if wellness_goal == "Muscle Gain":
            st.write(f"Target Protein per Meal: {user_weight/3:.2f}g")
        st.write(f"Average Saturated Fat: {result['SaturatedFatContent'].mean():.2f}g")
        st.write(f"Average Sugar Content: {result['SugarContent'].mean():.2f}g")
        
        return result
                                    
    except Exception as e:
        st.error(f"Error in recommendation process: {str(e)}")
        st.write("Full error details:", e)
        return pd.DataFrame()
        
# Streamlit UI
st.title('🍅🧀MyHealthMyFood🥑🥬')

# Load data and models first
df = load_data()
models = load_models()

# Initialize session state for storing previous recommendations
if 'previous_recommendations' not in st.session_state:
    st.session_state.previous_recommendations = set()

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
        st.write("### 🍳 Recommended Food Items")
        
        # Display each recipe in a vertical format
        for idx, row in recommendations.iterrows():
            with st.expander(f"📗 {row['Name']}"):
                # Create two columns for better layout
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
    
    recommendations = recommend_food(input_features, df, models)
    display_recommendations(recommendations)

# Update the reshuffle button section similarly:
if st.button("Reshuffle Recommendations") and hasattr(st.session_state, 'current_input_features'):
        recommendations = recommend_food(
            st.session_state.current_input_features,
            df,
            models,
            excluded_indices=list(st.session_state.previous_recommendations),
            wellness_goal=st.session_state.get('current_wellness_goal')
        )
        
        if not recommendations.empty:
            st.session_state.previous_recommendations.update(recommendations.index.tolist())
            st.write("New set of recommended food items:")
            st.write(recommendations)
        else:
            st.warning("No more recommendations available in this category. Try adjusting your inputs for more options.")
