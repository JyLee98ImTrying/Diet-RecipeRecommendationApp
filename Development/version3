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

def recommend_food(input_data, df, models, excluded_indices=None, wellness_goal=None):
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
            
        # Condition 1: Filter for weight loss
        if wellness_goal == "Lose Weight":
            cluster_data = cluster_data[
                (cluster_data['SaturatedFatContent'] <= 0.5) & 
                (cluster_data['SugarContent'] <= 2)
            ]
            
            if cluster_data.empty:
                st.warning("No foods match the weight loss criteria. Showing alternative recommendations.")
                # Fallback to original cluster data with a warning
                cluster_data = df[df['Cluster'] == cluster_label].copy()
        
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 
                          'CarbohydrateContent', 'SodiumContent', 
                          'CholesterolContent', 'SaturatedFatContent', 'FiberContent', 'SugarContent']
        
        cluster_features = cluster_data[required_columns]
        cluster_features_scaled = models['scaler'].transform(cluster_features)
        
        # Condition 2: Adjust similarity calculation for muscle gain
        if wellness_goal == "Muscle Gain":
            # Create a protein-weighted version of the input data
            protein_weighted_input = input_data_scaled.copy()
            protein_idx = required_columns.index('ProteinContent')
            protein_weighted_input[0, protein_idx] *= 6
            
            # Calculate similarity with protein-weighted input
            similarities = cosine_similarity(protein_weighted_input, cluster_features_scaled).flatten()
        else:
            similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
        
        cluster_data['Similarity'] = similarities
        
        rf_predictions = models['rf_classifier'].predict(cluster_features_scaled)
        cluster_data['Classification'] = rf_predictions
        
        final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(
            by='Similarity', ascending=False
        )
        
        if final_recommendations.empty:
            final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)
        
        return final_recommendations[['Name', 'Calories', 'ProteinContent', 'FatContent', 
                                    'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                                    'SaturatedFatContent', 'SugarContent', 'RecipeInstructions']].head(5)
                                    
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
    
    # Initialize wellness_goal
    wellness_goal = None
    if health_condition == "No Non-Communicable Disease":
        wellness_goal = st.selectbox("Select your wellness goal", 
                                   ["Maintain Weight", "Lose Weight", "Muscle Gain"])

    def recommend_food(input_data, df, models, excluded_indices=None, wellness_goal=None):
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

            # Debug information
            st.write(f"Current wellness goal: {wellness_goal}")
            st.write(f"Initial cluster size: {len(cluster_data)}")
                
            # Condition 1: Filter for weight loss
            if wellness_goal == "Lose Weight":
                st.write("Applying weight loss filters...")
                filtered_data = cluster_data[
                    (cluster_data['SaturatedFatContent'] <= 0.5) & 
                    (cluster_data['SugarContent'] <= 2)
                ]
                st.write(f"Foods matching weight loss criteria: {len(filtered_data)}")
                
                if not filtered_data.empty:
                    cluster_data = filtered_data
                else:
                    st.warning("No foods match the strict weight loss criteria. Showing alternatives with lowest fat and sugar content.")
                    # Sort by fat and sugar content and take top 20%
                    cluster_data['combined_score'] = (cluster_data['SaturatedFatContent'] + cluster_data['SugarContent'])
                    cluster_data = cluster_data.nsmallest(int(len(cluster_data) * 0.2), 'combined_score')
            
            required_columns = ['Calories', 'ProteinContent', 'FatContent', 
                              'CarbohydrateContent', 'SodiumContent', 
                              'CholesterolContent', 'SaturatedFatContent', 'FiberContent', 'SugarContent']
            
            cluster_features = cluster_data[required_columns]
            cluster_features_scaled = models['scaler'].transform(cluster_features)
            
            # Condition 2: Adjust similarity calculation for muscle gain
            if wellness_goal == "Muscle Gain":
                st.write("Applying muscle gain weightage...")
                # Create protein-weighted features
                protein_weight = 3.0  # Increased protein importance
                feature_weights = np.ones(len(required_columns))
                protein_idx = required_columns.index('ProteinContent')
                feature_weights[protein_idx] = protein_weight
                
                # Apply weights to both input and cluster features
                weighted_input = input_data_scaled * feature_weights
                weighted_cluster_features = cluster_features_scaled * feature_weights
                
                similarities = cosine_similarity(weighted_input, weighted_cluster_features).flatten()
                
                # Additional boost for high-protein foods
                protein_scores = cluster_features_scaled[:, protein_idx]
                similarities = similarities * (1 + protein_scores)
            else:
                similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
            
            cluster_data['Similarity'] = similarities
            
            rf_predictions = models['rf_classifier'].predict(cluster_features_scaled)
            cluster_data['Classification'] = rf_predictions
            
            final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(
                by='Similarity', ascending=False
            )
            
            if final_recommendations.empty:
                final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)
            
            # Add debug information to output
            st.write(f"Number of final recommendations before filtering: {len(final_recommendations)}")
            
            result = final_recommendations[['Name', 'Calories', 'ProteinContent', 'FatContent', 
                                        'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                                        'SaturatedFatContent', 'SugarContent', 'RecipeInstructions']].head(5)
            
            # Add debug information about the returned recommendations
            st.write("\nRecommendation Statistics:")
            st.write(f"Average Protein Content: {result['ProteinContent'].mean():.2f}")
            st.write(f"Average Saturated Fat: {result['SaturatedFatContent'].mean():.2f}")
            st.write(f"Average Sugar Content: {result['SugarContent'].mean():.2f}")
            
            return result
                                        
        except Exception as e:
            st.error(f"Error in recommendation process: {str(e)}")
            st.write("Full error details:", e)
            return pd.DataFrame()

    # Get Recommendations button
    if st.button("Get Recommendations"):
        # Calculate all nutritional needs
        daily_calories = calculate_caloric_needs(gender, weight, height, age)
        protein_grams = 0.8 * weight
        fat_calories = 0.25 * daily_calories
        carb_calories = 0.55 * daily_calories
        fat_grams = fat_calories / 9
        carb_grams = carb_calories / 4
        meal_fraction = 0.3
        
        # Clear previous recommendations
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
                
        # Store input features and wellness goal in session state
        st.session_state.current_input_features = input_features
        st.session_state.current_wellness_goal = wellness_goal
        
        recommendations = recommend_food(input_features, df, models, wellness_goal=wellness_goal)
        
        if not recommendations.empty:
            st.session_state.previous_recommendations.update(recommendations.index.tolist())
            st.write("Recommended food items:")
            st.write(recommendations)
        else:
            st.warning("No recommendations found. Please try different inputs.")

    # Reshuffle button
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
