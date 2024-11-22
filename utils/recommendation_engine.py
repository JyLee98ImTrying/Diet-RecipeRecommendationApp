import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd

def recommend_food(input_data, df, models, excluded_indices=None):
    try:
        input_data_reshaped = input_data.reshape(1, -1)
        input_data_scaled = models['scaler'].transform(input_data_reshaped)
        
        # Get current parameters from session state
        wellness_goal = st.session_state.get('current_wellness_goal')
        health_condition = st.session_state.get('current_health_condition')
        user_weight = st.session_state.get('current_weight')
        
        # Health condition filtering
        filtered_df = df.copy()
        if health_condition == "Diabetic":
            filtered_df = filtered_df[
                (filtered_df['SugarContent'] <= 5) &
                (filtered_df['FiberContent'] >= 3)
            ]
        elif health_condition == "High Blood Pressure":
            filtered_df = filtered_df[
                (filtered_df['SodiumContent'] <= 500)
            ]
        elif health_condition == "High Cholesterol":
            filtered_df = filtered_df[
                (filtered_df['CholesterolContent'] <= 50) &
                (filtered_df['SaturatedFatContent'] <= 3)
            ]
        
        if filtered_df.empty:
            st.warning(f"No foods exactly match the {health_condition} criteria. Showing best alternatives.")
            filtered_df = df.copy()
        
        # Cluster prediction and filtering
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
            
        # Wellness goal scoring
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
        
        elif wellness_goal == "Muscle Gain" and user_weight is not None:
            daily_protein_target = user_weight
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
        
        # Feature selection and similarity calculation
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 
                          'CarbohydrateContent', 'SodiumContent', 
                          'CholesterolContent', 'SaturatedFatContent', 'FiberContent', 'SugarContent']
        
        cluster_features = cluster_data[required_columns]
        cluster_features_scaled = models['scaler'].transform(cluster_features)
        similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
        
        # Health condition penalties
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
        
        # Random Forest classification
        rf_predictions = models['rf_classifier'].predict(cluster_features_scaled)
        cluster_data['Classification'] = rf_predictions
        
        final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(
            by='Similarity', ascending=False
        )
        
        if final_recommendations.empty:
            final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)
        
        # Select columns for result
        result = final_recommendations[['Name', 'Calories', 'ProteinContent', 'FatContent', 
                                      'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                                      'SaturatedFatContent', 'SugarContent', 'RecipeInstructions',
                                      'RecipeIngredientQuantities', 'RecipeIngredientParts']]
        
        display_recommendation_statistics(result, health_condition, wellness_goal, user_weight)
        
        return result
                                    
    except Exception as e:
        st.error(f"Error in recommendation process: {str(e)}")
        st.write("Full error details:", e)
        return pd.DataFrame()

def display_recommendation_statistics(result, health_condition, wellness_goal, user_weight):
    """Display statistics for recommendations based on health condition and wellness goal."""
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
