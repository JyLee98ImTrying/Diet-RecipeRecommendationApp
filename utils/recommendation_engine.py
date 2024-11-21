import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

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
        
        # Cluster prediction
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
            _normalize_and_filter_scores(cluster_data, 'weight_loss_score')
            
        elif wellness_goal == "Muscle Gain" and user_weight is not None:
            daily_protein_target = user_weight
            protein_per_meal = daily_protein_target / 3
            
            cluster_data['muscle_gain_score'] = (
                0.4 * (1 / (abs(cluster_data['ProteinContent'] - protein_per_meal) + 1)) +
                0.3 * cluster_data['ProteinContent'] +
                0.2 * cluster_data['Calories'] +
                0.1 * cluster_data['CarbohydrateContent']
            )
            _normalize_and_filter_scores(cluster_data, 'muscle_gain_score')
        
        # Feature similarity calculation
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 
                          'CarbohydrateContent', 'SodiumContent', 
                          'CholesterolContent', 'SaturatedFatContent', 'FiberContent', 'SugarContent']
        
        cluster_features = cluster_data[required_columns]
        cluster_features_scaled = models['scaler'].transform(cluster_features)
        similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
        
        # Health condition adjustments
        if health_condition != "No Non-Communicable Disease":
            similarities = _adjust_similarities_for_health(similarities, cluster_data, health_condition)
        
        cluster_data['Similarity'] = similarities
        
        # Random Forest classification
        rf_predictions = models['rf_classifier'].predict(cluster_features_scaled)
        cluster_data['Classification'] = rf_predictions
        
        final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(
            by='Similarity', ascending=False
        )
        
        if final_recommendations.empty:
            final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)
        
        return _prepare_final_recommendations(final_recommendations)
    
    except Exception as e:
        st.error(f"Error in recommendation process: {str(e)}")
        st.write("Full error details:", e)
        return pd.DataFrame()

def _normalize_and_filter_scores(df, score_column):
    """Helper function to normalize scores and filter top results"""
    max_score = df[score_column].max()
    min_score = df[score_column].min()
    if max_score != min_score:
        df[score_column] = (df[score_column] - min_score) / (max_score - min_score)
        return df.nlargest(int(len(df) * 0.5), score_column)
    return df

def _adjust_similarities_for_health(similarities, data, condition):
    """Adjust similarity scores based on health conditions"""
    if condition == "Diabetic":
        sugar_penalty = 1 - (data['SugarContent'] / data['SugarContent'].max())
        return similarities * (1 + sugar_penalty)
    elif condition == "High Blood Pressure":
        sodium_penalty = 1 - (data['SodiumContent'] / data['SodiumContent'].max())
        return similarities * (1 + sodium_penalty)
    elif condition == "High Cholesterol":
        cholesterol_penalty = 1 - (data['CholesterolContent'] / data['CholesterolContent'].max())
        return similarities * (1 + cholesterol_penalty)
    return similarities

def _prepare_final_recommendations(recommendations):
    """Prepare final recommendation DataFrame with selected columns"""
    return recommendations[['Name', 'Calories', 'ProteinContent', 'FatContent', 
                          'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                          'SaturatedFatContent', 'SugarContent', 'RecipeInstructions',
                          'RecipeIngredientQuantities', 'RecipeIngredientParts']]
