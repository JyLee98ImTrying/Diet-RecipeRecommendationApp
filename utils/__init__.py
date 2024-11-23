"""
Initialize the utils package, making utility functions and classes available for import.
"""
from .data_loader import (
    load_data,
    load_models,
)

from .recipe_utils import (
    format_recipe_instructions,
    combine_ingredients,
    create_nutrient_distribution_plot,
    create_calories_summary_plot
)

from .recommendation_engine import (
    recommend_food,
    calculate_caloric_needs
)

# Export all functions for easier imports
__all__ = [
    # Data loading utilities
    'load_data',
    'load_models',
    
    # Recipe processing utilities
    'format_recipe_instructions',
    'combine_ingredients',
    'create_nutrient_distribution_plot',
    'create_calories_summary_plot',
    
    # Recommendation utilities
    'recommend_food',
    'calculate_caloric_needs'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Your Name'

# Configuration constants that might be used across the application
DATA_CONFIG = {
    'CSV_URL': 'https://raw.githubusercontent.com/JyLee98ImTrying/Diet-RecipeRecommendationApp/master/df_sample.csv',
    'ENCODING': 'utf-8',
    'DELIMITER': ',',
}

MODEL_FILES = {
    'kmeans': 'kmeans.pkl',
    'rf_classifier': 'rf_classifier.pkl',
    'scaler': 'scaler.pkl'
}

# Nutrition-related constants
NUTRITION_COLUMNS = [
    'Calories',
    'ProteinContent',
    'FatContent',
    'CarbohydrateContent',
    'SodiumContent',
    'CholesterolContent',
    'SaturatedFatContent',
    'FiberContent',
    'SugarContent'
]

# Health condition related constants
HEALTH_CONDITIONS = {
    "No Non-Communicable Disease": {
        "constraints": {},
        "wellness_goals": ["Maintain Weight", "Lose Weight", "Muscle Gain"]
    },
    "Diabetic": {
        "constraints": {
            "max_sugar": 5,
            "min_fiber": 3
        }
    },
    "High Blood Pressure": {
        "constraints": {
            "max_sodium": 500
        }
    },
    "High Cholesterol": {
        "constraints": {
            "max_cholesterol": 50,
            "max_saturated_fat": 3
        }
    }
}
