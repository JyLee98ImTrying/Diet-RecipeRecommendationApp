"""
Initialize the pages package, making page classes available for import.
"""
from .readme_page import ReadmePage
from .recommendation_page import RecommendationPage
from .search_visualize_page import SearchVisualizePage

# Export classes for easier imports
__all__ = [
    'ReadmePage',
    'RecommendationPage',
    'SearchVisualizePage'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Your Name'

# Optional: Page configuration that can be used across different pages
DEFAULT_PAGE_CONFIG = {
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Optional: Common page utilities
PAGE_NAVIGATION = {
    "ReadMe 📖": "readme",
    "🍅🧀MyHealthMyFood🥑🥬": "recommendation",
    "🔎Search & Visualize📊": "search"
}
