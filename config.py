
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

# API Endpoints
APIS = {
    "universities": "http://universities.hipolabs.com/search",
    "news": "https://newsapi.org/v2/everything",
    "github_trending": "https://api.github.com/search/repositories",
    "public_apis": "https://api.publicapis.org/entries",
    "open_library": "https://openlibrary.org/search.json"
}

# Mistral AI settings
MISTRAL_MODEL = "mistral-tiny"  # or "mistral-small", "mistral-medium" for better quality
