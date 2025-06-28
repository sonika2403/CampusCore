import requests
from config import APIS, NEWS_API_KEY

def get_universities(query):
    """Get university information"""
    try:
        response = requests.get(f"{APIS['universities']}?name={query}")
        data = response.json()
        return data[:5] if data else []
    except Exception as e:
        print(f"Error fetching university data: {str(e)}")
        return []

def get_tech_news(query="technology education"):
    """Get technology news"""
    if not NEWS_API_KEY:
        return []
    
    try:
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "pageSize": 5,
            "sortBy": "publishedAt"
        }
        response = requests.get(APIS["news"], params=params)
        data = response.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def get_trending_projects(topic="education"):
    """Get trending GitHub projects"""
    try:
        params = {
            "q": f"topic:{topic}",
            "sort": "stars",
            "order": "desc"
        }
        response = requests.get(APIS["github_trending"], params=params)
        data = response.json()
        return data.get("items", [])[:5]
    except Exception as e:
        print(f"Error fetching GitHub projects: {str(e)}")
        return []

def get_public_apis(category="education"):
    """Get public APIs related to education"""
    try:
        params = {"category": category}
        response = requests.get(APIS["public_apis"], params=params)
        data = response.json()
        return data.get("entries", [])[:5]
    except Exception as e:
        print(f"Error fetching public APIs: {str(e)}")
        return []

def get_books(query="computer science education"):
    """Get book recommendations"""
    try:
        params = {
            "q": query,
            "limit": 5
        }
        response = requests.get(APIS["open_library"], params=params)
        data = response.json()
        return data.get("docs", [])[:5]
    except Exception as e:
        print(f"Error fetching books: {str(e)}")
        return []

def get_relevant_data(user_message):
    """Get relevant data based on user message"""
    context_data = {}
    
    # Determine what kind of information to fetch
    if any(word in user_message.lower() for word in ["university", "college", "school"]):
        query = " ".join([word for word in user_message.split() if len(word) > 3])
        context_data["universities"] = get_universities(query)
    
    if any(word in user_message.lower() for word in ["news", "trend", "latest", "current"]):
        query = "technology education"
        if "machine learning" in user_message.lower():
            query = "machine learning education"
        elif "web development" in user_message.lower():
            query = "web development education"
        context_data["news"] = get_tech_news(query)
    
    if any(word in user_message.lower() for word in ["project", "github", "code", "build", "develop"]):
        topic = "education"
        if "machine learning" in user_message.lower():
            topic = "machine-learning"
        elif "web" in user_message.lower():
            topic = "web-development"
        elif "mobile" in user_message.lower():
            topic = "mobile-development"
        context_data["projects"] = get_trending_projects(topic)
    
    if any(word in user_message.lower() for word in ["book", "read", "textbook", "resource"]):
        query = "computer science education"
        if "machine learning" in user_message.lower():
            query = "machine learning"
        elif "web development" in user_message.lower():
            query = "web development"
        context_data["books"] = get_books(query)
    
    if any(word in user_message.lower() for word in ["api", "service", "tool", "integration"]):
        context_data["apis"] = get_public_apis()
    
    return context_data
