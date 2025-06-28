from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import time
import os
from config import MISTRAL_API_KEY, MISTRAL_MODEL

# Initialize Mistral client
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable not set")

client = MistralClient(api_key=MISTRAL_API_KEY)

def format_context(context_data):
    """Format context data for the AI"""
    if not context_data:
        return ""
    
    context = "Here is some relevant information that might help answer the query:\n\n"
    for data_type, data in context_data.items():
        if data:
            context += f"{data_type.upper()}:\n"
            if isinstance(data, list):
                if data_type == "universities":
                    context += "\n".join([f"- {item.get('name', 'Unknown')} ({item.get('country', 'Unknown')})" for item in data[:3]])
                elif data_type == "news":
                    context += "\n".join([f"- {item.get('title', 'Unknown')} ({item.get('source', {}).get('name', 'Unknown')})" for item in data[:3]])
                elif data_type == "projects":
                    context += "\n".join([f"- {item.get('name', 'Unknown')}: {item.get('description', 'No description')}" for item in data[:3]])
                elif data_type == "books":
                    context += "\n".join([f"- {item.get('title', 'Unknown')} by {', '.join(item.get('author_name', ['Unknown']))}" for item in data[:3]])
                elif data_type == "apis":
                    context += "\n".join([f"- {item.get('API', 'Unknown')}: {item.get('Description', 'No description')}" for item in data[:3]])
                else:
                    context += str(data[:3])
            else:
                context += str(data)
            context += "\n\n"
    
    return context

def get_ai_response(user_message, context_data, max_retries=3):
    """Get AI response with retry logic"""
    context = format_context(context_data)
    
    # Create messages for the AI
    messages = [
        ChatMessage(
            role="system", 
            content="You are an educational assistant helping students find courses, projects, and educational resources. "
                    "Provide helpful, accurate information based on the context provided. "
                    "If you don't know something, admit it rather than making up information."
        ),
        ChatMessage(
            role="user", 
            content=f"{context}\n\nUser question: {user_message}"
        )
    ]
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            print(f"Sending request to Mistral AI (attempt {attempt+1}/{max_retries})")
            chat_response = client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            
            response_text = chat_response.choices[0].message.content
            print(f"Received response from Mistral AI: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            print(f"Error in attempt {attempt+1}: {str(e)}")
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1  # Exponential backoff: 1, 2, 4 seconds
                print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                # For other errors, also use backoff
                wait_time = (2 ** attempt) * 1
                print(f"Error occurred. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                # If we've exhausted all retries, return a friendly error message
                return "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."
