import os
import sys
import traceback
from flask import (
    Flask, render_template, request, jsonify, 
    session, redirect, url_for, flash, current_app
)
from werkzeug.utils import secure_filename
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo
from functools import wraps
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime 
from bson import json_util
import json
from database import users_collection, profiles_collection
from flask import jsonify
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import numpy as np
from collections import defaultdict
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

import ssl
# SSL Certificate handling
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    # Debug logging
import logging
logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NLTK resources before any requests
def init_nltk():
    try:
        wordnet.ensure_loaded()  # Force WordNet to load
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK resources initialized successfully")
    except Exception as e:
        print(f"Error initializing NLTK: {e}")

# Call initialization
init_nltk()

# Add these imports at the top of app.py, after the existing imports
try:
    from api.mistral import get_ai_response
    from api.external_apis import get_relevant_data
except ImportError:
    # Fallback function if imports fail
    def get_ai_response(message, context):
        return "The AI service is currently unavailable. Please try again later."
    def get_relevant_data(query):
        return {}
    
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)
print("Files in current directory:", os.listdir())

# Add these configurations
UPLOAD_FOLDER = 'static/uploads/profile_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5000", "http://localhost:5000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "credentials": True
    }
})
app.secret_key = "your_secret_key"  # Replace with a strong, random key in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/students_db"
mongo = PyMongo(app)
db = mongo.db
profiles_collection = db.profiles  # This is where profiles will be stored
users_collection = db.users       # This is where user accounts are stored
bcrypt = Bcrypt(app)

# After your MongoDB configuration
print("Checking MongoDB connection...")
try:
    # List all collections
    collections = db.list_collection_names()
    print(f"Available collections: {collections}")
    print(f"Number of profiles: {profiles_collection.count_documents({})}")
except Exception as e:
    print(f"MongoDB connection error: {e}")

class SearchEngine:
    def __init__(self):
        self.profiles = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.logger = logging.getLogger(__name__)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        # Lowercase and tokenize
        tokens = word_tokenize(text.lower())
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
        return " ".join(tokens)

    def prepare_profile_text(self, profile):
        """Extract and combine searchable text from profile"""
        text_parts = []
        
        # Personal Info
        if 'personalInfo' in profile:
            personal = profile['personalInfo']
            text_parts.extend([
                personal.get('name', ''),
                personal.get('bio', ''),
                personal.get('location', '')
            ])

        # Skills - with weighted importance
        for skill_category in profile.get('skills', []):
            if isinstance(skill_category, dict):
                # Add category with double weight
                category = skill_category.get('category', '')
                text_parts.extend([category] * 2)  # Repeat category twice
                
                # Add skills with triple weight
                skills = skill_category.get('items', [])
                for skill in skills:
                    # Add each skill three times to increase its weight
                    text_parts.extend([skill] * 3)
                    
                    # Add combinations of category and skill for better context matching
                    if category and skill:
                        text_parts.append(f"{category} {skill}")

        # Add skill combinations for better matching
        if profile.get('skills', []):
            all_skills = []
            for category in profile['skills']:
                if isinstance(category, dict):
                    all_skills.extend(category.get('items', []))
            
            # Add pairs of related skills to improve contextual matching
            if len(all_skills) > 1:
                for i in range(len(all_skills)):
                    for j in range(i + 1, len(all_skills)):
                        text_parts.append(f"{all_skills[i]} {all_skills[j]}")

        # Education
        for edu in profile.get('education', []):
            if isinstance(edu, dict):
                text_parts.extend([
                    edu.get('institution', ''),
                    edu.get('degree', ''),
                    edu.get('field', '')
                ])

        # Projects
        for project in profile.get('projects', []):
            if isinstance(project, dict):
                text_parts.extend([
                    project.get('title', ''),
                    project.get('description', '')
                ])

        # Combine and preprocess
        combined_text = ' '.join(filter(None, text_parts))
        return self.preprocess_text(combined_text)

    def index_profiles(self, profiles):
        """Build TF-IDF index from profiles"""
        self.logger.info("Starting profile indexing...")
        self.profiles = profiles
        
        # Prepare text documents
        documents = []
        for profile in profiles:
            doc_text = self.prepare_profile_text(profile)
            documents.append(doc_text)
        
        # Create TF-IDF matrix
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                min_df=2,
                max_df=0.95,
                max_features=5000
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            self.logger.info(f"Successfully indexed {len(profiles)} profiles")
            self.logger.info(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        except Exception as e:
            self.logger.error(f"Error creating TF-IDF matrix: {str(e)}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

    def search(self, query, max_results=10):
        """Search profiles using TF-IDF and fallback to keyword matching"""
        self.logger.info(f"\n=== Search Query: {query} ===")
        
        if not query:
            return []

        # Try TF-IDF search first
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            try:
                self.logger.info("Attempting TF-IDF search...")
                query_vector = self.tfidf_vectorizer.transform([self.preprocess_text(query)])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                
                # Get top matches
                top_indices = similarities.argsort()[-max_results:][::-1]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0:
                        profile = self.profiles[idx]
                        results.append({
                            'profile': profile,
                            'score': float(similarities[idx]),
                            'method': 'TF-IDF'
                        })
                
                if results:
                    self.logger.info(f"Found {len(results)} results using TF-IDF")
                    return results
                
            except Exception as e:
                self.logger.error(f"TF-IDF search failed: {str(e)}")

        # Fallback to keyword matching
        self.logger.info("Falling back to keyword matching...")
        query_terms = set(self.preprocess_text(query).split())
        
        keyword_results = []
        for profile in self.profiles:
            profile_text = self.prepare_profile_text(profile)
            profile_terms = set(profile_text.split())
            
            # Calculate simple matching score
            matching_terms = query_terms & profile_terms
            if matching_terms:
                score = len(matching_terms) / len(query_terms)
                keyword_results.append({
                    'profile': profile,
                    'score': score,
                    'method': 'keyword'
                })

        # Sort by score and return top results
        keyword_results.sort(key=lambda x: x['score'], reverse=True)
        results = keyword_results[:max_results]
        
        self.logger.info(f"Found {len(results)} results using keyword matching")
        return results
    
class ProfilePreFilter:
    def __init__(self):
        self.skill_levels = ['Beginner', 'Intermediate', 'Advanced']
        self.min_threshold = 0.2  # Minimum matching threshold
        self.optimal_threshold = 0.5  # Optimal matching threshold
        self.min_filtered_profiles = 5  # Minimum profiles after filtering

    def calculate_basic_match(self, user_skills, profile_skills):
        """Calculate basic skill match ratio with fallback for limited data"""
        if not user_skills:
            return 1.0  # Accept all profiles if user has no skills
        return len(user_skills.intersection(profile_skills)) / len(user_skills)

    def get_profile_skills(self, profile):
        """Extract skills with fallback for incomplete profiles"""
        skills = set()
        try:
            for skill_cat in profile.get('skills', []):
                skills.update(skill_cat.get('items', []))
        except Exception as e:
            print(f"Error extracting skills: {e}")
        return skills

    def filter_profiles(self, current_user, all_profiles, classifier):
        """Dynamic pre-filtering with debug information"""
        # Get user skills and level
        user_skills = self.get_profile_skills(current_user)
        print(f"User skills found: {len(user_skills)}")
        
        try:
            user_level = classifier.predict_skill_level(current_user)
        except Exception as e:
            print(f"Could not determine user level: {e}")
            user_level = 'Intermediate'  # Fallback
        
        filtered_profiles = []
        threshold = self.optimal_threshold
        
        while True:
            filtered_profiles = []
            print(f"\nTrying threshold: {threshold}")
            
            for profile in all_profiles:
                # Safe email access
                current_user_email = current_user.get('personalInfo', {}).get('email')
                profile_email = profile.get('personalInfo', {}).get('email')
                if profile_email and current_user_email and profile_email != current_user_email:
                    profile_skills = self.get_profile_skills(profile)
                    
                    try:
                        profile_level = classifier.predict_skill_level(profile)
                    except Exception as e:
                        print(f"Using fallback level for profile: {e}")
                        profile_level = 'Intermediate'
                    
                    # Calculate match score
                    match_ratio = self.calculate_basic_match(user_skills, profile_skills)
                    
                    # Check level compatibility
                    level_diff = abs(self.skill_levels.index(user_level) - 
                                self.skill_levels.index(profile_level))
                    
                    if match_ratio >= threshold and level_diff <= 1:
                        filtered_profiles.append({
                            'profile': profile,
                            'match_ratio': match_ratio,
                            'level': profile_level
                        })
            
            print(f"Filtered profiles at threshold {threshold}: {len(filtered_profiles)}")
            
            # Check if we have enough profiles
            if len(filtered_profiles) >= self.min_filtered_profiles or threshold <= self.min_threshold:
                break
            
            # Reduce threshold and try again
            threshold -= 0.1
            threshold = max(threshold, self.min_threshold)
        
        # Sort by match ratio
        filtered_profiles.sort(key=lambda x: x['match_ratio'], reverse=True)
        
        print(f"\nFinal filtered profiles: {len(filtered_profiles)}")        
        return [item['profile'] for item in filtered_profiles]

class SkillsClassifier:
    def __init__(self):
        self._debug_shown = False
        # Initialize with class weights to prevent bias towards Advanced
        class_weights = {
            0: 1.5,  # Beginner
            1: 1.5,  # Intermediate
            2: 0.3   # Advanced - reduce weight
        }
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weights,
            max_depth=5,  # Limit tree depth to prevent overfitting
            min_samples_leaf=5,
            random_state=42
        )
        self.mlb = MultiLabelBinarizer()
        self.label_encoder = LabelEncoder()
        self.skill_categories = ['Beginner', 'Intermediate', 'Advanced']
        self.is_fitted = False
        self.mlb_fitted = False
        # Pre-fit the label encoder with known categories
        self.label_encoder.fit(self.skill_categories)
        
    def prepare_features(self, profile):
        # 1. Get all skills
        all_skills = []
        for skill_category in profile.get('skills', []):
            all_skills.extend([skill.lower() for skill in skill_category.get('items', [])])
        
        # 2. Transform skills to binary format
        if not self.mlb_fitted:
            all_possible_skills = set(all_skills)
            self.mlb.fit([list(all_possible_skills)])
            self.mlb_fitted = True
        
        # 3. Transform skills using fitted MLBinarizer
        skill_vector = self.mlb.transform([all_skills])
        
        # 4. Get numeric features with actual experience duration
        exp_months = 0
        for exp in profile.get('experience', []):
            try:
                start_date = datetime.strptime(exp.get('startDate', ''), '%Y-%m')
                end_date = datetime.strptime(exp.get('endDate', ''), '%Y-%m')
                exp_months += (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            except:
                exp_months += 0
        
        exp_years = exp_months / 12  # Convert to years
        project_count = len(profile.get('projects', []))
        numeric_features = np.array([[exp_years, project_count]])
        
        # 5. Combine features using hstack
        features = np.hstack([skill_vector, numeric_features])            
        return features

    def determine_skill_level(self, profile):
        # Calculate actual experience in years
        exp_months = 0
        for exp in profile.get('experience', []):
            try:
                start_date = datetime.strptime(exp.get('startDate', ''), '%Y-%m')
                end_date = datetime.strptime(exp.get('endDate', ''), '%Y-%m')
                exp_months += (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            except:
                exp_months += 0
        
        exp_years = exp_months / 12
        project_count = len(profile.get('projects', []))
        skill_count = sum(len(cat.get('items', [])) for cat in profile.get('skills', []))
        
        # Only print debug info for current user's profile
        if (profile.get('personalInfo', {}).get('email') == session.get('user') and 
            not self._debug_shown):
            self._debug_shown = True  # Set flag to True after showing debug
            print(f"User's Experience years: {exp_years:.1f}")
            print(f"User's Project count: {project_count}")
            print(f"User's Skill count: {skill_count}")
        
        if exp_years >= 3 or (project_count >= 5 and skill_count >= 15):
            return 'Advanced'
        elif exp_years >= 0.5 or (project_count >= 1 and skill_count >= 5):  # Changed from 1 to 0.5 years
            return 'Intermediate'
        else:
            return 'Beginner'

    def train(self, profiles):
        try:
            if not profiles:
                print("No profiles available for training")
                return False
                
            # Collect all possible skills (case-normalized)
            all_possible_skills = set()
            X = []
            y = []
            
            # First pass: collect all skills
            for profile in profiles:
                for skill_category in profile.get('skills', []):
                    skills = [skill.lower() for skill in skill_category.get('items', [])]
                    all_possible_skills.update(skills)
            
            # Fit MLBinarizer once with all possible skills
            self.mlb.fit([list(all_possible_skills)])
            self.mlb_fitted = True
            
            # Second pass: prepare features and encoded labels
            for profile in profiles:
                try:
                    features = self.prepare_features(profile)
                    skill_level = self.determine_skill_level(profile)
                    # Convert skill level to encoded form
                    encoded_level = self.label_encoder.transform([skill_level])[0]
                    X.append(features)
                    y.append(encoded_level)
                except Exception as e:
                    print(f"Error processing profile: {e}")
                    continue
            
            if not X or not y:
                return False
            
            X = np.vstack(X)
            y = np.array(y)
            self.classifier.fit(X, y)
            self.is_fitted = True
            print(f"Trained on {len(X)} profiles with {X.shape[1]} features")
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False

    def predict_skill_level(self, profile):
        try:
            features = self.prepare_features(profile)
            
            # Get deterministic classification first
            deterministic_level = self.determine_skill_level(profile)
            if not self.is_fitted:
                print("Classifier not fitted, using deterministic classification")
                return deterministic_level
            # Get ML prediction
            prediction_proba = self.classifier.predict_proba(features)[0]
            prediction = np.argmax(prediction_proba)
            ml_level = self.skill_categories[prediction]
            
            # Only show debug info for current user
            if (profile.get('personalInfo', {}).get('email') == session.get('user') and 
                not hasattr(self, '_debug_shown')):
                self._debug_shown = True  # Mark that debug has been shown
                print(f"\nClassification probabilities:")
                for idx, prob in enumerate(prediction_proba):
                    print(f"{self.skill_categories[idx]}: {prob:.2f}")
                print(f"ML prediction: {ml_level}")
                print(f"Deterministic prediction: {deterministic_level}")
            
            # If deterministic says Intermediate/Beginner but ML says Advanced,
            # trust deterministic
            if (deterministic_level in ['Beginner', 'Intermediate'] and 
                ml_level == 'Advanced'):
                return deterministic_level
            
            # If predictions differ by more than one level, use deterministic
            det_idx = self.skill_categories.index(deterministic_level)
            ml_idx = self.skill_categories.index(ml_level)
            
            if abs(det_idx - ml_idx) > 1:
                return deterministic_level
                
            return ml_level
            
        except Exception as e:
            print(f"ML prediction failed, using fallback: {e}")
            return self.determine_skill_level(profile)

class EnhancedRecommender:
    def __init__(self, min_support=0.001, min_confidence=0.01):  # Lowered thresholds
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.skills_classifier = SkillsClassifier()
        self.pre_filter = ProfilePreFilter()

        # Initialize classifier
        print("\nInitializing Skills Classifier...")
        try:
            self.skills_classifier.mlb = MultiLabelBinarizer()
            self.skills_classifier.mlb_fitted = True
            print("Skills Classifier initialized")
        except Exception as e:
            print(f"Error initializing classifier: {e}")

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
        return " ".join(tokens)

    def prepare_profile_data(self, profile):
        items = set()
        if not isinstance(profile, dict):
            return items
            
        # Process skills (limit to 8 per category)
        for skill_category in profile.get('skills', [])[:5]:  # Max 5 categories
            for skill in skill_category.get('items', [])[:8]:  # Max 8 skills per category
                processed_skill = self.preprocess_text(skill)
                if processed_skill:  # Only add if not empty after processing
                    items.add(f"SKILL_{processed_skill}")
        
        # Process bio/interests (limit to 10 terms)
        if 'personalInfo' in profile and 'bio' in profile['personalInfo']:
            bio_text = self.preprocess_text(profile['personalInfo']['bio'])
            for term in bio_text.split()[:10]:  # Max 10 terms
                if len(term) > 2:  # Reduced from 3 to get more terms
                    items.add(f"INTEREST_{term}")
        
        # Process projects (limit to 3)
        for project in profile.get('projects', [])[:3]:
            if 'title' in project:
                processed_title = self.preprocess_text(project['title'])
                if processed_title:
                    items.add(f"PROJECT_{processed_title}")
                
        return items
    
    def validate_profile(self, profile):
        required_fields = ['skills', 'personalInfo']
        if not all(field in profile for field in required_fields):
            return False
        return True

    def initialize_classifier(self, profiles):
        print("\nInitializing ML Classifier...")
        try:
            success = self.skills_classifier.train(profiles)
            if success:
                print("ML Classifier initialized successfully")
            else:
                print("ML Classifier initialization failed, using fallback")
            return success
        except Exception as e:
            print(f"Error initializing classifier: {e}")
            return False

    def calculate_similarity(self, profile1_items, profile2_items):
        try:
            intersection = len(profile1_items & profile2_items)
            union = len(profile1_items | profile2_items)
            return intersection / union if union > 0 else 0
        except Exception as e:
            print(f"Similarity calculation error: {str(e)}")
            return 0

    def get_enhanced_recommendations(self, current_user_profile, all_profiles):
        try:
            # Initialize classifier if not already done
            if not self.skills_classifier.is_fitted:
                valid_profiles = [
                    profile for profile in all_profiles 
                    if profile.get('personalInfo', {}).get('email') and 
                    profile.get('personalInfo', {}).get('name') and
                    profile.get('skills')
                ]
                self.initialize_classifier(valid_profiles)

            # Stage 1 Debug
            print("\nSTAGE 1: PRE-FILTERING")
            print(f"Total profiles before filtering: {len(all_profiles)}")
            
            # Stage 1: Pre-filtering
            filtered_profiles = self.pre_filter.filter_profiles(
                current_user_profile, 
                all_profiles,
                self.skills_classifier
            )
            # Get user's skill level using ML classifier
            user_skill_level = self.skills_classifier.predict_skill_level(current_user_profile)
            print(f"User skill level (RF Classification): {user_skill_level}")  
            
            # Validate filtered profiles
            valid_filtered_profiles = [
                profile for profile in filtered_profiles 
                if self.validate_profile(profile)
            ]

            if len(valid_filtered_profiles) < len(filtered_profiles):
                self.logger.warning(
                    f"Filtered out {len(filtered_profiles) - len(valid_filtered_profiles)} invalid profiles"
                )

            print(f"\nProceeding to Stage 2 with {len(valid_filtered_profiles)} valid profiles")
            
            # Stage 2: Your existing recommendation logic
            print("\nSTAGE 2: DETAILED MATCHING")
            print("Using original 60:40 Content-based : Market Basket model")
            print("\nContent-Based Filtering:")

            current_user_items = self.prepare_profile_data(current_user_profile)
            
            # Get user's skill level using ML classifier
            user_skill_level = self.skills_classifier.predict_skill_level(current_user_profile)
            
            # Prepare data for both content-based and market basket analysis
            transactions = []
            profile_items = {}
            content_features = []
            
            # Process profiles in batches for memory efficiency
            batch_size = 50
            for i in range(0, len(filtered_profiles), batch_size):
                batch = valid_filtered_profiles[i:i + batch_size]
                for profile in batch:
                    items = self.prepare_profile_data(profile)
                    if items:
                        transactions.append(items)
                        profile_items[profile['personalInfo']['email']] = items
                        # Add to content features
                        content_features.append({
                            'email': profile['personalInfo']['email'],
                            'features': items,
                            'skill_level': self.skills_classifier.predict_skill_level(profile)  # Add skill level
                        })
            
            # Content-based filtering with skill level consideration
            content_based_scores = []
            for feature in content_features:
                if feature['email'] != current_user_profile['personalInfo']['email']:
                    # Calculate basic similarity
                    similarity = self.calculate_similarity(
                        current_user_items, 
                        feature['features']
                    )
                    if similarity > 0:
                        content_based_scores.append(similarity)  # Pure similarity score
            if content_based_scores:
                strong_matches = len([s for s in content_based_scores if s > 0.35])
                print(f"Found {strong_matches} strong profile matches")
            else:
                print("No significant content-based matches found")

            print("\nStarting Market Basket Analysis Phase")
            # Market basket analysis using existing apriori implementation
            print(f"Number of transactions: {len(transactions)}")
            print(f"Number of items per transaction: {len(current_user_items)}")
            
            all_unique_items = set()
            for trans in transactions:
                all_unique_items.update(trans)
            print("\nCreating Market Basket DataFrame")
            print(f"Number of unique items: {len(all_unique_items)}")
            df = pd.DataFrame(0, index=range(len(transactions)), columns=list(all_unique_items))
            for idx, trans in enumerate(transactions):
                df.loc[idx, [item for item in trans if item in all_unique_items]] = 1

            print(f"\nDataFrame shape: {df.shape}")
            print("DataFrame sample (first 5 columns):")
            print("Non-zero elements in first row:", df.iloc[0].sum())
            print(df.iloc[:5, :5])

            frequent_itemsets = apriori(
                df, 
                min_support=self.min_support,
                use_colnames=True,
                max_len=3
            )
            # Initialize recommended_items
            recommended_items = set()
            print(f"Frequent itemsets found: {len(frequent_itemsets) if not frequent_itemsets.empty else 0}")
            
            if not frequent_itemsets.empty:
                print("Sample itemsets:")
                print(frequent_itemsets.head())
                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=self.min_confidence
                )
                print(f"Number of rules generated: {len(rules)}")
                
                if not rules.empty:
                    print("\nProcessing rules for recommendations...")
                    for _, rule in rules.iterrows():
                        antecedents = set(rule['antecedents'])
                        if antecedents.issubset(current_user_items):
                            recommended_items.update(set(rule['consequents']))

                    print(f"Number of recommended items: {len(recommended_items)}")
            else:
                print("\nNo frequent itemsets found - check min_support threshold")
                # Try with lower support
                frequent_itemsets = apriori(
                    df, 
                    min_support=self.min_support/2,  # Try with half the support
                    use_colnames=True,
                    max_len=3
                )
                print(f"With lower support: {len(frequent_itemsets) if not frequent_itemsets.empty else 0} itemsets found")
            
            # Debug print recommended items
            print(f"Final recommended items: {len(recommended_items)}")

            # Get final recommendations combining all approaches
            final_recommendations = []
            for profile in valid_filtered_profiles:
                if profile['personalInfo']['email'] != current_user_profile['personalInfo']['email']:
                    profile_items = self.prepare_profile_data(profile)

                    # Calculate scores
                    content_score = self.calculate_similarity(current_user_items, profile_items)
                    market_basket_score = len(profile_items & recommended_items) / len(recommended_items) if recommended_items else 0
                    
                    # Combined score (weighted average)
                    final_score = (0.6 * content_score + 
                                    0.4 * market_basket_score )
                    
                    # Get common skills
                    user_skills = set()
                    profile_skills = set()
                    for skill_cat in current_user_profile.get('skills', []):
                        user_skills.update(skill_cat.get('items', []))
                    for skill_cat in profile.get('skills', []):
                        profile_skills.update(skill_cat.get('items', []))
                    common_skills = user_skills & profile_skills
                    
                    if final_score > 0.1:  # Threshold for meaningful recommendations
                        final_recommendations.append((profile, final_score, content_score, market_basket_score, list(common_skills)))
      
            # Sort and return top recommendations
            final_recommendations.sort(key=lambda x: x[1], reverse=True)

            # Print final recommendations in desired format
            print("\nFINAL RECOMMENDATIONS:")
            print("----------------------------------------")
            for profile, final_score, content_score, mba_score, common_skills in final_recommendations[:5]:
                print(f"\n{profile['personalInfo']['name']}")
                print(f"Similarity Score: {content_score:.2f}")
                print(f"Market Basket Score: {mba_score:.2f}")
                print(f"Combined Score: {final_score:.2f}")
                print(f"Common Skills: {', '.join(common_skills)}")
                print(f"Skill Level: {self.skills_classifier.predict_skill_level(profile)}")
                print("----------------------------------------")

            # Return the recommendations
            return {
                'recommended_items': list(recommended_items)[:10],
                'similar_profiles': [
                    {
                        'personalInfo': profile['personalInfo'],
                        'skills': profile.get('skills', []),
                        'education': profile.get('education', []),
                        'socialLinks': profile.get('socialLinks', {}),
                        'score': score,
                        'content_score': content_score,
                        'mba_score': mba_score
                    } for profile, score, content_score, mba_score, _ in final_recommendations[:5]
                ],
                'user_skill_level': user_skill_level,
                'method_used': 'combined'
            }
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return None

def ensure_default_profile_image():
    default_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'default.jpg')
    if not os.path.exists(default_image_path):
        import shutil
        source_default = os.path.join('static', 'img', 'default.jpg')  # Path to the source default image
        if os.path.exists(source_default):
            shutil.copy(source_default, default_image_path)
        else:
            # Create a blank default image if the source is missing
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='gray')
            img.save(default_image_path)

# Call the function to ensure the default image exists
ensure_default_profile_image()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper function for file uploads
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Default route
@app.route("/")
def home():
    if 'user' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

# Index route
@app.route("/index")
@login_required
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = mongo.db.users.find_one({"email": session["user"]})
    return render_template("index.html", user=user)

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user' in session:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = mongo.db.users.find_one({"email": email})
        if user and bcrypt.check_password_hash(user["password"], password):
            session["user"] = user["email"]
            flash(f'Welcome back, {user["name"]}!', 'success')
            return redirect(url_for("index"))
        else:
            flash("Incorrect email or password. Please try again.", "danger")
            return redirect(url_for("login"))
    
    return render_template("login.html")

# Registration route
@app.route("/register", methods=["GET", "POST"])
def register():
    if 'user' in session:
        return redirect(url_for('index'))
        
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        year_of_study = request.form["year_of_study"]
        degree = request.form["degree"]
        branch = request.form["branch"]
        area_of_interest = request.form["area_of_interest"]
        secret_word = request.form["secret_word"]

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("register"))

        if mongo.db.users.find_one({"email": email}):
            flash("Email already registered! Please login instead.", "danger")
            return redirect(url_for("register"))

        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        
        user_data = {
            "name": name,
            "email": email,
            "password": hashed_password,
            "year_of_study": year_of_study,
            "degree": degree,
            "branch": branch,
            "area_of_interest": area_of_interest,
            "secret_word": secret_word
        }
        
        mongo.db.users.insert_one(user_data)
        flash(f"Account created successfully! Welcome {name}!", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# Forgot Password route
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if 'user' in session:
        return redirect(url_for('index'))
        
    if request.method == "POST":
        email = request.form["email"]
        secret_word = request.form["secret_word"]
        new_password = request.form["new_password"]

        user = mongo.db.users.find_one({"email": email})

        if user and user["secret_word"] == secret_word:
            hashed_password = bcrypt.generate_password_hash(new_password).decode("utf-8")
            mongo.db.users.update_one(
                {"email": email}, 
                {"$set": {"password": hashed_password}}
            )
            flash("Password reset successful! You can now login.", "success")
            return redirect(url_for("login"))
        else:
            flash("Incorrect email or secret word!", "danger")

    return render_template("forgot_password.html")

# Logout route
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))

# search index
@app.route("/api/update-search-index", methods=['POST'])
@login_required
def update_search_index():
    try:
        profiles = list(mongo.db.profiles.find({}, {"_id": 0}))
        search_engine.index_profiles(profiles)
        return jsonify({
            "success": True,
            "message": f"Search index updated with {len(profiles)} profiles"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Chatbot route
@app.route("/chatbot")
@login_required
def chatbot():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = mongo.db.users.find_one({"email": session["user"]})
    return render_template("chatbot.html", user=user)

# API endpoint for the chatbot
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        user_message = data.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"Processing message: {user_message}")
        
        # Get relevant data from external APIs based on user query
        context_data = get_relevant_data(user_message)
        
        # Get AI response using Mistral
        response_text = get_ai_response(user_message, context_data)
        
        return jsonify({
            "response": response_text
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/market-basket', methods=['POST'])
@login_required
def debug_market_basket():
    try:
        data = request.json
        current_user_profile = data.get('current_user_profile')
        all_profiles = data.get('all_profiles')
        
        recommender = EnhancedRecommender()
        results = recommender.get_enhanced_recommendations(current_user_profile, all_profiles)
        
        return jsonify({
            'success': True,
            'results': results,
            'debug_info': {
                'transactions_created': True,
                'market_basket_completed': True
            }
        })
    except Exception as e:
        print(f"Market Basket Debug Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Account Settings route
@app.route("/account_settings", methods=["GET", "POST"])
@login_required
def account_settings():
    user = mongo.db.users.find_one({"email": session["user"]})
    
    if request.method == "POST":
        action = request.form.get("action")
        
        if action == "update_username":
            new_email = request.form.get("new_email")
            password = request.form.get("current_password")
            
            if bcrypt.check_password_hash(user["password"], password):
                if mongo.db.users.find_one({"email": new_email}) and new_email != user["email"]:
                    flash("Email already exists!", "danger")
                else:
                    mongo.db.users.update_one(
                        {"email": session["user"]},
                        {"$set": {"email": new_email}}
                    )
                    session["user"] = new_email
                    flash("Email updated successfully!", "success")
                    return redirect(url_for("account_settings"))
            else:
                flash("Current password is incorrect!", "danger")
                
        elif action == "update_password":
            current_password = request.form.get("current_password")
            new_password = request.form.get("new_password")
            confirm_password = request.form.get("confirm_password")
            
            if not bcrypt.check_password_hash(user["password"], current_password):
                flash("Current password is incorrect!", "danger")
            elif new_password != confirm_password:
                flash("New passwords do not match!", "danger")
            else:
                hashed_password = bcrypt.generate_password_hash(new_password).decode("utf-8")
                mongo.db.users.update_one(
                    {"email": session["user"]},
                    {"$set": {"password": hashed_password}}
                )
                flash("Password updated successfully!", "success")
                return redirect(url_for("account_settings"))
                
    return render_template("account_settings.html", user=user)

# Opportunities route
@app.route('/opportunities')
@login_required
def opportunities():
    return render_template('opportunities.html')

# Get posts route
@app.route('/posts', methods=['GET'])
@login_required
def get_posts():
    posts = list(mongo.db.posts.find({}, {'_id': False}))
    return jsonify(posts)

# Add post route
@app.route('/add_post', methods=['POST'])
@login_required
def add_post():
    try:
        data = request.json
        post = {
            'name': data['name'],
            'email': data['email'],
            'title': data['title'],
            'description': data['description'],
            'form_link': data.get('form_link', ''),  # Optional form link
            'likes': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        mongo.db.posts.insert_one(post)
        return jsonify({"message": "Post added successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Like post route
@app.route('/like_post', methods=['POST'])
@login_required
def like_post():
    try:
        data = request.json
        result = mongo.db.posts.update_one(
            {'timestamp': data['timestamp']},
            {'$inc': {'likes': 1}}
        )
        if result.matched_count == 0:
            return jsonify({"error": "Post not found"}), 404
        return jsonify({"message": "Like added successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html")

# Profile API routes
@app.route('/api/profile', methods=['GET'])
def get_profile():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # First try to find in the profiles collection
    profile = mongo.db.profiles.find_one({'personalInfo.email': session["user"]})
    
    if not profile:
        # Get user data from users collection
        user_data = mongo.db.users.find_one({'email': session["user"]})
        user_name = user_data.get('name', '') if user_data else ''
        
        # Create new profile with user's name from users collection
        profile = {
            'personalInfo': {
                'name': user_name,  # Now dynamically set from users collection
                'email': session["user"],
                'phone': '',
                'location': '',
                'bio': '',
                'profileImage': 'default.jpg'
            },
            'education': [],
            'skills': [],
            'experience': [],
            'projects': [],
            'extracurriculars': [],
            'socialLinks': {
                'linkedin': '',
                'github': '',
                'portfolio': ''
            }
        }
        # Insert the new profile
        mongo.db.profiles.insert_one(profile)
    if 'personalInfo' not in profile or 'profileImage' not in profile['personalInfo']:
        profile['personalInfo']['profileImage'] = 'default.jpg'
    
    return json.loads(json_util.dumps(profile))

@app.route('/api/profile/personal', methods=['POST'])
@login_required
def update_personal_info():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        # Ensure email from session is included in personal info
        data['email'] = session["user"]  # Add this line
        
        # Update only the personalInfo section
        result = mongo.db.profiles.update_one(
            {'personalInfo.email': session["user"]},
            {'$set': {'personalInfo': data}},
            upsert=True
        )
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating personal info: {str(e)}")
        return jsonify({'error': 'Failed to update personal information'}), 500


@app.route('/api/profile', methods=['POST'])
def update_profile():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    data['lastUpdated'] = datetime.utcnow()
    
    # Update using the correct collection and query
    result = mongo.db.profiles.update_one(
        {'personalInfo.email': session["user"]},
        {'$set': data},
        upsert=True
    )
    
    return jsonify({'success': True})

@app.route('/api/upload-image', methods=['POST'])
def upload_profile_image():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Create unique filename using email
        filename = secure_filename(f"{session['user']}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Update profile with new image filename
        mongo.db.profiles.update_one(
            {'personalInfo.email': session["user"]},
            {'$set': {'personalInfo.profileImage': filename}}
        )
        
        return jsonify({'filename': filename})
    
    return jsonify({'error': 'Invalid file type'}), 400

# Section-specific update routes
@app.route("/api/profile/education", methods=['POST'])
@login_required
def update_education():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400

        current_user_email = session.get('user')
        
        # Create education entry
        education_entry = {
            'degree': data.get('degree'),
            'field': data.get('field'),
            'startYear': data.get('startYear'),
            'endYear': data.get('endYear'),
            'grade': data.get('grade', ''),
            'institution': data.get('institution', ''),
            'activities': data.get('activities', '')
        }

        # Debug logging
        print("\n=== Education Update Debug ===")
        print(f"User Email: {current_user_email}")
        print(f"New Education Entry: {education_entry}")

        # Get current profile
        current_profile = profiles_collection.find_one({'personalInfo.email': current_user_email})
        
        # Check for duplicate entry
        if current_profile and 'education' in current_profile:
            for entry in current_profile['education']:
                if (entry.get('degree') == education_entry['degree'] and 
                    entry.get('field') == education_entry['field']):
                    return jsonify({
                        "success": False,
                        "message": "An education entry with this degree and field already exists"
                    }), 400

        # If no duplicate exists, update the profile
        result = profiles_collection.update_one(
            {'personalInfo.email': current_user_email},
            {'$push': {'education': education_entry}}
        )

        if result.modified_count == 0:
            return jsonify({
                "success": False,
                "message": "No profile was updated"
            }), 400

        return jsonify({
            "success": True,
            "message": "Education updated successfully",
            "education": education_entry
        })

    except Exception as e:
        print(f"Error updating education: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error updating education: {str(e)}"
        }), 500


@app.route('/api/profile/skills', methods=['POST'])
@login_required
def update_skills():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        result = mongo.db.profiles.update_one(
            {'personalInfo.email': session["user"]},
            {'$set': {'skills': data}},
            upsert=True
        )
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating skills: {str(e)}")
        return jsonify({'error': 'Failed to update skills'}), 500

@app.route('/api/profile/experience', methods=['POST'])
@login_required
def update_experience():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        result = mongo.db.profiles.update_one(
            {'personalInfo.email': session["user"]},
            {'$set': {'experience': data}},
            upsert=True
        )
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating experience: {str(e)}")
        return jsonify({'error': 'Failed to update experience'}), 500

@app.route('/api/profile/projects', methods=['POST'])
@login_required
def update_projects():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        print(f"Received projects data: {data}")
        result = mongo.db.profiles.update_one(
            {'personalInfo.email': session["user"]},
            {'$set': {'projects': data}},
            upsert=True
        )
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating projects: {str(e)}")
        return jsonify({'error': 'Failed to update projects'}), 500


#updating extracurriculars
@app.route('/api/profile/extracurriculars', methods=['POST'])
@login_required
def update_extracurriculars():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        result = mongo.db.profiles.update_one(
            {'personalInfo.email': session["user"]},
            {'$set': {'extracurriculars': data}},
            upsert=True
        )
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating extracurriculars: {str(e)}")
        return jsonify({'error': 'Failed to update extracurriculars'}), 500

@app.route('/api/profile/social', methods=['POST'])
@login_required
def update_social():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        result = mongo.db.profiles.update_one(
            {'personalInfo.email': session["user"]},
            {'$set': {'socialLinks': data}},
            upsert=True
        )
        return jsonify({'success': True})
    
    except Exception as e:
        print(f"Error updating social links: {str(e)}")
        return jsonify({'error': 'Failed to update social links'}), 500    

# Update your existing recommendations endpoint
@app.route('/api/recommendations')
@login_required
def get_recommendations():
    try:
        current_user_email = session.get('user')
        print("\n=== TWO-STAGE RECOMMENDATION PROCESS ===")
        print(f"\nProcessing for user: {current_user_email}")

        if not current_user_email:
            print("❌ Error: Current user email not found")
            return jsonify({
                'success': False,
                'recommendations': [],
                'message': 'User not authenticated'
            })

        # Get current user profile
        current_user = profiles_collection.find_one({'personalInfo.email': current_user_email})
        if not current_user:
            print("❌ Error: Current user profile not found")
            return jsonify({
                'success': False,
                'recommendations': [],
                'message': 'User profile not found'
            })
        
        print("\nGetting all profiles from database")
        # Get all profiles and ensure they have required fields
        all_profiles = list(profiles_collection.find({}, {'_id': 0}))
        valid_profiles = [
            profile for profile in all_profiles 
            if profile.get('personalInfo', {}).get('email') and 
            profile.get('personalInfo', {}).get('name') and
            profile.get('skills')
        ]
        
        print(f"\nFound {len(valid_profiles)} valid profiles")
        recommender = EnhancedRecommender(min_support=0.0005, min_confidence=0.005)
        recommendations = recommender.get_enhanced_recommendations(current_user, all_profiles)

        if recommendations and recommendations.get('similar_profiles'):
            formatted_recommendations = []
            for rec in recommendations['similar_profiles']:
                formatted_recommendations.append({
                    'personalInfo': rec['personalInfo'],
                    'skills': rec.get('skills', []),
                    'education': rec.get('education', []),
                    'socialLinks': rec.get('socialLinks', {}),
                    'similarity_score': round(rec['score'] * 100, 2),
                    'content_score': round(rec.get('content_score', 0) * 100, 2),
                    'mba_score': round(rec.get('mba_score', 0) * 100, 2)
                })
                                
            return jsonify({
                'success': True,
                'recommendations': formatted_recommendations,
                'total_matches': len(formatted_recommendations),
                'method': 'primary',
                'user_skill_level': recommendations['user_skill_level']
            })

        # If EnhancedRecommender fails, use fallback
        print("\n⚠️ Enhanced recommendations failed, trying fallback...")
        
        # Extract user skills for fallback
        user_skills_set = set()
        for skill_cat in current_user.get('skills', []):
            skills = [skill.lower() for skill in skill_cat.get('items', [])]
            user_skills_set.update(skills)
        
        try:
            print(f"\nTrying skill matching fallback")
            user_skills_list = [skill.lower() for skill in user_skills_set]
            print(f"User skills for matching: {user_skills_list}")
            
            # Simple fallback - find profiles with at least one matching skill
            fallback_profiles = list(profiles_collection.find(
                {
                    'personalInfo.email': {'$ne': current_user_email},
                    'skills': {
                        '$elemMatch': {
                            'items': {
                                '$elemMatch': {
                                    '$regex': f'^(?:{")|(?:".join(user_skills_list)})$',
                                    '$options': 'i'  # case insensitive
                                }
                            }
                        }
                    }
                },
                {'_id': 0}
            ).limit(5))

            if fallback_profiles:
                print(f"\nFound {len(fallback_profiles)} profiles with matching skills")
                # Calculate basic similarity for fallback profiles
                for profile in fallback_profiles:
                    profile_skills = set()
                    for skill_cat in profile.get('skills', []):
                        skills = [skill.lower() for skill in skill_cat.get('items', [])]
                        profile_skills.update(skills)
                           
                    common_skills = user_skills_set.intersection(profile_skills)
                    
                    # Store original case skills for display
                    original_case_common_skills = []
                    for skill_cat in profile.get('skills', []):
                        for skill in skill_cat.get('items', []):
                            if skill.lower() in common_skills:
                                original_case_common_skills.append(skill)
                    
                    profile['common_skills'] = original_case_common_skills
                    profile['similarity_score'] = round(len(common_skills) / len(user_skills_set) * 100, 2)

                print("\nReturning skill-matched fallback profiles")
                return jsonify({
                    'success': True,
                    'recommendations': fallback_profiles,
                    'method': 'skill_match_fallback'
                })
            else:
                print("\nNo skill matches found, using random fallback")
                any_profiles = list(profiles_collection.find(
                    {'personalInfo.email': {'$ne': current_user_email}},
                    {'_id': 0}
                ).limit(5))
                
                for profile in any_profiles:
                    profile['similarity_score'] = 0
                    profile['common_skills'] = []

                return jsonify({
                    'success': True,
                    'recommendations': any_profiles,
                    'method': 'random_fallback'
                })

        except Exception as fallback_error:
            print(f"❌ Fallback error: {str(fallback_error)}")
            print(traceback.format_exc())
            return jsonify({
                'success': True,
                'recommendations': [],
                'method': 'failed',
                'error': 'Unable to fetch recommendations'
            })

    except Exception as e:
        print(f"❌ Error in recommendations: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': True,
            'recommendations': [],
            'error': 'Unable to fetch recommendations'
        })
  
#students/search route    

@app.route("/search")
def search():
    return render_template("search.html")

os.makedirs("static/data", exist_ok=True)

# Add this near your other route definitions in app.py

# Ensure the directory exists for storing JSON data
# os.makedirs("static/data", exist_ok=True)

# Initialize search engine at the top level of your app.py (after your imports)
search_engine = SearchEngine()

@app.route("/api/search")
def api_search():
    query = request.args.get("query", "").strip()
    
    try:
        # Initialize search engine if needed
        if not search_engine.profiles:
            profiles = list(profiles_collection.find({}, {"_id": 0}))
            search_engine.index_profiles(profiles)
            logger.info(f"Search engine initialized with {len(profiles)} profiles")

        # Perform search
        results = search_engine.search(query)
        
        # Log results
        logger.info("\n=== Search Results ===")
        for idx, result in enumerate(results, 1):
            profile = result['profile']
            logger.info(f"\nResult {idx}:")
            logger.info(f"Name: {profile.get('personalInfo', {}).get('name', 'Unknown')}")
            logger.info(f"Score: {result['score']:.4f}")
            logger.info(f"Method: {result['method']}")

        return jsonify({
            "success": True,
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "An error occurred while searching",
            "details": str(e)
        }), 500

    
# @app.route("/test/create-profiles-json")
# def test_create_profiles_json():
#     try:
#         # Fetch profiles
#         profiles = list(mongo.db.profiles.find({}, {"_id": 0}))
        
#         # Create path
#         json_path = os.path.join(current_app.static_folder, "data", "profiles.json")
        
#         # Ensure directory exists
#         os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
#         # Write file
#         with open(json_path, "w", encoding='utf-8') as f:
#             json.dump(profiles, f, indent=4, default=str)
            
#         return jsonify({
#             "success": True,
#             "message": f"File created at {json_path}",
#             "profiles_count": len(profiles)
#         })
#     except Exception as e:
#         return jsonify({
#             "error": str(e)
#         }), 500


@app.route("/test/create-profiles-json")
def test_create_profiles_json():
    try:
        # Fetch profiles
        profiles = list(mongo.db.profiles.find({}, {"_id": 0}))
        
        # Create path
        json_path = os.path.join(current_app.static_folder, "data", "profiles.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        # Write file
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(profiles, f, indent=4, default=str)
            
        return jsonify({
            "success": True,
            "message": f"File created at {json_path}",
            "profiles_count": len(profiles)
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

def update_profiles_json():
    with app.app_context():  # Ensure the function runs inside Flask's app context
        try:
            # Fetch profiles from MongoDB
            profiles = list(mongo.db.profiles.find({}, {"_id": 0}))

            # Create the path
            json_path = os.path.join(app.static_folder, "data", "profiles.json")

            # Ensure directory exists
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            # Write the JSON file
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(profiles, f, indent=4, default=str)

            print(f"✅ Profile JSON updated: {json_path} ({len(profiles)} profiles)")
        except Exception as e:
            print(f"❌ Error updating profile JSON: {e}")
    
#contact route
@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    origin = request.headers.get('Origin')
    if origin in ['http://127.0.0.1:5000', 'http://localhost:5000']:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# 💡 Run this function **inside** the Flask app context before starting the server
with app.app_context():
    update_profiles_json()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)