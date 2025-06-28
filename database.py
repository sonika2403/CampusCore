from flask_pymongo import PyMongo
from flask import current_app

mongo = PyMongo()

def get_db():
    return mongo.db

def init_db(app):
    mongo.init_app(app)
    return mongo

# Get collection references
def get_collections():
    db = get_db()
    users_collection = db.users
    profiles_collection = db.profiles
    return users_collection, profiles_collection

# Initialize collections with indexes
def init_collections():
    users_collection, profiles_collection = get_collections()
    
    # Create indexes
    profiles_collection.create_index('personalInfo.email', unique=True)
    
    return users_collection, profiles_collection

users_collection, profiles_collection = None, None

def init_app(app):
    global users_collection, profiles_collection
    mongo.init_app(app)
    users_collection, profiles_collection = init_collections()
