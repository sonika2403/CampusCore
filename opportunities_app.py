from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import json

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['students_db']
posts_collection = db['posts']

@app.route('/')
def home():
    return render_template('opportunities.html')

@app.route('/posts', methods=['GET'])
def get_posts():
    posts = list(posts_collection.find({}, {'_id': False}))
    return jsonify(posts)

@app.route('/add_post', methods=['POST'])
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
        posts_collection.insert_one(post)
        return jsonify({"message": "Post added successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/like_post', methods=['POST'])
def like_post():
    data = request.json
    posts_collection.update_one(
        {'timestamp': data['timestamp']},
        {'$inc': {'likes': 1}}
    )
    return jsonify({"message": "Like added successfully"})

if __name__ == '__main__':
    app.run(debug=True)
