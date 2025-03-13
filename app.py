from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo

app = Flask(__name__)
app.secret_key = "your_secret_key"

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/students_db"
mongo = PyMongo(app)
bcrypt = Bcrypt(app)

# -------------------- Home Page (Login) --------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = mongo.db.users.find_one({"email": email})
        if user and bcrypt.check_password_hash(user["password"], password):
            session["user"] = user["email"]
            return redirect(url_for("profile"))

        flash("Invalid email or password", "danger")
    
    return render_template("login.html")


# -------------------- Registration Page --------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")
        confirm_password = request.form["confirm_password"]
        year_of_study = request.form["year_of_study"]
        degree = request.form["degree"]
        branch = request.form["branch"]
        area_of_interest = request.form["area_of_interest"]
        secret_word = request.form["secret_word"]

        if request.form["password"] != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("register"))

        # Check if user already exists
        if mongo.db.users.find_one({"email": email}):
            flash("Email already registered!", "danger")
            return redirect(url_for("register"))

        user_data = {
            "name": name,
            "email": email,
            "password": password,
            "year_of_study": year_of_study,
            "degree": degree,
            "branch": branch,
            "area_of_interest": area_of_interest,
            "secret_word": secret_word
        }
        
        mongo.db.users.insert_one(user_data)
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# -------------------- Forgot Password --------------------
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]
        secret_word = request.form["secret_word"]
        new_password = request.form["new_password"]

        user = mongo.db.users.find_one({"email": email})

        if user and user["secret_word"] == secret_word:
            hashed_password = bcrypt.generate_password_hash(new_password).decode("utf-8")
            mongo.db.users.update_one({"email": email}, {"$set": {"password": hashed_password}})
            flash("Password reset successful! You can log in now.", "success")
            return redirect(url_for("login"))

        flash("Incorrect email or secret word!", "danger")

    return render_template("forgot_password.html")


# -------------------- Profile Page --------------------
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    user = mongo.db.users.find_one({"email": session["user"]})

    if request.method == "POST":
        updated_data = {
            "name": request.form["name"],
            "year_of_study": request.form["year_of_study"],
            "degree": request.form["degree"],
            "branch": request.form["branch"],
            "area_of_interest": request.form["area_of_interest"]
        }
        mongo.db.users.update_one({"email": session["user"]}, {"$set": updated_data})
        flash("Profile updated successfully!", "success")
    
    return render_template("profile.html", user=user)


# -------------------- Logout --------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)


