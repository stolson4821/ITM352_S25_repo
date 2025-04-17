from flask import Flask, render_template, request, redirect, session, url_for
import json

app = Flask(__name__)
app.secret_key = "ITM352"

# Simple user database
USERS = {
    "port": "port123",
    "Teachasst": "teachme123",
    "spencer": "spencer123",
    "visitor": "visit123"
}

# Load questions from file at app startup
with open("questions.json") as f:
    QUESTIONS = json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if USERS.get(username) == password:
            session["username"] = username
            session["score"] = 0
            session["question_index"] = 0
            return redirect(url_for("ready_to_begin"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")
#I added this page because there was always a google password save/security 
#breach pop up so i added a page to ensure that the user is ready. 
@app.route("/ready_to_begin")
def ready_to_begin():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("ready_to_begin.html", user=session["username"])
#Begin the questions
@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if "username" not in session:
        return redirect(url_for("login"))

    # Get current question index
    index = session.get("question_index", 0)
    
    # Process answer submission
    if request.method == "POST":
        selected_answer = request.form.get("answer")
        current_question = QUESTIONS[index]
        correct_answer = current_question["answer"]
        
        # Check if answer is correct and update score
        if selected_answer == correct_answer:
            session["score"] = session.get("score", 0) + 1
        
        # Move to next question
        index += 1
        session["question_index"] = index

    # Check if we've finished all questions
    if index >= len(QUESTIONS):
        return redirect(url_for("thank_you"))
    
    # Show the current question
    question = QUESTIONS[index]
    return render_template("questions.html", question=question, number=index + 1)

@app.route("/thank_you")
def thank_you():
    if "username" not in session:
        return redirect(url_for("login"))
    
    score = session.get("score", 0)
    total = len(QUESTIONS)
    
    return render_template("thank_you.html", user=session["username"], score=score, total=total)

if __name__ == "__main__":
    app.run(debug=True)