from flask import Flask, render_template, request, redirect, session, url_for, jsonify, make_response
import json
import random
import time
from datetime import datetime

app = Flask(__name__)
app.secret_key = "ITM352"

# Hardcoded user database
USERS = {
    "port": "port123",
    "Teachasst": "teachme123",
    "spencer": "spencer123",
    "visitor": "visit123"
}

# Load questions from file at app startup
try:
    with open("questions.json") as f:
        ALL_QUESTIONS = json.load(f)
        
    # Validate questions data structure
    for i, q in enumerate(ALL_QUESTIONS):
        if "question" not in q or "choices" not in q or "answer" not in q:
            print(f"Warning: Question at index {i} has invalid format")
            continue
        
        # Ensure answer is in choices
        if q["answer"] not in q["choices"]:
            print(f"Warning: Answer '{q['answer']}' not in choices for question: {q['question']}")
            # Add the answer to choices if not present
            q["choices"].append(q["answer"])
            
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading questions: {e}")
    ALL_QUESTIONS = []

# Scores database
try:
    with open("scores.json", "r") as f:
        SCORES = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    SCORES = {}

def save_scores():
    with open("scores.json", "w") as f:
        json.dump(SCORES, f)

def get_leaderboard():
    all_scores = []
    for username, scores in SCORES.items():
        for score_data in scores:
            all_scores.append({
                "username": username,
                "score": score_data["score"],
                "total": score_data["total"],
                "percentage": score_data["percentage"],
                "difficulty": score_data["difficulty"],
                "date": score_data["date"]
            })
    
    # Sort by percentage, then by score
    sorted_scores = sorted(all_scores, key=lambda x: (x["percentage"], x["score"]), reverse=True)
    return sorted_scores[:5]  # Return top 5

@app.route("/")
def index():
    # Check if user has visited before
    if "username" in session:
        return render_template("index.html", returning=True, username=session["username"])
    return render_template("index.html", returning=False)

@app.route("/login", methods=["GET", "POST"])
def login():
    leaderboard = get_leaderboard()
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if USERS.get(username) == password:
            session["username"] = username
            session["score"] = 0
            session["question_index"] = 0
            
            # Initialize user in scores database if new
            if username not in SCORES:
                SCORES[username] = []
                save_scores()
                
            return redirect(url_for("select_difficulty"))
        else:
            return render_template("login.html", error="Invalid credentials", leaderboard=leaderboard)
    return render_template("login.html", leaderboard=leaderboard)

@app.route("/select_difficulty")
def select_difficulty():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("select_difficulty.html", user=session["username"])

@app.route("/start_quiz/<difficulty>")
def start_quiz(difficulty):
    if "username" not in session:
        return redirect(url_for("login"))
    
    # Error check - make sure we have questions
    if not ALL_QUESTIONS:
        return render_template("error.html", message="No questions available. Please check questions.json file.")
    
    # Set up quiz based on difficulty
    if difficulty == "easy":
        # First 5 questions or all if less than 5
        questions = ALL_QUESTIONS[:5] if len(ALL_QUESTIONS) >= 5 else ALL_QUESTIONS.copy()
    else:  # hard
        # Last 5 questions or all if less than 5
        questions = ALL_QUESTIONS[-5:] if len(ALL_QUESTIONS) >= 5 else ALL_QUESTIONS.copy()
    
    # Make a deep copy of questions to avoid modifying the original
    questions_copy = []
    for q in questions:
        # Ensure answer is in choices (safety check)
        choices = q["choices"].copy()
        correct_answer = q["answer"]
        
        if correct_answer not in choices:
            # Add the answer to choices if not present
            choices.append(correct_answer)
        
        questions_copy.append({
            "question": q["question"],
            "choices": choices,
            "answer": correct_answer
        })
    
    # Randomize question order
    random.shuffle(questions_copy)
    
    # Randomize answer choices for each question
    for q in questions_copy:
        choices = q["choices"].copy()
        correct_answer = q["answer"]
        
        # Double check answer is in choices
        if correct_answer not in choices:
            choices.append(correct_answer)
            
        random.shuffle(choices)
        
        try:
            # Find the new position of the correct answer
            correct_index = choices.index(correct_answer)
            # Update question with shuffled choices and new correct index
            q["choices"] = choices
            q["correct_index"] = correct_index
        except ValueError:
            # Fallback if something goes wrong
            q["choices"] = choices
            q["correct_index"] = 0  # Default to first option
            q["choices"][0] = correct_answer  # Set first option to correct answer

    # Store in session
    session["questions"] = questions_copy
    session["difficulty"] = difficulty
    session["start_time"] = time.time()
    session["question_index"] = 0
    session["score"] = 0
    session["answers"] = []  # To track user answers
    
    return redirect(url_for("ready_to_begin"))

@app.route("/ready_to_begin")
def ready_to_begin():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("ready_to_begin.html", user=session["username"], difficulty=session.get("difficulty", "easy"))

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if "username" not in session or "questions" not in session:
        return redirect(url_for("login"))

    questions = session.get("questions", [])
    index = session.get("question_index", 0)
    
    # Safety check
    if not questions or index >= len(questions):
        return redirect(url_for("thank_you"))
    
    # Process answer submission
    if request.method == "POST":
        selected_answer = request.form.get("answer")
        current_question = questions[index]
        
        # Safety check for correct_index
        if "correct_index" not in current_question or current_question["correct_index"] >= len(current_question["choices"]):
            correct_answer = current_question["answer"]
        else:
            correct_answer = current_question["choices"][current_question["correct_index"]]
        
        # Record answer
        is_correct = (selected_answer == correct_answer)
        session["answers"].append({
            "question": current_question["question"],
            "selected": selected_answer,
            "correct": correct_answer,
            "is_correct": is_correct
        })
        
        # Update score
        if is_correct:
            # Score based on difficulty
            if session["difficulty"] == "easy":
                session["score"] += 1
            else:  # hard
                session["score"] += 3
        
        # Move to next question
        index += 1
        session["question_index"] = index

    # Check if we've finished all questions
    if index >= len(questions):
        return redirect(url_for("thank_you"))
    
    # Show the current question
    question = questions[index]
    return render_template("questions.html", question=question, number=index + 1, total=len(questions))

@app.route("/thank_you")
def thank_you():
    if "username" not in session:
        return redirect(url_for("login"))
    
    username = session["username"]
    difficulty = session.get("difficulty", "easy")
    score = session.get("score", 0)
    answers = session.get("answers", [])
    questions = session.get("questions", [])
    total = len(questions) if questions else 0
    
    # Calculate time taken
    start_time = session.get("start_time", time.time())
    time_taken = round(time.time() - start_time, 2)
    
    # Calculate stats
    correct_count = sum(1 for answer in answers if answer.get("is_correct", False))
    incorrect_count = total - correct_count
    percentage = (correct_count / total) * 100 if total > 0 else 0
    
    # Save score history
    score_data = {
        "score": score,
        "total": total,
        "percentage": percentage,
        "time_taken": time_taken,
        "difficulty": difficulty,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "answers": answers
    }
    
    if username in SCORES:
        SCORES[username].append(score_data)
    else:
        SCORES[username] = [score_data]
    save_scores()
    
    # Get user history
    history = SCORES.get(username, [])
    
    # Get leaderboard
    leaderboard = get_leaderboard()
    
    return render_template("thank_you.html", 
                           user=username, 
                           score=score, 
                           total=total,
                           time_taken=time_taken,
                           correct_count=correct_count,
                           incorrect_count=incorrect_count,
                           percentage=percentage,
                           history=history,
                           leaderboard=leaderboard,
                           difficulty=difficulty)

# RESTful API endpoints
@app.route("/api/questions", methods=["GET"])
def api_questions():
    difficulty = request.args.get("difficulty", "easy")
    
    if difficulty == "easy":
        questions = ALL_QUESTIONS[:5] if len(ALL_QUESTIONS) >= 5 else ALL_QUESTIONS.copy()
    else:  # hard
        questions = ALL_QUESTIONS[-5:] if len(ALL_QUESTIONS) >= 5 else ALL_QUESTIONS.copy()
    
    return jsonify(questions)

@app.route("/api/scores", methods=["GET"])
def api_scores():
    username = request.args.get("username")
    if username:
        user_scores = SCORES.get(username, [])
        return jsonify(user_scores)
    return jsonify(SCORES)

@app.route("/api/leaderboard", methods=["GET"])
def api_leaderboard():
    return jsonify(get_leaderboard())

@app.route("/api/save_score", methods=["POST"])
def api_save_score():
    data = request.get_json()
    username = data.get("username")
    score_data = data.get("score_data")
    
    if not username or not score_data:
        return jsonify({"error": "Missing data"}), 400
    
    if username in SCORES:
        SCORES[username].append(score_data)
    else:
        SCORES[username] = [score_data]
    save_scores()
    
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True)