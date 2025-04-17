from flask import Flask, render_template, request, redirect, session, url_for
import json

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Hardcoded user database
USERS = {
    "port": "port123",
    "Teachasst": "teachme123",
    "spencer": "spencer123",
    "visitor": "visit123"
}

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
            return redirect(url_for("ready_to_begin"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/ready_to_begin")
def ready_to_begin():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("ready_to_begin.html", user=session["username"])

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        session["answers"] = request.form
        return redirect(url_for("thank_you"))

    return render_template("questions.html", questions=QUESTIONS)

@app.route("/thank_you")
def thank_you():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("thank_you.html", user=session["username"])

if __name__ == "__main__":
    app.run(debug=True)