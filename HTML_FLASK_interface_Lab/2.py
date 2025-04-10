from flask import Flask, render_template, request

app = Flask(__name__)

# Hardcoded user database
USERS = {
    "spencer": "spencer123",
    "olson": "olson123"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if USERS.get(username) == password:
            return render_template("success.html", user=username)
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)
