from flask import Flask, render_template, request
import datetime

app = Flask(__name__)

# Hardcoded user database
USERS = {
    "port": "port123",
    "Teachasst": "teachme123",
    "spencer": "spencer123",
    "olson": "olson123"
}
#Bring user to index
@app.route("/")
def index():
    return render_template("index.html")
#login page and processing
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if USERS.get(username) == password:
            #Returns success page
            return render_template("success.html", user=username)
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)
