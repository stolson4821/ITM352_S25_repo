from flask import Flask, render_template
import requests

app = Flask(__name__)

@app.route("/")
def meme():
    url = "https://meme-api.com/gimme/wholesomememes"
    response = requests.request("GET", url)
    data = response.json()
    meme_url = data.get("url")
    subreddit = data.get("subreddit")
    return render_template("meme.html", meme_url=meme_url, subreddit=subreddit)

if __name__ == "__main__":
    app.run(debug=True)