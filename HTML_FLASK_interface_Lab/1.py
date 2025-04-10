from flask import Flask

app = Flask(__name__)


your_name = f'Spencer'

@app.route('/')
def get_Welcome():
    return f"Welcome to {your_name}'s Website"

if __name__ == '__main__':
    app.run(debug=True)