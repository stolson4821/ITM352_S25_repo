from flask import Flask
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def get_daetime():
    current_time = datetime.now().strftime('%y-%m-%d %H:%m:%S')
    return f'Current Server Date and Time: {current_time}'

if __name__ == '__main__':
    app.run(debug=True)