import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import seaborn as sns
from analysis_functions import run_analysis

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

cached_df = None

def df_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def get_columns_info(df):
    return {
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=[np.number]).columns.tolist()
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global cached_df
    if 'file' not in request.files:
        return jsonify(success=False, message='No file part')
    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, message='No selected file')
    try:
        df = pd.read_csv(file)
        cached_df = df
        col_info = get_columns_info(df)
        return jsonify(success=True, message='File uploaded successfully', data=col_info)
    except Exception as e:
        return jsonify(success=False, message=str(e))

@app.route('/analyze', methods=['POST'])
def analyze():
    global cached_df
    if cached_df is None:
        return jsonify(success=False, message='No data loaded')

    try:
        params = {
            'analysis_type': request.form.get('analysis_type'),
            'columns': request.form.getlist('columns'),
            'dependent_var': request.form.get('dependent_var'),
            'degree': int(request.form.get('degree', 2)),
            'var1': request.form.get('var1'),
            'var2': request.form.get('var2')
        }
        result = run_analysis(cached_df, params)
        return jsonify(success=True, result=result)
    except Exception as e:
        return jsonify(success=False, message=str(e))

if __name__ == '__main__':
    app.run(debug=True)