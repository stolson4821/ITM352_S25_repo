import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import chi2_contingency
import seaborn as sns


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

cached_df = None

# Helper functions
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
        analysis_type = request.form.get('analysis_type')
        columns = request.form.getlist('columns')
        dependent_var = request.form.get('dependent_var')
        degree = int(request.form.get('degree', 2))
        var1 = request.form.get('var1')
        var2 = request.form.get('var2')

        result = {}
        df = cached_df.dropna()

        if analysis_type == 'descriptive_statistics':
            result['statistics'] = df[columns].describe().to_dict()

        elif analysis_type == 'scatter_matrix':
            sns.set(style="ticks")
            fig = sns.pairplot(df[columns])
            result['scatter_matrix'] = [{'title': 'Scatter Matrix', 'image': df_to_base64(fig.fig)}]

        elif analysis_type == 'linear_regression':
            X = df[columns]
            y = df[dependent_var]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            fig, ax = plt.subplots()
            ax.scatter(y, y_pred)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Linear Regression Prediction')
            result['regression'] = [{'title': 'Linear Regression', 'image': df_to_base64(fig)}]

        elif analysis_type == 'multiple_regression':
            X = df[columns]
            y = df[dependent_var]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            fig, ax = plt.subplots()
            ax.scatter(y, y_pred)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Multiple Regression')
            result['multiple_regression'] = [{'title': 'Multiple Regression', 'image': df_to_base64(fig)}]

        elif analysis_type == 'polynomial_regression':
            X = df[columns]
            y = df[dependent_var]
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
            y_pred = model.predict(X_poly)
            fig, ax = plt.subplots()
            ax.scatter(y, y_pred)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'Polynomial Regression (Degree {degree})')
            result['polynomial_regression'] = [{'title': 'Polynomial Regression', 'image': df_to_base64(fig)}]

        elif analysis_type == 'logistic_regression':
            X = df[columns]
            y = df[dependent_var]
            model = LogisticRegression(max_iter=200).fit(X, y)
            y_pred = model.predict(X)
            report = classification_report(y, y_pred, output_dict=True)
            result['logistic_regression'] = report

        elif analysis_type == 'anova':
            model = stats.f_oneway(*(df.groupby(dependent_var)[col].values for col in columns))
            result['anova'] = {'F-statistic': model.statistic, 'p-value': model.pvalue}

        elif analysis_type == 'chi_square':
            contingency = pd.crosstab(df[var1], df[var2])
            chi2, p, dof, expected = chi2_contingency(contingency)
            result['chi_square'] = {'chi2': chi2, 'p': p, 'dof': dof, 'expected_freq': expected.tolist()}

        elif analysis_type == 'correlation':
            corr = df[columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            result['correlation'] = [{'title': 'Correlation Heatmap', 'image': df_to_base64(fig)}]

        elif analysis_type == 'pca':
            X = df[columns]
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots()
            ax.scatter(components[:, 0], components[:, 1])
            ax.set_title('PCA: First 2 Components')
            result['pca'] = [{'title': 'PCA Plot', 'image': df_to_base64(fig)}]

        elif analysis_type == 'clustering':
            X = df[columns]
            kmeans = KMeans(n_clusters=3, n_init='auto')
            clusters = kmeans.fit_predict(X)
            fig, ax = plt.subplots()
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
            ax.set_title('KMeans Clustering')
            result['clustering'] = [{'title': 'KMeans Clustering', 'image': df_to_base64(fig)}]

        elif analysis_type == 'residual_diagnostics':
            X = df[columns]
            y = df[dependent_var]
            model = LinearRegression().fit(X, y)
            residuals = y - model.predict(X)
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title('Residuals Distribution')
            result['residuals'] = [{'title': 'Residual Diagnostics', 'image': df_to_base64(fig)}]

        else:
            return jsonify(success=False, message='Unknown analysis type')

        return jsonify(success=True, result=result)

    except Exception as e:
        return jsonify(success=False, message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
