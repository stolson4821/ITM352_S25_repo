import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for web environment
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, r2_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import traceback
import logging
from analysis_functions import StatisticalAnalysisTool, run_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
cached_df = None
analysis_tool = None
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB limit

def df_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string"""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting figure to base64: {str(e)}")
        plt.close(fig)
        return None

def get_columns_info(df):
    """Extract column information from dataframe"""
    try:
        return {
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=[np.number]).columns.tolist(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
        }
    except Exception as e:
        logger.error(f"Error getting column info: {str(e)}")
        return {}

def validate_csv(file_path):
    """Validate CSV file format and content"""
    try:
        # Check file extension
        if not file_path.lower().endswith('.csv'):
            return False, "File must have .csv extension"
        
        # Check if file is readable
        df = pd.read_csv(file_path, nrows=5)
        
        # Check if file has data
        if df.empty:
            return False, "CSV file is empty"
            
        # Check total file size
        file_size = os.path.getsize(file_path)
        if file_size > MAX_UPLOAD_SIZE:
            return False, f"File size exceeds maximum limit of {MAX_UPLOAD_SIZE/(1024*1024)}MB"
            
        # Check for at least some numeric columns (needed for statistical analysis)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return False, "CSV file must contain at least one numeric column for statistical analysis"
            
        return True, "File is valid"
    except pd.errors.EmptyDataError:
        return False, "CSV file is empty"
    except pd.errors.ParserError:
        return False, "CSV file format is invalid"
    except Exception as e:
        return False, f"Error validating CSV: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global cached_df, analysis_tool
    
    if 'file' not in request.files:
        return jsonify(success=False, message='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, message='No selected file')
    
    try:
        # Create upload folder if it doesn't exist
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        # Save file to disk
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Validate CSV file
        is_valid, message = validate_csv(file_path)
        if not is_valid:
            # Remove invalid file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify(success=False, message=message)
        
        # Create a new StatisticalAnalysisTool instance
        analysis_tool = StatisticalAnalysisTool()
        
        # Load the CSV directly into the tool
        try:
            analysis_tool.df = pd.read_csv(file_path)
            cached_df = analysis_tool.df
            
            # Log basic info about the dataset
            logger.info(f"Loaded CSV: {file.filename}, Shape: {cached_df.shape}")
            
            # Categorize columns in the tool
            analysis_tool.numeric_columns = analysis_tool.df.select_dtypes(include=[np.number]).columns.tolist()
            analysis_tool.categorical_columns = analysis_tool.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Get additional column metadata
            col_info = get_columns_info(cached_df)
            
            # Dataset preview (first 5 rows)
            preview_html = cached_df.head(5).to_html(classes='table table-striped table-sm')
            
            return jsonify(
                success=True, 
                message='File uploaded successfully', 
                data=col_info,
                preview=preview_html
            )
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return jsonify(success=False, message=f"Error loading CSV: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify(success=False, message=f"Error: {str(e)}")

@app.route('/analyze', methods=['POST'])
def analyze():
    global cached_df, analysis_tool
    
    if cached_df is None or analysis_tool is None:
        return jsonify(success=False, message='No data loaded')

    try:
        analysis_type = request.form.get('analysis_type')
        logger.info(f"Requested analysis: {analysis_type}")
        
        # Capture all form data and selected columns
        selected_columns = request.form.getlist('columns')
        
        # Set selected columns in the tool
        analysis_tool.selected_columns = selected_columns
        
        # Create a buffer to capture matplotlib output
        buf = io.BytesIO()
        
        # Initialize result dictionary
        result = {}
        
        # Process based on analysis type
        if analysis_type == 'descriptive_statistics':
            # Get numeric columns
            numeric_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns]
            if not numeric_cols:
                return jsonify(success=False, message='Please select at least one numeric column')
            
            # Handle row filtering if provided
            start_row = None
            end_row = None
            
            if request.form.get('filter_rows') == 'true':
                try:
                    start_row = int(request.form.get('start_row', 0))
                    end_row = int(request.form.get('end_row', len(cached_df)))
                    
                    if start_row < 0 or end_row > len(cached_df) or start_row >= end_row:
                        return jsonify(success=False, 
                            message=f'Invalid row range. Valid range is 0 to {len(cached_df)}')
                except ValueError:
                    return jsonify(success=False, message='Invalid row values')
            
            # Use the tool's method but capture the results
            analysis_tool.selected_columns = numeric_cols
            
            # Filter data if needed
            data = cached_df
            if start_row is not None and end_row is not None:
                data = cached_df.iloc[start_row:end_row]
                
            # Create a basic descriptive stats dataframe
            stats_df = data[numeric_cols].describe().T
            stats_df['median'] = data[numeric_cols].median()
            try:
                stats_df['mode'] = data[numeric_cols].mode().iloc[0]
            except IndexError:
                stats_df['mode'] = np.nan
                
            stats_df['skewness'] = data[numeric_cols].skew()
            stats_df['kurtosis'] = data[numeric_cols].kurtosis()
            stats_df['missing'] = data[numeric_cols].isna().sum()
            stats_df['missing_percent'] = (data[numeric_cols].isna().sum() / len(data)) * 100
            
            # Convert to HTML for display
            result['statistics_html'] = stats_df.to_html(classes='table table-striped', float_format='%.4f')
            result['raw_statistics'] = stats_df.to_dict()
            
            # Add metadata about the analysis
            result['metadata'] = {
                'total_rows_analyzed': len(data),
                'columns_analyzed': numeric_cols,
                'row_filter_applied': start_row is not None
            }
            
            # Create visualizations
            fig = plt.figure(figsize=(10, 5 * len(numeric_cols)))
            
            for i, col in enumerate(numeric_cols, 1):
                # Histogram
                plt.subplot(len(numeric_cols), 2, 2*i-1)
                sns.histplot(data[col].dropna(), kde=True)
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                
                # Box plot
                plt.subplot(len(numeric_cols), 2, 2*i)
                sns.boxplot(x=data[col].dropna(), orient='h')
                plt.title(f'Box Plot of {col}')
                plt.xlabel(col)
            
            plt.tight_layout()
            result['visualization'] = df_to_base64(fig)
            
        elif analysis_type == 'scatter_matrix':
            if len(selected_columns) < 2:
                return jsonify(success=False, message='Please select at least two columns')
                
            # Use only numeric columns
            numeric_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns]
            
            if len(numeric_cols) < 2:
                return jsonify(success=False, message='Please select at least two numeric columns')
            
            # Create scatter matrix
            plt.figure(figsize=(12, 10))
            sns.set_theme(style="ticks")
            
            scatter_plot = sns.pairplot(cached_df[numeric_cols], diag_kind="kde", markers="o", 
                                       plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                                       height=2.5)
                                       
            plt.suptitle("Scatter Matrix with KDE Diagonal", y=1.02, fontsize=16)
            plt.tight_layout()
            
            result['visualization'] = df_to_base64(scatter_plot.fig)
            # Add correlation matrix for additional insight
            result['correlation_matrix_html'] = cached_df[numeric_cols].corr().to_html(
                classes='table table-striped', float_format='%.4f')
            
        elif analysis_type == 'linear_regression':
            # Get dependent variable
            y_col = request.form.get('dependent_var')
            
            if not y_col or y_col not in analysis_tool.numeric_columns:
                return jsonify(success=False, message='Please select a valid numeric dependent variable')
                
            # Use the first selected column as independent variable
            x_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns and col != y_col]
            
            if not x_cols:
                return jsonify(success=False, message='Please select at least one numeric independent variable')
                
            x_col = x_cols[0]  # Use the first column for simple linear regression
            
            # Create a clean dataframe
            clean_df = cached_df[[x_col, y_col]].dropna()
            
            if len(clean_df) < 2:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            # Fit the model
            X = clean_df[x_col].values.reshape(-1, 1)
            y = clean_df[y_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions and calculate statistics
            y_pred = model.predict(X)
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(X, y)
            
            # Calculate additional statistics
            n = len(clean_df)
            residuals = y - y_pred
            mean_squared_error = np.mean(residuals**2)
            residual_sum_of_squares = np.sum(residuals**2)
            total_sum_of_squares = np.sum((y - np.mean(y))**2)
            
            # Safe calculation of F-statistic to avoid division by zero
            if (n - 2) > 0 and residual_sum_of_squares > 0:
                f_statistic = ((total_sum_of_squares - residual_sum_of_squares) / 1) / (residual_sum_of_squares / (n - 2))
                p_value = 1 - stats.f.cdf(f_statistic, 1, n - 2)
            else:
                f_statistic = np.nan
                p_value = np.nan
            
            # Store results
            result['equation'] = f"{y_col} = {intercept:.4f} + {slope:.4f} * {x_col}"
            result['slope'] = f"{slope:.4f}"
            result['intercept'] = f"{intercept:.4f}"
            result['r_squared'] = f"{r_squared:.4f}"
            result['mean_squared_error'] = f"{mean_squared_error:.4f}"
            result['f_statistic'] = f"{f_statistic:.4f}" if not np.isnan(f_statistic) else "N/A"
            result['p_value'] = f"{p_value:.4f}" if not np.isnan(p_value) else "N/A"
            result['n_observations'] = n
            result['significance'] = "Significant (p < 0.05)" if p_value < 0.05 else "Not significant (p >= 0.05)"
            
            # Create visualizations
            fig = plt.figure(figsize=(15, 10))
            
            # Plot 1: Scatter plot with regression line
            plt.subplot(2, 2, 1)
            plt.scatter(X, y, alpha=0.7)
            plt.plot(X, y_pred, color='red', linewidth=2)
            plt.title(f'Linear Regression: {y_col} vs {x_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            
            # Plot 2: Residuals vs Fitted
            plt.subplot(2, 2, 2)
            plt.scatter(y_pred, residuals, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Residuals vs Fitted')
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            
            # Plot 3: Histogram of residuals
            plt.subplot(2, 2, 3)
            plt.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Histogram of Residuals')
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            
            # Plot 4: Q-Q plot
            plt.subplot(2, 2, 4)
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Q-Q Plot of Residuals')
            
            plt.tight_layout()
            result['visualization'] = df_to_base64(fig)
            
        elif analysis_type == 'multiple_regression':
            # Get dependent variable
            y_col = request.form.get('dependent_var')
            
            if not y_col or y_col not in analysis_tool.numeric_columns:
                return jsonify(success=False, message='Please select a valid numeric dependent variable')
                
            # Use remaining selected columns as independent variables
            x_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns and col != y_col]
            
            if not x_cols:
                return jsonify(success=False, message='Please select at least one numeric independent variable')
                
            # Create a clean dataframe
            analysis_cols = x_cols + [y_col]
            clean_df = cached_df[analysis_cols].dropna()
            
            if len(clean_df) < len(x_cols) + 1:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            # Check for multicollinearity using VIF
            X = sm.add_constant(clean_df[x_cols])
            
            # Calculate VIF only if we have more than one independent variable
            if len(x_cols) > 1:
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X.columns
                
                # Handle errors in VIF calculation
                try:
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    result['vif_data_html'] = vif_data.to_html(classes='table table-striped', float_format='%.4f')
                    
                    # Check for high multicollinearity
                    high_vif = vif_data[vif_data["VIF"] > 10].shape[0] > 0
                    if high_vif:
                        result['multicollinearity_warning'] = "Warning: High multicollinearity detected (VIF > 10)"
                except Exception as e:
                    logger.warning(f"VIF calculation error: {str(e)}")
                    result['vif_error'] = f"Could not calculate VIF: {str(e)}"
            
            # Fit the model using statsmodels for detailed statistics
            formula = f"{y_col} ~ {' + '.join(x_cols)}"
            
            try:
                model = ols(formula, data=clean_df).fit()
                
                # Store summary as HTML
                result['summary_html'] = model.summary().as_html()
                
                # Extract key statistics
                result['r_squared'] = f"{model.rsquared:.4f}"
                result['adj_r_squared'] = f"{model.rsquared_adj:.4f}"
                result['f_statistic'] = f"{model.fvalue:.4f}"
                result['p_value'] = f"{model.f_pvalue:.4f}"
                result['n_observations'] = model.nobs
                result['aic'] = f"{model.aic:.4f}"
                result['significance'] = "Significant (p < 0.05)" if model.f_pvalue < 0.05 else "Not significant (p >= 0.05)"
                
                # Coefficients table
                coef_df = pd.DataFrame({
                    'Coefficient': model.params,
                    'Std Error': model.bse,
                    't-value': model.tvalues,
                    'p-value': model.pvalues
                })
                result['coefficients_html'] = coef_df.to_html(classes='table table-striped', float_format='%.4f')
                
                # For sklearn version (for visualization)
                X = clean_df[x_cols]
                y = clean_df[y_col]
                
                sk_model = LinearRegression()
                sk_model.fit(X, y)
                y_pred = sk_model.predict(X)
                residuals = y - y_pred
                
                # Create visualizations
                fig = plt.figure(figsize=(15, 10))
                
                # Plot 1: Actual vs Predicted
                plt.subplot(2, 2, 1)
                plt.scatter(y, y_pred, alpha=0.7)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                plt.title('Actual vs Predicted')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                
                # Plot 2: Residuals vs Fitted
                plt.subplot(2, 2, 2)
                plt.scatter(y_pred, residuals, alpha=0.7)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.title('Residuals vs Fitted')
                plt.xlabel('Fitted values')
                plt.ylabel('Residuals')
                
                # Plot 3: Histogram of residuals
                plt.subplot(2, 2, 3)
                plt.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Histogram of Residuals')
                plt.xlabel('Residual Value')
                plt.ylabel('Frequency')
                
                # Plot 4: Q-Q plot
                plt.subplot(2, 2, 4)
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title('Q-Q Plot of Residuals')
                
                plt.tight_layout()
                result['visualization'] = df_to_base64(fig)
                
            except Exception as e:
                logger.error(f"Multiple regression error: {str(e)}")
                return jsonify(success=False, message=f"Error in regression analysis: {str(e)}")
            
        elif analysis_type == 'polynomial_regression':
            # Get dependent variable
            y_col = request.form.get('dependent_var')
            
            if not y_col or y_col not in analysis_tool.numeric_columns:
                return jsonify(success=False, message='Please select a valid numeric dependent variable')
                
            # Use the first selected column as independent variable
            x_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns and col != y_col]
            
            if not x_cols:
                return jsonify(success=False, message='Please select at least one numeric independent variable')
                
            x_col = x_cols[0]  # Use the first column for polynomial regression
            
            # Get degree
            try:
                degree = int(request.form.get('degree', 2))
                if degree < 1 or degree > 10:
                    return jsonify(success=False, message='Polynomial degree must be between 1 and 10')
            except ValueError:
                return jsonify(success=False, message='Invalid degree value')
                
            # Create a clean dataframe
            clean_df = cached_df[[x_col, y_col]].dropna()
            
            if len(clean_df) < 3:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            # Prepare data
            X = clean_df[x_col].values.reshape(-1, 1)
            y = clean_df[y_col].values
            
            # Generate polynomial features
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            
            # Fit the model
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Make predictions
            y_pred = model.predict(X_poly)
            
            # Calculate statistics
            r_squared = r2_score(y, y_pred)
            residuals = y - y_pred
            mse = np.mean(residuals**2)
            
            # Adjusted R-squared calculation
            n = len(y)
            p = degree + 1  # number of parameters
            adj_r_squared = 1 - (1-r_squared)*(n-1)/(n-p-1) if n-p-1 > 0 else np.nan
            
            # Store results
            result['degree'] = degree
            result['r_squared'] = f"{r_squared:.4f}"
            result['adj_r_squared'] = f"{adj_r_squared:.4f}" if not np.isnan(adj_r_squared) else "N/A"
            result['mean_squared_error'] = f"{mse:.4f}"
            result['n_observations'] = n
            
            # Display the polynomial equation
            coefficients = model.coef_
            intercept = model.intercept_
            
            equation = f"{y_col} = {intercept:.4f}"
            for i in range(1, degree + 1):
                sign = "+" if coefficients[i] >= 0 else "-"
                equation += f" {sign} {abs(coefficients[i]):.4f} × {x_col}^{i}"
                
            result['equation'] = equation
            
            # Create visualizations
            fig = plt.figure(figsize=(15, 10))
            
            # Plot 1: Scatter plot with polynomial curve
            plt.subplot(2, 2, 1)
            plt.scatter(X, y, alpha=0.7)
            
            # Sort X values for a smooth curve
            X_sorted = np.sort(X, axis=0)
            X_poly_sorted = poly_features.transform(X_sorted)
            y_poly_sorted = model.predict(X_poly_sorted)
            
            plt.plot(X_sorted, y_poly_sorted, color='red', linewidth=2)
            plt.title(f'Polynomial Regression (Degree {degree}): {y_col} vs {x_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            
            # Plot 2: Residuals vs Fitted
            plt.subplot(2, 2, 2)
            plt.scatter(y_pred, residuals, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Residuals vs Fitted')
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            
            # Plot 3: Histogram of residuals
            plt.subplot(2, 2, 3)
            plt.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Histogram of Residuals')
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            
            # Plot 4: Q-Q plot
            plt.subplot(2, 2, 4)
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Q-Q Plot of Residuals')
            
            plt.tight_layout()
            result['visualization'] = df_to_base64(fig)
            
        elif analysis_type == 'logistic_regression':
            # Get dependent variable
            y_col = request.form.get('dependent_var')
            
            if not y_col:
                return jsonify(success=False, message='Please select a valid dependent variable')
                
            # Check if dependent variable is binary
            unique_vals = cached_df[y_col].nunique()
            if unique_vals != 2:
                return jsonify(success=False, message='Dependent variable should be binary (have exactly 2 unique values)')
                
            # Use remaining selected columns as independent variables
            x_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns and col != y_col]
            
            if not x_cols:
                return jsonify(success=False, message='Please select at least one numeric independent variable')
                
            # Create a clean dataframe
            analysis_cols = x_cols + [y_col]
            clean_df = cached_df[analysis_cols].dropna()
            
            if len(clean_df) < len(x_cols) + 1:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            # Check if we need to encode the target variable
            y = clean_df[y_col]
            if not pd.api.types.is_numeric_dtype(y):
                unique_vals = y.unique()
                result['encoding_info'] = f"0 = {unique_vals[0]}, 1 = {unique_vals[1]}"
                y = pd.factorize(y)[0]
            else:
                y = y.values
                
            X = clean_df[x_cols].values
            
            # Fit the model
            try:
                model = LogisticRegression(max_iter=1000, solver='liblinear')
                model.fit(X, y)
                
                # Make predictions
                y_pred_proba = model.predict_proba(X)[:, 1]
                y_pred = model.predict(X)
                
                # Calculate statistics
                class_report = classification_report(y, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y, y_pred)
                accuracy = model.score(X, y)
                
                # Add metrics
                result['accuracy'] = f"{accuracy:.4f}"
                result['precision'] = f"{class_report['1']['precision']:.4f}"
                result['recall'] = f"{class_report['1']['recall']:.4f}"
                result['f1_score'] = f"{class_report['1']['f1-score']:.4f}"
                
                # Store coefficient information
                coef_info = []
                for i, col in enumerate(x_cols):
                    coef_info.append(f"{col}: {model.coef_[0][i]:.4f}")
                result['coefficients'] = coef_info
                result['intercept'] = f"{model.intercept_[0]:.4f}"
                
                # Format confusion matrix for display
                result['confusion_matrix'] = conf_matrix.tolist()
                result['confusion_matrix_labels'] = ["Predicted Negative", "Predicted Positive"]
                result['classification_report'] = class_report
                
                # Create a DataFrame for coefficients
                coef_df = pd.DataFrame({
                    'Feature': x_cols,
                    'Coefficient': model.coef_[0],
                    'Odds_Ratio': np.exp(model.coef_[0])
                })
                result['coefficients_html'] = coef_df.to_html(classes='table table-striped', float_format='%.4f')
                
                # Create visualizations
                fig = plt.figure(figsize=(15, 10))
                
                # Plot 1: Confusion Matrix
                plt.subplot(2, 2, 1)
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Plot 2: ROC Curve
                plt.subplot(2, 2, 2)
                
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title('ROC Curve')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc='lower right')
                
                # Plot 3: Predicted Probabilities
                plt.subplot(2, 2, 3)
                plt.hist(y_pred_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Histogram of Predicted Probabilities')
                plt.xlabel('Predicted Probability')
                plt.ylabel('Frequency')
                
                # Plot 4: Coefficient values
                plt.subplot(2, 2, 4)
                coef_df_sorted = coef_df.sort_values('Coefficient')
                
                sns.barplot(x='Coefficient', y='Feature', data=coef_df_sorted)
                plt.title('Feature Coefficients')
                plt.axvline(x=0, color='r', linestyle='-')
                
                plt.tight_layout()
                result['visualization'] = df_to_base64(fig)
                
            except Exception as e:
                logger.error(f"Logistic regression error: {str(e)}")
                return jsonify(success=False, message=f"Error in logistic regression: {str(e)}")
            
        elif analysis_type == 'anova':
            # Get dependent variable
            y_col = request.form.get('dependent_var')
            
            if not y_col or y_col not in analysis_tool.numeric_columns:
                return jsonify(success=False, message='Please select a valid numeric dependent variable')
                
            # Get categorical variable (from selected columns that aren't numeric)
            grouping_var = request.form.get('grouping_var')
            if not grouping_var:
                return jsonify(success=False, message='Please select a grouping variable')
                
            # Create a clean dataframe
            clean_df = cached_df[[y_col, grouping_var]].dropna()
            
            if len(clean_df) < 3:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            # Check if there are enough groups
            groups = clean_df[grouping_var].unique()
            if len(groups) < 2:
                return jsonify(success=False, message='ANOVA requires at least 2 groups for comparison')
                
            # Prepare data for ANOVA
            groups_data = []
            labels = []
            
            for group in groups:
                group_data = clean_df[clean_df[grouping_var] == group][y_col].values
                if len(group_data) > 0:
                    groups_data.append(group_data)
                    labels.append(str(group))
            
            # Perform one-way ANOVA
            try:
                f_statistic, p_value = stats.f_oneway(*groups_data)
                
                # Store results
                result['f_statistic'] = f"{f_statistic:.4f}"
                result['p_value'] = f"{p_value:.4f}"
                result['significance'] = 'Significant' if p_value < 0.05 else 'Not Significant'
                result['alpha'] = 0.05
                result['interpretation'] = "There is a statistically significant difference between group means" if p_value < 0.05 else "There is no statistically significant difference between group means"
                
                # Descriptive statistics per group
                group_stats = clean_df.groupby(grouping_var)[y_col].agg(['count', 'mean', 'std', 'min', 'max'])
                result['group_stats_html'] = group_stats.to_html(classes='table table-striped')
                
                # Run Tukey's HSD post-hoc test if there are more than 2 groups
                if len(groups) > 2:
                    try:
                        tukey_results = pairwise_tukeyhsd(clean_df[y_col], clean_df[grouping_var], alpha=0.05)
                        result['tukey_results'] = str(tukey_results)
                        
                        # Create a more readable version of Tukey results
                        tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], 
                                               columns=tukey_results._results_table.data[0])
                        result['tukey_html'] = tukey_df.to_html(classes='table table-striped')
                    except Exception as e:
                        result['tukey_error'] = str(e)
                
                # Create visualizations
                fig = plt.figure(figsize=(15, 10))
                
                # Plot 1: Box plot by group
                plt.subplot(2, 2, 1)
                sns.boxplot(x=grouping_var, y=y_col, data=clean_df)
                plt.title(f'Box Plot of {y_col} by {grouping_var}')
                plt.xticks(rotation=45 if max([len(str(g)) for g in groups]) > 3 else 0)
                
                # Plot 2: Violin plot by group
                plt.subplot(2, 2, 2)
                sns.violinplot(x=grouping_var, y=y_col, data=clean_df)
                plt.title(f'Violin Plot of {y_col} by {grouping_var}')
                plt.xticks(rotation=45 if max([len(str(g)) for g in groups]) > 3 else 0)
                
                # Plot 3: Bar plot with error bars
                plt.subplot(2, 2, 3)
                sns.barplot(x=grouping_var, y=y_col, data=clean_df, ci=95)
                plt.title(f'Mean {y_col} by {grouping_var} (with 95% CI)')
                plt.xticks(rotation=45 if max([len(str(g)) for g in groups]) > 3 else 0)
                
                # Plot 4: Strip plot overlaid on box plot
                plt.subplot(2, 2, 4)
                sns.boxplot(x=grouping_var, y=y_col, data=clean_df, showmeans=True, 
                           meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"})
                sns.stripplot(x=grouping_var, y=y_col, data=clean_df, jitter=True, 
                             alpha=0.4, color='black', size=4)
                plt.title(f'Box Plot with Data Points of {y_col} by {grouping_var}')
                plt.xticks(rotation=45 if max([len(str(g)) for g in groups]) > 3 else 0)
                
                plt.tight_layout()
                result['visualization'] = df_to_base64(fig)
                
            except Exception as e:
                logger.error(f"ANOVA error: {str(e)}")
                return jsonify(success=False, message=f"Error in ANOVA: {str(e)}")
            
        elif analysis_type == 'chi_square':
            # Get two categorical variables
            var1 = request.form.get('var1')
            var2 = request.form.get('var2')
            
            if not var1 or not var2 or var1 == var2:
                return jsonify(success=False, message='Please select two different categorical variables')
                
            # Create a clean dataframe
            clean_df = cached_df[[var1, var2]].dropna()
            
            if len(clean_df) < 5:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            try:
                # Create contingency table
                contingency_table = pd.crosstab(clean_df[var1], clean_df[var2])
                
                # Calculate row and column totals
                row_totals = contingency_table.sum(axis=1)
                col_totals = contingency_table.sum(axis=0)
                total = contingency_table.sum().sum()
                
                # Calculate expected frequencies
                expected = np.outer(row_totals, col_totals) / total
                expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
                
                # Check assumptions
                expected_less_than_5 = (expected_df < 5).sum().sum()
                total_cells = expected_df.size
                
                if expected_less_than_5 / total_cells > 0.2:
                    result['warning'] = 'More than 20% of cells have expected frequencies less than 5. Chi-square test may not be reliable.'
                    
                # Perform Chi-Square test
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                
                # Store results
                result['chi_square'] = f"{chi2:.4f}"
                result['p_value'] = f"{p:.4f}"
                result['degrees_of_freedom'] = dof
                result['significance'] = 'Significant (p < 0.05)' if p < 0.05 else 'Not Significant (p >= 0.05)'
                
                if p < 0.05:
                    result['interpretation'] = 'There is a significant association between the variables.'
                else:
                    result['interpretation'] = 'There is no significant association between the variables.'
                    
                # Calculate Cramer's V for effect size
                n = total
                min_dim = min(contingency_table.shape) - 1
                if min_dim > 0 and n > 0 and chi2 > 0:
                    cramers_v = np.sqrt(chi2 / (n * min_dim))
                    result['cramers_v'] = f"{cramers_v:.4f}"
                    
                    # Interpret Cramer's V
                    if cramers_v < 0.1:
                        effect_size = "Negligible"
                    elif cramers_v < 0.3:
                        effect_size = "Small"
                    elif cramers_v < 0.5:
                        effect_size = "Medium"
                    else:
                        effect_size = "Large"
                    result['effect_size'] = effect_size
                
                # Store tables as HTML
                result['contingency_table_html'] = contingency_table.to_html(classes='table table-striped')
                result['expected_table_html'] = expected_df.to_html(classes='table table-striped', float_format='%.2f')
                
                # Create visualizations
                fig = plt.figure(figsize=(15, 10))
                
                # Plot 1: Heatmap of observed frequencies
                plt.subplot(2, 2, 1)
                sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
                plt.title('Observed Frequencies')
                
                # Plot 2: Heatmap of expected frequencies
                plt.subplot(2, 2, 2)
                sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='YlGnBu')
                plt.title('Expected Frequencies')
                
                # Plot 3: Bar plot (using seaborn's countplot)
                plt.subplot(2, 2, 3)
                
                # Reshape data for plotting
                plot_data = clean_df.copy()
                plot_data['count'] = 1
                
                sns.countplot(x=var1, hue=var2, data=plot_data)
                plt.title('Count Plot')
                plt.xticks(rotation=45 if len(contingency_table.index) > 3 else 0)
                plt.legend(title=var2)
                
                # Plot 4: Stacked percentage bar chart
                plt.subplot(2, 2, 4)
                
                # Calculate percentages
                percentage_table = contingency_table.div(row_totals, axis=0) * 100
                percentage_table.plot(kind='bar', stacked=True)
                plt.title('Percentage Bar Chart')
                plt.xlabel(var1)
                plt.ylabel('Percentage')
                plt.xticks(rotation=45 if len(contingency_table.index) > 3 else 0)
                plt.legend(title=var2)
                
                plt.tight_layout()
                result['visualization'] = df_to_base64(fig)
                
            except Exception as e:
                logger.error(f"Chi-square test error: {str(e)}")
                return jsonify(success=False, message=f"Error in Chi-square test: {str(e)}")
            
        elif analysis_type == 'correlation':
            # Check if there are enough numeric columns
            numeric_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns]
            
            if len(numeric_cols) < 2:
                return jsonify(success=False, message='Correlation heatmap requires at least 2 numeric columns')
                
            # Create a clean dataframe
            clean_df = cached_df[numeric_cols].dropna()
            
            if len(clean_df) < 2:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            # Get correlation method
            corr_method = request.form.get('method', 'pearson')
            if corr_method not in ['pearson', 'spearman', 'kendall']:
                corr_method = 'pearson'  # Default to Pearson if invalid
                
            try:
                # Calculate correlation matrix
                correlation_matrix = clean_df.corr(method=corr_method)
                
                # Store as HTML
                result['correlation_matrix_html'] = correlation_matrix.to_html(
                    classes='table table-striped', float_format='%.4f')
                    
                # Find strongest correlations
                corr_unstack = correlation_matrix.unstack()
                # Remove self-correlations and duplicates
                corr_unstack = corr_unstack[corr_unstack < 1.0]
                
                # Get top 5 positive and negative correlations
                top_pos = corr_unstack.nlargest(5)
                top_neg = corr_unstack.nsmallest(5)
                
                # Format for display
                top_corr = pd.concat([top_pos, top_neg])
                top_corr_df = pd.DataFrame({'Variable Pair': top_corr.index, 'Correlation': top_corr.values})
                result['top_correlations_html'] = top_corr_df.to_html(
                    classes='table table-striped', float_format='%.4f')
                
                # Create visualization
                plt.figure(figsize=(12, 10))
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                
                # Plot heatmap
                sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', annot=True, 
                           fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
                           
                plt.title(f'{corr_method.capitalize()} Correlation Heatmap')
                plt.tight_layout()
                
                result['visualization'] = df_to_base64(plt.gcf())
                
                # Add interpretation guide
                result['interpretation_guide'] = """
                <div class="alert alert-info">
                    <strong>Interpreting Correlation Coefficients:</strong>
                    <ul>
                        <li>1.0: Perfect positive correlation</li>
                        <li>0.7-0.9: Strong positive correlation</li>
                        <li>0.4-0.6: Moderate positive correlation</li>
                        <li>0.1-0.3: Weak positive correlation</li>
                        <li>0: No correlation</li>
                        <li>-0.1 to -0.3: Weak negative correlation</li>
                        <li>-0.4 to -0.6: Moderate negative correlation</li>
                        <li>-0.7 to -0.9: Strong negative correlation</li>
                        <li>-1.0: Perfect negative correlation</li>
                    </ul>
                </div>
                """
                
            except Exception as e:
                logger.error(f"Correlation analysis error: {str(e)}")
                return jsonify(success=False, message=f"Error in correlation analysis: {str(e)}")
            
        elif analysis_type == 'pca':
            # Check if there are enough numeric columns
            numeric_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns]
            
            if len(numeric_cols) < 2:
                return jsonify(success=False, message='PCA requires at least 2 numeric columns')
                
            # Create a clean dataframe
            clean_df = cached_df[numeric_cols].dropna()
            
            if len(clean_df) < 2:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            try:
                # Standardize the data (mean=0, std=1)
                X = clean_df.values
                X_scaled = StandardScaler().fit_transform(X)
                
                # Get number of components (default to min of features or 5)
                n_components = min(len(numeric_cols), min(len(clean_df), 5))
                
                # Perform PCA
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(X_scaled)
                
                # Component loadings
                loadings = pca.components_
                
                # Calculate explained variance
                explained_variance = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                # Store results
                result['n_components'] = n_components
                
                # Explained variance as HTML
                var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(n_components)],
                    'Explained Variance (%)': explained_variance * 100,
                    'Cumulative Variance (%)': cumulative_variance * 100
                })
                result['variance_html'] = var_df.to_html(classes='table table-striped', float_format='%.2f')
                
                # Component loadings as HTML
                loadings_df = pd.DataFrame(loadings.T, index=numeric_cols, 
                                          columns=[f'PC{i+1}' for i in range(n_components)])
                result['loadings_html'] = loadings_df.to_html(classes='table table-striped', float_format='%.4f')
                
                # Create visualizations
                fig = plt.figure(figsize=(15, 10))
                
                # Plot 1: Scree plot
                plt.subplot(2, 2, 1)
                plt.bar(range(1, n_components + 1), explained_variance * 100)
                plt.plot(range(1, n_components + 1), cumulative_variance * 100, 'ro-')
                plt.axhline(y=80, color='r', linestyle='--')
                plt.title('Scree Plot')
                plt.xlabel('Principal Component')
                plt.ylabel('Explained Variance (%)')
                plt.xticks(range(1, n_components + 1))
                plt.ylim(0, 100)
                
                # Plot 2: Biplot (first two components)
                plt.subplot(2, 2, 2)
                
                # Only create biplot if we have at least 2 components
                if n_components >= 2:
                    # Scale factors for visualization
                    scale_x = 1.0/(loadings[0].max() - loadings[0].min())
                    scale_y = 1.0/(loadings[1].max() - loadings[1].min())
                    
                    # Plot data points
                    plt.scatter(components[:, 0], components[:, 1], alpha=0.5)
                    
                    # Plot feature vectors
                    for i, col in enumerate(numeric_cols):
                        plt.arrow(0, 0, loadings[0, i] * scale_x, loadings[1, i] * scale_y, 
                                 color='r', width=0.001, head_width=0.05)
                        plt.text(loadings[0, i] * scale_x * 1.15, loadings[1, i] * scale_y * 1.15, 
                                col, color='g')
                                
                    plt.title('PCA Biplot (PC1 vs PC2)')
                    plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
                    plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
                    plt.grid()
                
                # Plot 3: Loading plot - heatmap of loadings
                plt.subplot(2, 2, 3)
                
                # Show only first 5 components or less
                n_display = min(5, n_components)
                display_loadings = loadings[:n_display, :]
                
                sns.heatmap(display_loadings, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5,
                           xticklabels=numeric_cols, yticklabels=[f'PC{i+1}' for i in range(n_display)])
                plt.title('Component Loadings Heatmap')
                
                # Plot 4: Scatter plot of first two components with additional info
                plt.subplot(2, 2, 4)
                
                if n_components >= 2:
                    if len(components) > 10:
                        # Use density plot if many points
                        sns.kdeplot(x=components[:, 0], y=components[:, 1], cmap="Blues", fill=True)
                        plt.title('Density of PC1 vs PC2')
                        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
                        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
                    else:
                        plt.scatter(components[:, 0], components[:, 1])
                        
                        # Add indices for each point
                        for i, (x, y) in enumerate(zip(components[:, 0], components[:, 1])):
                            plt.text(x, y, str(i), fontsize=9)
                            
                        plt.title('PC1 vs PC2 with Indices')
                        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
                        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
                        plt.grid(True)
                
                plt.tight_layout()
                result['visualization'] = df_to_base64(fig)
                
                # Store transformed data
                result['transformed_data_preview'] = pd.DataFrame(
                    components, 
                    columns=[f'PC{i+1}' for i in range(n_components)]
                ).head(10).to_html(classes='table table-striped', float_format='%.4f')
                
            except Exception as e:
                logger.error(f"PCA analysis error: {str(e)}")
                return jsonify(success=False, message=f"Error in PCA analysis: {str(e)}")
            
        elif analysis_type == 'clustering':
            # Check if there are enough numeric columns
            numeric_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns]
            
            if len(numeric_cols) < 2:
                return jsonify(success=False, message='Clustering requires at least 2 numeric columns')
                
            # Create a clean dataframe
            clean_df = cached_df[numeric_cols].dropna()
            
            if len(clean_df) < 2:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            try:
                # Get number of clusters
                k = int(request.form.get('n_clusters', 3))
                if k < 2 or k > 10:
                    k = 3  # Default to 3 clusters if out of range
                    
                # Standardize the data
                X = clean_df.values
                X_scaled = StandardScaler().fit_transform(X)
                
                # Perform K-means
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Add cluster labels to dataframe
                cluster_df = clean_df.copy()
                cluster_df['Cluster'] = clusters
                
                # Output results
                result['n_clusters'] = k
                
                # Cluster sizes
                cluster_sizes = pd.Series(clusters).value_counts().sort_index()
                cluster_size_info = []
                for i, size in enumerate(cluster_sizes):
                    cluster_size_info.append(f"Cluster {i}: {size} points ({size/len(clusters)*100:.1f}%)")
                result['cluster_sizes'] = cluster_size_info
                
                # Cluster centers table
                centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
                centers_df.index.name = 'Cluster'
                result['centers_html'] = centers_df.to_html(classes='table table-striped', float_format='%.4f')
                
                # Calculate cluster statistics
                cluster_stats = cluster_df.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
                result['cluster_stats_html'] = cluster_stats.to_html(classes='table table-striped', float_format='%.4f')
                
                # Determine 2 features to visualize
                # Use first two for simplicity, or two with highest variance
                if len(numeric_cols) >= 2:
                    # Calculate variance of each feature
                    variances = clean_df.var().sort_values(ascending=False)
                    vis_cols = list(variances.index)[:2]
                else:
                    vis_cols = numeric_cols  # Just in case
                
                # Create visualizations
                fig = plt.figure(figsize=(15, 10))
                
                # Plot 1: Scatter plot of clusters
                plt.subplot(2, 2, 1)
                for i in range(k):
                    cluster_points = cluster_df[cluster_df['Cluster'] == i]
                    plt.scatter(cluster_points[vis_cols[0]], cluster_points[vis_cols[1]], 
                               label=f'Cluster {i}')
                               
                # Add cluster centers
                plt.scatter(centers_df[vis_cols[0]], centers_df[vis_cols[1]], 
                           s=200, marker='*', c='red', label='Centroids')
                           
                plt.title(f'Clusters in {vis_cols[0]} vs {vis_cols[1]}')
                plt.xlabel(vis_cols[0])
                plt.ylabel(vis_cols[1])
                plt.legend()
                
                # Plot 2: Box plots of features by cluster
                plt.subplot(2, 2, 2)
                
                # Reshape data for box plotting
                melt_df = pd.melt(cluster_df.reset_index(), 
                                 id_vars=['index', 'Cluster'], 
                                 value_vars=numeric_cols, 
                                 var_name='Feature', value_name='Value')
                                 
                sns.boxplot(x='Feature', y='Value', hue='Cluster', data=melt_df)
                plt.title('Feature Distributions by Cluster')
                plt.xticks(rotation=45)
                plt.legend(title='Cluster')
                
                # Plot 3: Cluster means heatmap
                plt.subplot(2, 2, 3)
                cluster_means = cluster_df.groupby('Cluster').mean()
                
                # Standardize for better visualization
                if not cluster_means.empty:
                    cluster_means_scaled = (cluster_means - cluster_means.mean()) / cluster_means.std()
                    sns.heatmap(cluster_means_scaled, cmap='coolwarm', annot=True, fmt=".2f")
                    plt.title('Standardized Cluster Means')
                
                # Plot 4: Parallel coordinates plot
                plt.subplot(2, 2, 4)
                
                # Create a dataframe for parallel coordinates with standardized values
                parallel_df = cluster_df.copy()
                
                # Standardize each feature
                for col in numeric_cols:
                    if parallel_df[col].std() > 0:  # Avoid division by zero
                        parallel_df[col] = (parallel_df[col] - parallel_df[col].mean()) / parallel_df[col].std()
                    
                # Plot each cluster
                for i in range(k):
                    cluster_data = parallel_df[parallel_df['Cluster'] == i]
                    if not cluster_data.empty:
                        cluster_means = cluster_data[numeric_cols].mean()
                        
                        # Convert to format for parallel coordinates
                        x = range(len(numeric_cols))
                        y = cluster_means.values
                        
                        plt.plot(x, y, 'o-', linewidth=2, label=f'Cluster {i}')
                        
                plt.title('Parallel Coordinates Plot of Cluster Means')
                plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
                plt.ylabel('Standardized Value')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                result['visualization'] = df_to_base64(fig)
                
                # Add cluster assignments to result for download
                cluster_assignments = pd.DataFrame({
                    'Row_Index': range(len(clusters)),
                    'Cluster': clusters
                })
                result['cluster_assignments_html'] = cluster_assignments.head(20).to_html(
                    classes='table table-striped', index=False)
                result['download_available'] = True
                
            except Exception as e:
                logger.error(f"Clustering analysis error: {str(e)}")
                return jsonify(success=False, message=f"Error in clustering analysis: {str(e)}")
            
        elif analysis_type == 'residual_diagnostics':
            # Get dependent variable
            y_col = request.form.get('dependent_var')
            
            if not y_col or y_col not in analysis_tool.numeric_columns:
                return jsonify(success=False, message='Please select a valid numeric dependent variable')
                
            # Use remaining selected columns as independent variables
            x_cols = [col for col in selected_columns if col in analysis_tool.numeric_columns and col != y_col]
            
            if not x_cols:
                return jsonify(success=False, message='Please select at least one numeric independent variable')
                
            # Create a clean dataframe
            analysis_cols = x_cols + [y_col]
            clean_df = cached_df[analysis_cols].dropna()
            
            if len(clean_df) < len(x_cols) + 1:
                return jsonify(success=False, message='Not enough data points after removing missing values')
                
            try:
                # Fit the model using statsmodels for detailed diagnostics
                X = sm.add_constant(clean_df[x_cols])
                y = clean_df[y_col]
                
                model = sm.OLS(y, X).fit()
                
                # Store model summary
                result['summary_html'] = model.summary().as_html()
                
                # Calculate predictions and residuals
                predictions = model.predict(X)
                residuals = model.resid
                standardized_residuals = model.get_influence().resid_studentized_internal
                
                # Run diagnostic tests
                # Normality test of residuals
                _, norm_p_value = stats.shapiro(residuals)
                result['shapiro_p_value'] = f"{norm_p_value:.4f}"
                result['normality_interpretation'] = f"Residuals are {'normally' if norm_p_value > 0.05 else 'not normally'} distributed."
                
                # Heteroscedasticity test
                try:
                    bp_lm, bp_p_value, _, _ = het_breuschpagan(residuals, X)
                    result['bp_p_value'] = f"{bp_p_value:.4f}"
                    result['heteroscedasticity_interpretation'] = f"{'Constant' if bp_p_value > 0.05 else 'Non-constant'} variance in residuals."
                except Exception as e:
                    logger.warning(f"Breusch-Pagan test error: {str(e)}")
                    result['bp_error'] = str(e)
                
                # Create diagnostic plots
                fig = plt.figure(figsize=(15, 12))
                
                # Plot 1: Residuals vs Fitted
                plt.subplot(2, 2, 1)
                plt.scatter(predictions, residuals, alpha=0.7)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.title('Residuals vs Fitted Values')
                plt.xlabel('Fitted Values')
                plt.ylabel('Residuals')
                plt.grid(True, linestyle="--", alpha=0.7)

                # Plot 2: QQ Plot
                plt.subplot(2, 2, 2)
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title('Normal Q-Q Plot')
                plt.grid(True, linestyle="--", alpha=0.7)

                # Plot 3: Scale-Location Plot (Square root of abs residuals vs Fitted)
                plt.subplot(2, 2, 3)
                plt.scatter(predictions, np.sqrt(np.abs(residuals)), alpha=0.7)
                plt.title('Scale-Location Plot')
                plt.xlabel('Fitted Values')
                plt.ylabel('$\\sqrt{|\\textrm{Standardized Residuals}|}')
                plt.grid(True, linestyle="--", alpha=0.7)

                # Plot 4: Histogram of residuals
                plt.subplot(2, 2, 4)
                plt.hist(residuals, bins=20, edgecolor='black')
                plt.title('Histogram of Residuals')
                plt.xlabel('Residuals')
                plt.ylabel('Frequency')
                plt.grid(True, linestyle="--", alpha=0.7)

                plt.tight_layout()
                result['visualization'] = df_to_base64(fig)

                # Display statistics about residuals
                residual_stats = pd.Series(residuals).describe()
                result['residual_stats_html'] = residual_stats.to_frame().to_html(classes='table table-striped')

                # Durbin-Watson test for autocorrelation
                try:
                    dw_stat = durbin_watson(residuals)
                    result['durbin_watson'] = f"{dw_stat:.4f}"
                    
                    if dw_stat < 1.5:
                        dw_interpretation = "Positive autocorrelation may be present."
                    elif dw_stat > 2.5:
                        dw_interpretation = "Negative autocorrelation may be present."
                    else:
                        dw_interpretation = "No significant autocorrelation detected."
                        
                    result['durbin_watson_interpretation'] = dw_interpretation
                except Exception as e:
                    logger.warning(f"Durbin-Watson test error: {str(e)}")
                    result['dw_error'] = str(e)
                    
                # Add diagnostic summary
                result['diagnostic_summary'] = """
                <div class="alert alert-info">
                    <strong>Residual Diagnostic Summary:</strong>
                    <ul>
                        <li><strong>Normality:</strong> {normality}</li>
                        <li><strong>Homoscedasticity:</strong> {homoscedasticity}</li>
                        <li><strong>Autocorrelation:</strong> {autocorrelation}</li>
                    </ul>
                </div>
                """.format(
                    normality=result.get('normality_interpretation', 'Test not available'),
                    homoscedasticity=result.get('heteroscedasticity_interpretation', 'Test not available'),
                    autocorrelation=result.get('durbin_watson_interpretation', 'Test not available')
                )
                
            except Exception as e:
                logger.error(f"Residual diagnostics error: {str(e)}")
                return jsonify(success=False, message=f"Error in residual diagnostics: {str(e)}")

        else:
            return jsonify(success=False, message='Unknown analysis type')
            
        return jsonify(success=True, result=result)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        return jsonify(success=False, message=f"Error: {str(e)}", traceback=traceback.format_exc())

@app.route('/download/<analysis_type>', methods=['POST'])
def download_results(analysis_type):
    """Download analysis results as CSV"""
    global cached_df
    
    if cached_df is None:
        return jsonify(success=False, message='No data loaded')
    
    try:
        data = request.json
        if not data:
            return jsonify(success=False, message='No data provided')
            
        # Export based on analysis type
        if analysis_type == 'clustering':
            # Get cluster assignments
            cluster_data = pd.DataFrame({
                'index': range(len(data['clusters'])),
                'cluster': data['clusters']
            })
            
            # Convert to CSV
            csv_data = cluster_data.to_csv(index=False)
            
            return jsonify(success=True, data=csv_data, filename='cluster_assignments.csv')
            
        elif analysis_type == 'pca':
            # Get PCA components
            components_data = pd.DataFrame(data['components'], columns=[f'PC{i+1}' for i in range(len(data['components'][0]))])
            
            # Convert to CSV
            csv_data = components_data.to_csv(index=False)
            
            return jsonify(success=True, data=csv_data, filename='pca_components.csv')
            
        else:
            return jsonify(success=False, message='Download not supported for this analysis type')
            
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify(success=False, message=f"Error: {str(e)}")

@app.route('/memory_cleanup', methods=['POST'])
def memory_cleanup():
    """Clear cached data to free memory"""
    global cached_df, analysis_tool
    
    try:
        cached_df = None
        analysis_tool = None
        import gc
        gc.collect()
        return jsonify(success=True, message='Memory cleared successfully')
    except Exception as e:
        logger.error(f"Memory cleanup error: {str(e)}")
        return jsonify(success=False, message=f"Error: {str(e)}")

@app.route('/get_memory_usage', methods=['GET'])
def get_memory_usage():
    """Get current memory usage info"""
    global cached_df
    
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        result = {
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'dataframe_memory_mb': cached_df.memory_usage(deep=True).sum() / (1024 * 1024) if cached_df is not None else 0,
            'system_available_memory_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
        
        return jsonify(success=True, memory_info=result)
    except Exception as e:
        logger.error(f"Memory usage check error: {str(e)}")
        return jsonify(success=False, message=f"Error: {str(e)}")

# Add error handler for internal server errors
@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify(success=False, message="Internal server error. Please check the logs for details."), 500

if __name__ == '__main__':
    app.run(debug=True)