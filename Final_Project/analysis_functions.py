# Standard Library
import os
import sys
import io
import base64
import tkinter as tk
from tkinter import filedialog
# Data Handling
import numpy as np
import pandas as pd
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Scikit-learn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, r2_score
# Scipy
from scipy import stats
from scipy.stats import chi2_contingency
# Statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class StatisticalAnalysisTool:
    def __init__(self):
        self.df = None
        self.selected_columns = None
        self.file_path = None
        self.numeric_columns = []
        self.categorical_columns = []

    def start(self):
        """Main entry point for the application"""
        print("\n--- Interactive Statistical Analysis Tool for CSV Data ---\n")
        self.load_csv_file()
        if self.df is not None:
            self.display_menu()

    def load_csv_file(self):
        """Open a file dialog to select and load a CSV file"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        self.file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not self.file_path:
            print("No file selected. Exiting.")
            return

        try:
            self.df = pd.read_csv(self.file_path)
            
            if self.df.empty:
                print("Error: The CSV file is empty.")
                self.df = None
                return
                
            print(f"\nLoaded file: {os.path.basename(self.file_path)}")
            print(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            #print("\nFirst 5 rows:")
            #print(self.df.head())
            
            # Categorize columns
            self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            print(f"\nNumeric columns: {len(self.numeric_columns)}")
            print(f"Categorical columns: {len(self.categorical_columns)}")
            
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            self.df = None
    
    def select_columns(self, message="Select columns:", numeric_only=False, 
                      categorical_only=False, min_selections=1, max_selections=None):
        """
        Prompt user to select columns from the dataframe
        
        Args:
            message: The prompt message
            numeric_only: If True, only show numeric columns
            categorical_only: If True, only show categorical columns
            min_selections: Minimum number of columns to select
            max_selections: Maximum number of columns to select
            
        Returns:
            list: Selected column names
        """
        available_columns = []
        
        if numeric_only:
            available_columns = self.numeric_columns
        elif categorical_only:
            available_columns = self.categorical_columns
        else:
            available_columns = self.df.columns.tolist()
            
        if not available_columns:
            print("No suitable columns available for selection.")
            return []
            
        # If there are predefined selected columns, use those that are available
        if self.selected_columns:
            filtered_columns = [col for col in self.selected_columns if col in available_columns]
            if filtered_columns:
                use_selected = input(f"Use previously selected columns ({', '.join(filtered_columns)})? (y/n): ")
                if use_selected.lower() == 'y':
                    return filtered_columns
        
        print(f"\n{message}")
        for i, col in enumerate(available_columns, 1):
            dtype = self.df[col].dtype
            print(f"{i}. {col} ({dtype})")
            
        while True:
            selection = input(f"\nEnter column numbers (comma-separated, {min_selections}-{max_selections or 'all'}): ")
            try:
                # Handle empty input
                if not selection.strip():
                    print(f"Please select at least {min_selections} column(s).")
                    continue
                    
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected = [available_columns[i] for i in indices if 0 <= i < len(available_columns)]
                
                if len(selected) < min_selections:
                    print(f"Please select at least {min_selections} column(s).")
                    continue
                    
                if max_selections and len(selected) > max_selections:
                    print(f"Please select at most {max_selections} column(s).")
                    continue
                    
                return selected
                
            except (ValueError, IndexError):
                print("Invalid selection. Please enter valid column numbers.")

    def descriptive_statistics(self):
        """Display descriptive statistics for selected numeric columns and optional row filtering."""
        # Step 1: Select numeric columns
        columns = self.select_columns(
            message="Select numeric columns for descriptive statistics:",
            numeric_only=True,
            min_selections=1
        )
        if not columns:
            return

        # Step 2: Optional row selection
        print(f"\nData has {len(self.df)} rows.")
        filter_rows = input("Do you want to filter by row range? (y/n): ").strip().lower()
        if filter_rows == 'y':
            try:
                start = int(input("Enter starting row index (e.g., 0): ").strip())
                end = int(input("Enter ending row index (exclusive, e.g., 100): ").strip())
                if start < 0 or end > len(self.df) or start >= end:
                    print("Invalid row range. Using full dataset instead.")
                    data = self.df[columns]
                else:
                    data = self.df.loc[start:end, columns]
            except ValueError:
                print("Invalid input. Using full dataset instead.")
                data = self.df[columns]
        else:
            data = self.df[columns]

        # Step 3: Compute statistics
        stats_df = data.describe().T
        stats_df['median'] = data.median()

        try:
            stats_df['mode'] = data.mode().iloc[0]
        except IndexError:
            stats_df['mode'] = np.nan

        stats_df['skewness'] = data.skew()
        stats_df['kurtosis'] = data.kurtosis()
        stats_df['missing'] = data.isna().sum()
        stats_df['missing_percent'] = (data.isna().sum() / len(data)) * 100

        print("\nDescriptive Statistics:")
        with pd.option_context('display.float_format', '{:.4f}'.format):
            print(stats_df)

        # Step 4: Visualization options
        print("\nVisualization options:")
        print("1. Histograms")
        print("2. Box plots")
        print("3. Both")
        print("4. None")

        vis_choice = input("Select visualization type (1-4): ").strip()

        if vis_choice not in ('1', '2', '3'):
            return

        num_plots = len(columns) * (2 if vis_choice == '3' else 1)
        plt.figure(figsize=(10, 5 * len(columns)))

        plot_index = 1
        for col in columns:
            if vis_choice in ('1', '3'):
                plt.subplot(num_plots, 1, plot_index)
                sns.histplot(data[col].dropna(), kde=True)
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plot_index += 1

            if vis_choice in ('2', '3'):
                plt.subplot(num_plots, 1, plot_index)
                sns.boxplot(x=data[col].dropna(), orient='h')
                plt.title(f'Box Plot of {col}')
                plt.xlabel(col)
                plot_index += 1

        plt.tight_layout()
        plt.show()

    def scatter_matrix(self):
        """Display scatter matrix for selected columns"""
        print("\n--- Scatter Matrix ---")
        
        # Select numeric columns for the scatter matrix
        columns = self.select_columns(message="Select numeric columns for scatter matrix:", 
        numeric_only=True, min_selections=2)
        
        if not columns or len(columns) < 2:
            print("At least 2 numeric columns are required for a scatter matrix.")
            return
            
        print(f"Creating scatter matrix for: {', '.join(columns)}")
        
        # Create scatter matrix
        plt.figure(figsize=(12, 10))
        sns.set_theme(style="ticks")
        
        scatter_plot = sns.pairplot(self.df[columns], diag_kind="kde", markers="o", 
                                   plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                                   height=2.5)
                                   
        plt.suptitle("Scatter Matrix with KDE Diagonal", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

    def linear_regression_single(self):
        """Perform simple linear regression analysis"""
        print("\n--- Linear Regression (Single) ---")
        
        # Check if there are enough numeric columns
        if len(self.numeric_columns) < 2:
            print("Linear regression requires at least 2 numeric columns.")
            return
            
        # Select columns for analysis
        y_col = self.select_columns(message="Select dependent variable (Y):", 
                                   numeric_only=True, min_selections=1, max_selections=1)[0]
                                   
        x_col = self.select_columns(message="Select independent variable (X):", 
                                   numeric_only=True, min_selections=1, max_selections=1)[0]
        
        # Create a clean dataframe by removing rows with NA in selected columns
        clean_df = self.df[[x_col, y_col]].dropna()
        
        if len(clean_df) < 2:
            print("Not enough data points after removing missing values.")
            return
            
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
        f_statistic = ((total_sum_of_squares - residual_sum_of_squares) / 1) / (residual_sum_of_squares / (n - 2))
        p_value = 1 - stats.f.cdf(f_statistic, 1, n - 2)
        
        # Output results
        print("\nLinear Regression Results:")
        print(f"Formula: {y_col} = {intercept:.4f} + {slope:.4f} * {x_col}")
        print(f"Slope (coefficient): {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Mean Squared Error: {mean_squared_error:.4f}")
        print(f"F-statistic: {f_statistic:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        # Visualize
        plt.figure(figsize=(15, 10))
        
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
        plt.show()

    def multiple_linear_regression(self):
        """Perform multiple linear regression analysis"""
        print("\n--- Multiple Linear Regression ---")
        
        # Check if there are enough numeric columns
        if len(self.numeric_columns) < 2:
            print("Multiple linear regression requires at least 2 numeric columns.")
            return
            
        # Select columns for analysis
        y_col = self.select_columns(message="Select dependent variable (Y):", 
                                   numeric_only=True, min_selections=1, max_selections=1)[0]
                                   
        x_cols = self.select_columns(message="Select independent variables (X):", 
                                    numeric_only=True, min_selections=1)
        
        # Check if Y is not in X
        if y_col in x_cols:
            x_cols.remove(y_col)
            
        if not x_cols:
            print("Need at least one independent variable.")
            return
            
        # Create a clean dataframe by removing rows with NA in selected columns
        analysis_cols = x_cols + [y_col]
        clean_df = self.df[analysis_cols].dropna()
        
        if len(clean_df) < len(x_cols) + 1:
            print("Not enough data points after removing missing values.")
            return
            
        # Check for multicollinearity using Variance Inflation Factor (VIF)
        if len(x_cols) > 1:
            X = sm.add_constant(clean_df[x_cols])
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            
            print("\nMulticollinearity Check (VIF):")
            print(vif_data)
            print("Note: VIF > 10 indicates high multicollinearity\n")
            
        # Fit the model using statsmodels for detailed statistics
        formula = f"{y_col} ~ {' + '.join(x_cols)}"
        model = ols(formula, data=clean_df).fit()
        
        # Output results
        print("\nMultiple Linear Regression Results:")
        print(model.summary())
        
        # For sklearn version
        X = clean_df[x_cols]
        y = clean_df[y_col]
        
        sk_model = LinearRegression()
        sk_model.fit(X, y)
        y_pred = sk_model.predict(X)
        residuals = y - y_pred
        
        # Visualize
        plt.figure(figsize=(15, 10))
        
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
        plt.show()

    def polynomial_regression(self):
        """Perform polynomial regression analysis"""
        print("\n--- Polynomial Regression ---")
        
        # Check if there are enough numeric columns
        if len(self.numeric_columns) < 2:
            print("Polynomial regression requires at least 2 numeric columns.")
            return
            
        # Select columns for analysis
        y_col = self.select_columns(message="Select dependent variable (Y):", 
                                   numeric_only=True, min_selections=1, max_selections=1)[0]
                                   
        x_col = self.select_columns(message="Select independent variable (X):", 
                                   numeric_only=True, min_selections=1, max_selections=1)[0]
        
        # Create a clean dataframe by removing rows with NA in selected columns
        clean_df = self.df[[x_col, y_col]].dropna()
        
        if len(clean_df) < 3:
            print("Not enough data points after removing missing values.")
            return
            
        # Get polynomial degree
        while True:
            try:
                degree = int(input("\nEnter polynomial degree (1-10): "))
                if 1 <= degree <= 10:
                    break
                else:
                    print("Please enter a value between 1 and 10.")
            except ValueError:
                print("Please enter a valid integer.")
        
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
        
        # Output results
        print("\nPolynomial Regression Results:")
        print(f"Polynomial Degree: {degree}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        
        # Display the polynomial equation
        coefficients = model.coef_
        intercept = model.intercept_
        
        equation = f"{y_col} = {intercept:.4f}"
        for i in range(1, degree + 1):
            equation += f" + {coefficients[i]:.4f} * {x_col}^{i}"
            
        print(f"\nEquation: {equation}")
        
        # Visualize
        plt.figure(figsize=(15, 10))
        
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
        plt.show()

    def logistic_regression(self):
        """Perform logistic regression analysis"""
        print("\n--- Logistic Regression ---")
        
        # Check if there are enough numeric columns
        if len(self.numeric_columns) < 1:
            print("Logistic regression requires at least 1 numeric column.")
            return
            
        # Select dependent variable (must be binary)
        print("\nSelect binary dependent variable (Y):")
        all_columns = self.df.columns.tolist()
        
        for i, col in enumerate(all_columns, 1):
            unique_vals = self.df[col].nunique()
            is_binary = unique_vals == 2
            print(f"{i}. {col} - Unique values: {unique_vals}{' (binary)' if is_binary else ''}")
            
        while True:
            try:
                choice = int(input("\nEnter column number for dependent variable: "))
                if 1 <= choice <= len(all_columns):
                    y_col = all_columns[choice - 1]
                    if self.df[y_col].nunique() != 2:
                        print("Warning: Selected column is not binary. Results may be unreliable.")
                        confirm = input("Continue anyway? (y/n): ")
                        if confirm.lower() != 'y':
                            return
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
                
        # Select independent variables
        x_cols = self.select_columns(message="Select independent variables (X):", 
                                    numeric_only=True, min_selections=1)
        
        # Check if Y is not in X
        if y_col in x_cols:
            x_cols.remove(y_col)
            
        if not x_cols:
            print("Need at least one independent variable.")
            return
            
        # Create a clean dataframe by removing rows with NA in selected columns
        analysis_cols = x_cols + [y_col]
        clean_df = self.df[analysis_cols].dropna()
        
        if len(clean_df) < len(x_cols) + 1:
            print("Not enough data points after removing missing values.")
            return
            
        # Check if we need to encode the target variable
        y = clean_df[y_col]
        if not pd.api.types.is_numeric_dtype(y):
            print(f"Encoding {y_col} into binary values...")
            unique_vals = y.unique()
            print(f"0 = {unique_vals[0]}, 1 = {unique_vals[1]}")
            y = pd.factorize(y)[0]
        else:
            y = y.values
            
        X = clean_df[x_cols].values
        
        # Fit the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # Calculate statistics
        class_report = classification_report(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        accuracy = model.score(X, y)
        
        # Output results
        print("\nLogistic Regression Results:")
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nCoefficients:")
        for i, col in enumerate(x_cols):
            print(f"{col}: {model.coef_[0][i]:.4f}")
        print(f"Intercept: {model.intercept_[0]:.4f}")
        
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        print("\nClassification Report:")
        print(class_report)
        
        # Visualize
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Confusion Matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Plot 2: ROC Curve
        if len(np.unique(y)) == 2:  # Only for binary classification
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
        coef_df = pd.DataFrame({'Feature': x_cols, 'Coefficient': model.coef_[0]})
        coef_df = coef_df.sort_values('Coefficient')
        
        sns.barplot(x='Coefficient', y='Feature', data=coef_df)
        plt.title('Feature Coefficients')
        plt.axvline(x=0, color='r', linestyle='-')
        
        plt.tight_layout()
        plt.show()

    def anova(self):
        """Perform Analysis of Variance (ANOVA)"""
        print("\n--- ANOVA (Analysis of Variance) ---")
        
        # Select dependent variable (numeric)
        y_col = self.select_columns(message="Select dependent variable (numeric):", 
                                   numeric_only=True, min_selections=1, max_selections=1)[0]
        
        # Select categorical variable for grouping
        categorical_cols = self.categorical_columns.copy()
        
        # Include numeric columns with low cardinality as potential categorical variables
        for col in self.numeric_columns:
            if self.df[col].nunique() <= 10 and col != y_col:
                categorical_cols.append(col)
                
        if not categorical_cols:
            print("No suitable categorical columns found for ANOVA.")
            return
            
        print("\nSelect categorical column for grouping:")
        for i, col in enumerate(categorical_cols, 1):
            unique_vals = self.df[col].nunique()
            print(f"{i}. {col} - Categories: {unique_vals}")
            
        while True:
            try:
                choice = int(input("\nEnter column number for categorical variable: "))
                if 1 <= choice <= len(categorical_cols):
                    cat_col = categorical_cols[choice - 1]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
                
        # Create a clean dataframe by removing rows with NA in selected columns
        clean_df = self.df[[y_col, cat_col]].dropna()
        
        if len(clean_df) < 3:
            print("Not enough data points after removing missing values.")
            return
            
        # Check if there are enough groups
        groups = clean_df[cat_col].unique()
        if len(groups) < 2:
            print("ANOVA requires at least 2 groups for comparison.")
            return
            
        # Prepare data for ANOVA
        groups_data = []
        labels = []
        
        for group in groups:
            group_data = clean_df[clean_df[cat_col] == group][y_col].values
            if len(group_data) > 0:
                groups_data.append(group_data)
                labels.append(str(group))
        
        # Perform one-way ANOVA
        f_statistic, p_value = stats.f_oneway(*groups_data)
        
        # Output results
        print("\nANOVA Results:")
        print(f"F-statistic: {f_statistic:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"\nSignificance: {'Significant' if p_value < 0.05 else 'Not Significant'} at 5% level")
        
        # Descriptive statistics per group
        print("\nDescriptive Statistics by Group:")
        group_stats = clean_df.groupby(cat_col)[y_col].agg(['count', 'mean', 'std', 'min', 'max'])
        print(group_stats)
        
        # Run Tukey's HSD post-hoc test if there are more than 2 groups
        if len(groups) > 2:
            print("\nTukey's HSD Post-hoc Test:")
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                tukey_results = pairwise_tukeyhsd(clean_df[y_col], clean_df[cat_col], alpha=0.05)
                print(tukey_results)
            except Exception as e:
                print(f"Could not perform Tukey's test: {str(e)}")
        
        # Visualize
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Box plot by group
        plt.subplot(2, 2, 1)
        sns.boxplot(x=cat_col, y=y_col, data=clean_df)
        plt.title(f'Box Plot of {y_col} by {cat_col}')
        plt.xticks(rotation=45 if len(str(groups[0])) > 3 else 0)
        
        # Plot 2: Violin plot by group
        plt.subplot(2, 2, 2)
        sns.violinplot(x=cat_col, y=y_col, data=clean_df)
        plt.title(f'Violin Plot of {y_col} by {cat_col}')
        plt.xticks(rotation=45 if len(str(groups[0])) > 3 else 0)
        
        # Plot 3: Bar plot with error bars
        plt.subplot(2, 2, 3)
        sns.barplot(x=cat_col, y=y_col, data=clean_df, ci=95)
        plt.title(f'Mean {y_col} by {cat_col} (with 95% CI)')
        plt.xticks(rotation=45 if len(str(groups[0])) > 3 else 0)
        
        # Plot 4: Strip plot overlaid on box plot
        plt.subplot(2, 2, 4)
        sns.boxplot(x=cat_col, y=y_col, data=clean_df, showmeans=True, 
                   meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"})
        sns.stripplot(x=cat_col, y=y_col, data=clean_df, jitter=True, 
                     alpha=0.4, color='black', size=4)
        plt.title(f'Box Plot with Data Points of {y_col} by {cat_col}')
        plt.xticks(rotation=45 if len(str(groups[0])) > 3 else 0)
        
        plt.tight_layout()
        plt.show()

    def chi_square_test(self):
        """Perform Chi-Square Test of Independence"""
        print("\n--- Chi-Square Test of Independence ---")
        
        # Need at least 2 categorical columns
        if len(self.categorical_columns) < 2:
            print("Chi-Square test requires at least 2 categorical columns.")
            
            # Check if there are numeric columns with low cardinality that could be used
            potential_cats = []
            for col in self.numeric_columns:
                if self.df[col].nunique() <= 10:
                    potential_cats.append(col)
                    
            if len(self.categorical_columns) + len(potential_cats) < 2:
                print("Not enough suitable categorical columns found.")
                return
            else:
                print("Will include numeric columns with 10 or fewer unique values as categorical.")
                
        # Select two categorical columns
        cat_cols = self.categorical_columns.copy()
        
        # Include numeric columns with low cardinality
        for col in self.numeric_columns:
            if self.df[col].nunique() <= 10:
                cat_cols.append(col)
                
        print("\nSelect first categorical variable:")
        for i, col in enumerate(cat_cols, 1):
            unique_vals = self.df[col].nunique()
            print(f"{i}. {col} - Categories: {unique_vals}")
            
        while True:
            try:
                choice1 = int(input("\nEnter column number for first variable: "))
                if 1 <= choice1 <= len(cat_cols):
                    cat_col1 = cat_cols[choice1 - 1]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
                
        print("\nSelect second categorical variable:")
        for i, col in enumerate(cat_cols, 1):
            if col != cat_col1:  # Don't show the first selected column
                unique_vals = self.df[col].nunique()
                print(f"{i}. {col} - Categories: {unique_vals}")
            else:
                print(f"{i}. {col} - Already selected")
                
        while True:
            try:
                choice2 = int(input("\nEnter column number for second variable: "))
                if 1 <= choice2 <= len(cat_cols) and cat_cols[choice2 - 1] != cat_col1:
                    cat_col2 = cat_cols[choice2 - 1]
                    break
                elif cat_cols[choice2 - 1] == cat_col1:
                    print("Please select a different column than the first one.")
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
                
        # Create a clean dataframe by removing rows with NA in selected columns
        clean_df = self.df[[cat_col1, cat_col2]].dropna()
        
        if len(clean_df) < 5:
            print("Not enough data points after removing missing values.")
            return
            
        # Create contingency table
        contingency_table = pd.crosstab(clean_df[cat_col1], clean_df[cat_col2])
        print("\nContingency Table (Observed Frequencies):")
        print(contingency_table)
        
        # Calculate row and column totals
        row_totals = contingency_table.sum(axis=1)
        col_totals = contingency_table.sum(axis=0)
        total = contingency_table.sum().sum()
        
        # Calculate expected frequencies
        expected = np.outer(row_totals, col_totals) / total
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        
        print("\nExpected Frequencies:")
        print(expected_df)
        
        # Check assumptions
        expected_less_than_5 = (expected_df < 5).sum().sum()
        total_cells = expected_df.size
        
        if expected_less_than_5 / total_cells > 0.2:
            print("\nWarning: More than 20% of cells have expected frequencies less than 5.")
            print("Chi-square test may not be reliable.")
            
        # Perform Chi-Square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Output results
        print("\nChi-Square Test Results:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"p-value: {p:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"\nSignificance: {'Significant' if p < 0.05 else 'Not Significant'} at 5% level")
        
        if p < 0.05:
            print("\nInterpretation: There is a significant association between the variables.")
        else:
            print("\nInterpretation: There is no significant association between the variables.")
            
        # Calculate Cramer's V for effect size
        n = total
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        print(f"\nCramer's V (Effect Size): {cramers_v:.4f}")
        
        # Visualize
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Heatmap of observed frequencies
        plt.subplot(2, 2, 1)
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Observed Frequencies')
        
        # Plot 2: Heatmap of expected frequencies
        plt.subplot(2, 2, 2)
        sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='YlGnBu')
        plt.title('Expected Frequencies')
        
        # Plot 3: Mosaic plot (using seaborn's catplot)
        plt.subplot(2, 2, 3)
        
        # Reshape data for plotting
        plot_data = clean_df.copy()
        plot_data['count'] = 1
        
        sns.countplot(x=cat_col1, hue=cat_col2, data=plot_data)
        plt.title('Count Plot')
        plt.xticks(rotation=45 if len(contingency_table.index) > 3 else 0)
        plt.legend(title=cat_col2)
        
        # Plot 4: Stacked percentage bar chart
        plt.subplot(2, 2, 4)
        
        # Calculate percentages
        percentage_table = contingency_table.div(row_totals, axis=0) * 100
        percentage_table.plot(kind='bar', stacked=True)
        plt.title('Percentage Bar Chart')
        plt.xlabel(cat_col1)
        plt.ylabel('Percentage')
        plt.xticks(rotation=45 if len(contingency_table.index) > 3 else 0)
        plt.legend(title=cat_col2)
        
        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self):
        """Display correlation heatmap for selected columns"""
        print("\n--- Correlation Heatmap ---")
        
        # Check if there are enough numeric columns
        if len(self.numeric_columns) < 2:
            print("Correlation heatmap requires at least 2 numeric columns.")
            return
            
        # Select columns for analysis
        columns = self.select_columns(message="Select numeric columns for correlation analysis:", 
                                     numeric_only=True, min_selections=2)
        
        if not columns or len(columns) < 2:
            print("At least 2 numeric columns are required for correlation analysis.")
            return
            
        # Create a clean dataframe by removing rows with NA in selected columns
        clean_df = self.df[columns].dropna()
        
        if len(clean_df) < 2:
            print("Not enough data points after removing missing values.")
            return
            
        # Calculate correlation matrix
        correlation_matrix = clean_df.corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        # Select correlation method
        print("\nSelect correlation method:")
        print("1. Pearson (linear relationship)")
        print("2. Spearman (monotonic relationship)")
        print("3. Kendall (ordinal relationship)")
        
        method_choice = input("\nEnter choice (1-3), default is Pearson: ") or "1"
        
        method_map = {"1": "pearson", "2": "spearman", "3": "kendall"}
        corr_method = method_map.get(method_choice, "pearson")
        
        # Recalculate with selected method if not default
        if corr_method != "pearson":
            correlation_matrix = clean_df.corr(method=corr_method)
            print(f"\n{corr_method.capitalize()} Correlation Matrix:")
            print(correlation_matrix)
        
        # Visualize
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', annot=True, 
                   fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
                   
        plt.title(f'{corr_method.capitalize()} Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def pca_analysis(self):
        """Perform Principal Component Analysis (PCA)"""
        print("\n--- Principal Component Analysis (PCA) ---")
        
        # Check if there are enough numeric columns
        if len(self.numeric_columns) < 2:
            print("PCA requires at least 2 numeric columns.")
            return
            
        # Select columns for analysis
        columns = self.select_columns(message="Select numeric columns for PCA:", 
                                     numeric_only=True, min_selections=2)
        
        if not columns or len(columns) < 2:
            print("At least 2 numeric columns are required for PCA.")
            return
            
        # Create a clean dataframe by removing rows with NA in selected columns
        clean_df = self.df[columns].dropna()
        
        if len(clean_df) < 2:
            print("Not enough data points after removing missing values.")
            return
            
        # Standardize the data (mean=0, std=1)
        X = clean_df.values
        X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Perform PCA
        n_components = min(len(columns), len(clean_df))
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)
        
        # Component loadings
        loadings = pca.components_
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Output results
        print("\nPCA Results:")
        print(f"Number of components: {n_components}")
        
        print("\nExplained Variance Ratio:")
        for i, var in enumerate(explained_variance, 1):
            print(f"PC{i}: {var:.4f} ({var*100:.2f}%)")
            
        print("\nCumulative Explained Variance:")
        for i, cum_var in enumerate(cumulative_variance, 1):
            print(f"PC1-PC{i}: {cum_var:.4f} ({cum_var*100:.2f}%)")
            
        print("\nComponent Loadings (eigenvectors):")
        loadings_df = pd.DataFrame(loadings.T, index=columns, 
                                  columns=[f'PC{i+1}' for i in range(n_components)])
        print(loadings_df.head())
        
        # Visualize
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Scree plot
        plt.subplot(2, 2, 1)
        plt.bar(range(1, n_components + 1), explained_variance)
        plt.plot(range(1, n_components + 1), cumulative_variance, 'ro-')
        plt.axhline(y=0.8, color='r', linestyle='--')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, min(n_components, 10) + 1))
        
        # Plot 2: Biplot (first two components)
        plt.subplot(2, 2, 2)
        
        # Scale factors for visualization
        scale_x = 1.0/(loadings[0].max() - loadings[0].min())
        scale_y = 1.0/(loadings[1].max() - loadings[1].min())
        
        # Plot data points
        plt.scatter(components[:, 0], components[:, 1], alpha=0.5)
        
        # Plot feature vectors
        for i, col in enumerate(columns):
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
                   xticklabels=columns, yticklabels=[f'PC{i+1}' for i in range(n_display)])
        plt.title('Component Loadings Heatmap')
        plt.tight_layout()
        
        # Plot 4: Scatter plot of first two components with additional info
        if len(components) > 10:
            plt.subplot(2, 2, 4)
            
            # Use density plot if many points
            sns.kdeplot(x=components[:, 0], y=components[:, 1], cmap="Blues", fill=True)
            plt.title('Density of PC1 vs PC2')
            plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
        else:
            plt.subplot(2, 2, 4)
            plt.scatter(components[:, 0], components[:, 1])
            
            # Add indices for each point
            for i, (x, y) in enumerate(zip(components[:, 0], components[:, 1])):
                plt.text(x, y, str(i), fontsize=9)
                
            plt.title('PC1 vs PC2 with Indices')
            plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def kmeans_clustering(self):
        """Perform K-means clustering analysis"""
        print("\n--- K-means Clustering ---")
        
        # Check if there are enough numeric columns
        if len(self.numeric_columns) < 2:
            print("K-means clustering requires at least 2 numeric columns.")
            return
            
        # Select columns for analysis
        columns = self.select_columns(message="Select numeric columns for clustering:", 
                                     numeric_only=True, min_selections=2)
        
        if not columns or len(columns) < 2:
            print("At least 2 numeric columns are required for clustering.")
            return
            
        # Create a clean dataframe by removing rows with NA in selected columns
        clean_df = self.df[columns].dropna()
        
        if len(clean_df) < 2:
            print("Not enough data points after removing missing values.")
            return
            
        # Get number of clusters (k)
        while True:
            try:
                k = int(input("\nEnter number of clusters (k) between 2 and 10: "))
                if 2 <= k <= 10:
                    break
                else:
                    print("Please enter a value between 2 and 10.")
            except ValueError:
                print("Please enter a valid integer.")
                
        # Standardize the data
        X = clean_df.values
        X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Perform K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        cluster_df = clean_df.copy()
        cluster_df['Cluster'] = clusters
        
        # Output results
        print("\nK-means Clustering Results:")
        print(f"Number of clusters: {k}")
        
        print("\nCluster Sizes:")
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        for i, size in enumerate(cluster_sizes):
            print(f"Cluster {i}: {size} points ({size/len(clusters)*100:.1f}%)")
            
        print("\nCluster Centers (scaled features):")
        centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=columns)
        centers_df.index.name = 'Cluster'
        print(centers_df)
        
        # Calculate cluster statistics
        print("\nCluster Statistics:")
        cluster_stats = cluster_df.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
        print(cluster_stats)
        
        # Determine 2 features to visualize
        print("\nSelect two features for visualization:")
        for i, col in enumerate(columns, 1):
            print(f"{i}. {col}")
            
        vis_indices = []
        for i in range(2):
            while True:
                try:
                    idx = int(input(f"\nSelect feature {i+1} (1-{len(columns)}): "))
                    if 1 <= idx <= len(columns):
                        vis_indices.append(idx - 1)
                        break
                    else:
                        print(f"Please enter a value between 1 and {len(columns)}.")
                except ValueError:
                    print("Please enter a valid integer.")
        
        vis_cols = [columns[i] for i in vis_indices]
        
        # Visualize
        plt.figure(figsize=(15, 10))
        
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
                         value_vars=columns, 
                         var_name='Feature', value_name='Value')
                         
        sns.boxplot(x='Feature', y='Value', hue='Cluster', data=melt_df)
        plt.title('Feature Distributions by Cluster')
        plt.xticks(rotation=45)
        plt.legend(title='Cluster')
        
        # Plot 3: Cluster means heatmap
        plt.subplot(2, 2, 3)
        cluster_means = cluster_df.groupby('Cluster').mean()
        
        # Standardize for better visualization
        cluster_means_scaled = (cluster_means - cluster_means.mean()) / cluster_means.std()
        
        sns.heatmap(cluster_means_scaled, cmap='coolwarm', annot=True, fmt=".2f")
        plt.title('Standardized Cluster Means')
        
        # Plot 4: Parallel coordinates plot
        plt.subplot(2, 2, 4)
        
        # Create a dataframe for parallel coordinates with standardized values
        parallel_df = cluster_df.copy()
        
        # Standardize each feature
        for col in columns:
            parallel_df[col] = (parallel_df[col] - parallel_df[col].mean()) / parallel_df[col].std()
            
        # Plot each cluster
        for i in range(k):
            cluster_data = parallel_df[parallel_df['Cluster'] == i]
            cluster_means = cluster_data[columns].mean()
            
            # Convert to format for parallel coordinates
            x = range(len(columns))
            y = cluster_means.values
            
            plt.plot(x, y, 'o-', linewidth=2, label=f'Cluster {i}')
            
        plt.title('Parallel Coordinates Plot of Cluster Means')
        plt.xticks(range(len(columns)), columns, rotation=45)
        plt.ylabel('Standardized Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Offer to save cluster assignments
        save_clusters = input("\nSave cluster assignments back to dataframe? (y/n): ")
        if save_clusters.lower() == 'y':
            # Create a mapping from clean_df indices to original df indices
            cluster_mapping = dict(zip(clean_df.index, clusters))
            
            # Create a new column in the original dataframe
            col_name = 'KMeans_Cluster'
            counter = 1
            while col_name in self.df.columns:
                col_name = f'KMeans_Cluster_{counter}'
                counter += 1
                
            self.df[col_name] = np.nan
            
            # Assign cluster values to corresponding rows
            for idx, cluster in cluster_mapping.items():
                self.df.loc[idx, col_name] = cluster
                
            print(f"\nCluster assignments saved to column '{col_name}'")

    def residual_diagnostics(self):
        """Perform residual diagnostics on a regression model"""
        print("\n--- Residual Diagnostics ---")
        y_pred = model.predict(X)
        # Check if there are enough numeric columns
        if len(self.numeric_columns) < 2:
            print("Residual diagnostics requires at least 2 numeric columns.")
            return
            
        # Select dependent variable
        y_col = self.select_columns(message="Select dependent variable (Y):", 
                                   numeric_only=True, min_selections=1, max_selections=1)[0]
                                   
        # Select independent variables
        x_cols = self.select_columns(message="Select independent variables (X):", 
                                    numeric_only=True, min_selections=1)
        
        # Check if Y is not in X
        if y_col in x_cols:
            x_cols.remove(y_col)
            
        if not x_cols:
            print("Need at least one independent variable.")
            return
            
        # Create a clean dataframe by removing rows with NA in selected columns
        analysis_cols = x_cols + [y_col]
        clean_df = self.df[analysis_cols].dropna()
        
        if len(clean_df) < len(x_cols) + 1:
            print("Not enough data points after removing missing values.")
            return
            
        # Fit the model using statsmodels for detailed diagnostics
        X = sm.add_constant(clean_df[x_cols])
        y = clean_df[y_col]
        
        model = sm.OLS(y, X).fit()
        
        # Output model summary
        print("\nRegression Model Summary:")
        print(model.summary())
        
        # Calculate predictions and residuals
        predictions = model.predict(X)
        residuals = model.resid
        standardized_residuals = model.get_influence().resid_studentized_internal
        
        # Print diagnostic tests
        print("\nDiagnostic Tests:")
        
        # Normality test of residuals
        _, norm_p_value = stats.shapiro(residuals)
        print(f"Shapiro-Wilk Normality Test: p = {norm_p_value:.4f}")
        print(f"Interpretation: Residuals are {'normally' if norm_p_value > 0.05 else 'not normally'} distributed.")
        
        # Heteroscedasticity test
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_lm, bp_p_value, _, _ = het_breuschpagan(residuals, X)
            print(f"\nBreusch-Pagan Test for Heteroscedasticity: p = {bp_p_value:.4f}")
            print(f"Interpretation: {'Constant' if bp_p_value > 0.05 else 'Non-constant'} variance in residuals.")
        except Exception as e:
            print(f"Could not perform Breusch-Pagan test: {str(e)}")
        
        # Create diagnostic plots
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Residuals vs Fitted
        plt.subplot(2, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residuals vs Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.grid(True, linestyle="--", alpha=0.7)

        # Plot 2: QQ Plot
        plt.subplot(2, 2, 2)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot')
        plt.grid(True, linestyle="--", alpha=0.7)

        # Plot 3: Scale-Location Plot (Square root of abs residuals vs Fitted)
        plt.subplot(2, 2, 3)
        plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.7)
        plt.title('Scale-Location Plot')
        plt.xlabel('Fitted Values')
        plt.ylabel('$\\sqrt{|\\textrm{Standardized Residuals}|}$')
        plt.grid(True, linestyle="--", alpha=0.7)

        # Plot 4: Histogram of residuals
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=20, edgecolor='black')
        plt.title('Histogram of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

        # Display statistics about residuals
        print("\nResidual Statistics:")
        residual_stats = pd.Series(residuals).describe()
        print(residual_stats)

        # Durbin-Watson test for autocorrelation
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw_stat = durbin_watson(residuals)
            print(f"\nDurbin-Watson Statistic: {dw_stat:.4f}")
            print("Interpretation:")
            if dw_stat < 1.5:
                print("Positive autocorrelation may be present.")
            elif dw_stat > 2.5:
                print("Negative autocorrelation may be present.")
            else:
                print("No significant autocorrelation detected.")
        except Exception as e:
            print(f"Could not perform Durbin-Watson test: {str(e)}")

        # Shapiro-Wilk test for normality of residuals
        try:
            _, normality_p_value = stats.shapiro(residuals)
            print(f"\nShapiro-Wilk Test for Normality: p = {normality_p_value:.4f}")
            print(f"Interpretation: Residuals {'appear' if normality_p_value > 0.05 else 'do not appear'} to be normally distributed.")
        except Exception as e:
            print(f"Could not perform Shapiro-Wilk test: {str(e)}")

        # Return to main menu
        input("\nPress Enter to return to the main menu...")

    def display_menu(self):
            """Display the main menu and handle user input"""
            while True:
                print("\n--- MAIN MENU ---")
                print("1. View Descriptive Statistics")
                print("2. Scatter Matrix")
                print("3. Linear Regression (Single)")
                print("4. Multiple Linear Regression")
                print("5. Polynomial Regression")
                print("6. Logistic Regression")
                print("7. ANOVA")
                print("8. Chi-Square Test of Independence")
                print("9. Correlation Heatmap")
                print("10. PCA (Principal Component Analysis)")
                print("11. Clustering (KMeans)")
                print("12. Residual Diagnostics")
                print("13. Select Specific Columns for Analysis")
                print("0. Exit Program")
                
                choice = input("\nEnter your choice (1-13) or 0 to exit: ")
                
                try:
                    choice = int(choice)
                    if choice == 1:
                        self.descriptive_statistics()
                    elif choice == 2:
                        self.scatter_matrix()
                    elif choice == 3:
                        self.linear_regression_single()
                    elif choice == 4:
                        self.multiple_linear_regression()
                    elif choice == 5:
                        self.polynomial_regression()
                    elif choice == 6:
                        self.logistic_regression()
                    elif choice == 7:
                        self.anova()
                    elif choice == 8:
                        self.chi_square_test()
                    elif choice == 9:
                        self.correlation_heatmap()
                    elif choice == 10:
                        self.pca_analysis()
                    elif choice == 11:
                        self.kmeans_clustering()
                    elif choice == 12:
                        self.residual_diagnostics()
                    elif choice == 13:
                        self.select_columns()
                    elif choice == 0:
                        print("Exiting program. Goodbye!")
                        sys.exit(0)
                    else:
                        print("Invalid choice. Please enter a number between 1 and 16.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                
                input("\nPress Enter to return to the main menu...")
       
def run_analysis(df, params):
    df = df.dropna()
    result = {}

    atype = params['analysis_type']
    cols = params['columns']
    dep = params['dependent_var']
    degree = params.get('degree', 2)
    var1 = params.get('var1')
    var2 = params.get('var2')

    if atype == 'descriptive_statistics':
        result['statistics'] = df[cols].describe().to_dict()

    elif atype == 'scatter_matrix':
        sns.set_theme(style="ticks")
        fig = sns.pairplot(df[cols])
        result['scatter_matrix'] = [{'title': 'Scatter Matrix', 'image': df_to_base64(fig.fig)}]

    elif atype in ['linear_regression', 'multiple_regression']:
        X = df[cols]
        y = df[dep]
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Linear Regression' if atype == 'linear_regression' else 'Multiple Regression')
        result[atype] = [{'title': ax.get_title(), 'image': df_to_base64(fig)}]

    elif atype == 'polynomial_regression':
        X = df[cols]
        y = df[dep]
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Polynomial Regression (Degree {degree})')
        result[atype] = [{'title': ax.get_title(), 'image': df_to_base64(fig)}]

    elif atype == 'logistic_regression':
        X = df[cols]
        y = df[dep]
        model = LogisticRegression(max_iter=200).fit(X, y)
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        result[atype] = report

    elif atype == 'anova':
        model = stats.f_oneway(*(df.groupby(dep)[col].values for col in cols))
        result[atype] = {'F-statistic': model.statistic, 'p-value': model.pvalue}

    elif atype == 'chi_square':
        contingency = pd.crosstab(df[var1], df[var2])
        chi2, p, dof, expected = chi2_contingency(contingency)
        result[atype] = {'chi2': chi2, 'p': p, 'dof': dof, 'expected_freq': expected.tolist()}

    elif atype == 'correlation':
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        result[atype] = [{'title': 'Correlation Heatmap', 'image': df_to_base64(fig)}]

    elif atype == 'pca':
        X = df[cols]
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots()
        ax.scatter(components[:, 0], components[:, 1])
        ax.set_title('PCA: First 2 Components')
        result[atype] = [{'title': ax.get_title(), 'image': df_to_base64(fig)}]

    elif atype == 'clustering':
        X = df[cols]
        kmeans = KMeans(n_clusters=3, n_init='auto')
        clusters = kmeans.fit_predict(X)
        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
        ax.set_title('KMeans Clustering')
        result[atype] = [{'title': ax.get_title(), 'image': df_to_base64(fig)}]

    elif atype == 'residual_diagnostics':
        X = df[cols]
        y = df[dep]
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title('Residuals Distribution')
        result[atype] = [{'title': ax.get_title(), 'image': df_to_base64(fig)}]

    else:
        raise ValueError("Unknown analysis type")

    return result

def df_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
#Run the APP#
if __name__ == "__main__":
    app = StatisticalAnalysisTool()
app.start()