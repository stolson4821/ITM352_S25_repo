Statistics Analysis Program:
statistical-analysis-tool/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # HTML file
└── analysis_functions.py  # Existing Python functions

Below is the preffered data columns that ensure a result from the functions within the program. This is provided because many may not know whay type of data can and should be used for each analysis type.

Test case with the Dataset.csv is:
1. View Descriptive Statistics
    Columns:
    ➔ Age, Income, Credit_Score, Purchase_Amount, Satisfaction_Score, Website_Visit_Frequency, Time_on_Website_Minutes, Spending_Score, Polynomial_Target

    These are continuous and ideal for measures like mean, std dev, min, max.

2. Scatter Matrix
    Columns:
    ➔ Age, Income, Credit_Score, Purchase_Amount, Satisfaction_Score, Spending_Score

    Scatter plots will show visible relationships, especially between Income and Spending_Score.

3. Linear Regression (Single)
    Target: Spending_Score

    Predictor: Income

    Strong linear relationship because Spending_Score was generated with Income as a major driver.

4. Multiple Linear Regression
    Target: Spending_Score

    Predictors:
    ➔ Income, Website_Visit_Frequency, Is_Returning_Customer

    Good realistic multivariable relationship built into the data generation.

5. Polynomial Regression
    Target: Polynomial_Target

    Predictor: Age

    Polynomial_Target was explicitly generated as a quadratic function of Age.

6. Logistic Regression
    Target: Churn (0/1)

    Predictors:
    ➔ Income, Credit_Score, Satisfaction_Score, Website_Visit_Frequency

    Realistic for churn prediction modeling.

7. ANOVA (Analysis of Variance)
    Groups: Membership_Status

    Continuous Variable: Purchase_Amount

    Compare mean purchase amounts across membership levels (None, Silver, Gold, Platinum).

8. Chi-Square Test
    Variables:
    ➔ Region vs Product_Category

    Both are categorical. Test if product categories are distributed differently by region.

9. Correlation Heatmap
    Columns:
    ➔ Age, Income, Credit_Score, Purchase_Amount, Satisfaction_Score, Website_Visit_Frequency, Time_on_Website_Minutes, Spending_Score

    Will show strong and weak correlations, especially Income vs Spending_Score.

10. PCA (Principal Component Analysis)
    Columns:
    ➔ Age, Income, Credit_Score, Purchase_Amount, Website_Visit_Frequency, Time_on_Website_Minutes

    Standardized numeric data ideal for PCA dimensionality reduction.

11. Clustering (KMeans)
    Columns:
    ➔ Income, Credit_Score, Purchase_Amount, Satisfaction_Score

    Natural customer segments should emerge.

12. Residual Diagnostics (for regression residuals)
    Model:
    ➔ Predicting Spending_Score from Income, Website_Visit_Frequency, Is_Returning_Customer

    Check for normality, homoscedasticity, and independence of residuals.