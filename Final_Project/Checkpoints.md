# Checkpoints - Updated April 29,2025

## 1. Technical Requirements

### 1.1 Data Handling and Input Validation
The application must be able to accept CSV files as input and validate the file format. Ensure proper error handling if the file is missing required columns or is formatted incorrectly. The user should receive clear feedback if the upload fails.

### 1.2 Data Preprocessing and Cleaning
Before performing any statistical analysis, the application must clean the data. This includes handling missing values, removing duplicates, and ensuring that columns with categorical data are correctly formatted.

### 1.3 Descriptive Statistics Calculation
The application must compute the mean, standard deviation, minimum, and maximum values for all continuous columns in the dataset (e.g., Age, Income, Satisfaction Score). The result should be presented in a clear, tabular format.

### 1.4 Support for Multiple Statistical Tests
The application must support the following statistical tests:
- ANOVA (Analysis of Variance) for comparing means of different groups.
- Chi-Square Test for testing independence between categorical variables.
- Correlation Analysis to evaluate relationships between continuous variables.
- Linear Regression for modeling and prediction.

### 1.5 Regression Analysis (Linear, Multiple, Polynomial)
The application must provide regression analysis capabilities:
- Single Linear Regression: Use a single predictor to predict the target variable.
- Multiple Linear Regression: Use multiple predictors for a target variable.
- Polynomial Regression: Apply polynomial regression for data that exhibits non-linear patterns.

### 1.6 Visualization of Results
The application must generate visualizations for key statistical outputs:
- Descriptive Stats: Display histograms, boxplots, or bar charts for numerical features.
- Scatter Plots: For exploring relationships between pairs of continuous variables.
- Heatmaps: Display correlation matrices between variables.
- Regression Plots: Show the fit line for linear and polynomial regressions.

### 1.7 Principal Component Analysis (PCA)
The application should perform PCA for dimensionality reduction on a dataset with multiple continuous variables and provide a visualization of the explained variance ratio for each principal component.

### 1.8 Clustering (KMeans)
The application must implement KMeans Clustering to identify natural groups within the dataset based on selected features (e.g., Income, Credit_Score, etc.). It should display the resulting clusters in a 2D scatter plot.

### 1.9 Logistic Regression for Classification
The application must support Logistic Regression for predicting a binary outcome, such as Churn (0/1), using relevant predictors such as Income, Satisfaction_Score, and Website_Visit_Frequency.

### 1.10 Model Diagnostics and Residual Analysis
After performing regression analysis, the application should include residual diagnostics to evaluate the quality of the model:
- Normality of Residuals: Use a histogram or Q-Q plot to assess the distribution of residuals.
- Homoscedasticity: Check for constant variance of residuals.
- Independence: Ensure that residuals are not autocorrelated.

### 1.11 User Interface (UI)
The application should have a user-friendly web interface built using Flask. Users should be able to:
- Upload datasets via an intuitive interface.
- Select which analysis to run via dropdowns or checkboxes.
- View results and visualizations in a well-organized format.

### 1.12 Output and Result Export
The application must allow users to export results from any analysis (e.g., regression coefficients, ANOVA results) in CSV or Excel format for further examination or reporting.

### 1.13 Performance and Scalability
The application should handle large datasets (with hundreds or thousands of rows) without crashing or significant performance degradation. Efficient algorithms should be implemented for both statistical analysis and data visualization.

### 1.14 Error Handling and Logging
The application must have proper error handling throughout the codebase:
- If an error occurs during data processing, statistical analysis, or model fitting, the application should log the error and display a user-friendly message.
- The application should include logging mechanisms for debugging purposes.

### 1.15 Testing and Debugging
The application should include comprehensive unit tests to ensure that each statistical function and data preprocessing step works correctly. Additionally, integration tests should verify that the overall system functions as expected when combining data upload, analysis, and visualization components.

## 2. Software Design

### Libraries:
- **Flask**: Web framework for creating the application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For mathematical operations.
- **Matplotlib/Seaborn/Plotly**: For data visualizations.
- **Scikit-learn**: For regression models, PCA, and clustering.
- **SciPy**: For statistical tests (e.g., ANOVA, Chi-Square).
- **Statsmodels**: For statistical modeling and regression diagnostics.

### Functions:
- Data cleaning functions (e.g., handling missing values, data transformations).
- Descriptive statistics function (mean, std, min, max).
- Regression functions (linear regression, multiple regression, polynomial regression).
- Statistical test functions (ANOVA, Chi-Square).
- PCA and Clustering: Functions to apply PCA for dimensionality reduction and KMeans for clustering.
- Visualization functions: For generating scatter plots, heatmaps, and other visual outputs.

### UI Components:
- File upload section: Allow users to upload CSV datasets.
- Selection options: Dropdown menus to select analysis types, target variables, predictors, etc.
- Results display: Tables for statistical outputs and embedded charts for visualizations.

## 3. UI Sketch

### Main Screen Layout:
- Header: "Statistics Analysis Program" with options to upload data, select analysis type, and view results.
- Data Upload Section: A simple file input for CSV files.
- Analysis Selection Section: Dropdowns for analysis type (e.g., Descriptive Stats, Regression), columns to use for targets and predictors.
- Submit Button: To run the selected analysis.

### Results Section:
- Display tables for Descriptive Stats, Regression Coefficients, etc.
- Display interactive charts (e.g., correlation heatmap, scatter plots).

## 4. Testing Plan

### Unit Testing:
- **Data Loading**: Ensure that CSV files are correctly loaded, parsed, and errors are handled for missing or incorrect formats.
- **Descriptive Stats**: Test that the mean, std, min, and max are calculated correctly for different columns.
- **Regression Models**: Test that the regression models output coefficients, p-values, and R-squared correctly.
- **Statistical Tests**: Ensure that ANOVA, Chi-Square tests, etc., return correct p-values based on mock datasets.

### Integration Testing:
- **UI Testing**: Verify that the file upload, analysis selection, and result display features work smoothly together.
- **Visualization Testing**: Ensure that scatter plots, heatmaps, and regression plots are rendered correctly with real data.

### Edge Case Testing:
- Empty or invalid datasets (missing values, wrong format).
- Test with different numbers of predictors and targets (e.g., single vs. multiple predictors).
- Large datasets for performance and scalability testing.

### Regression Testing: 
After each new feature or fix, test the application as a whole to ensure no other functionality has been broken.

*Note: Project plan is in the ReadMe file.*
    Unit Testing:

    Data Loading: Ensure that CSV files are correctly loaded, parsed, and errors are handled for missing or incorrect formats.

    Descriptive Stats: Test that the mean, std, min, and max are calculated correctly for different columns.

    Regression Models: Test that the regression models output coefficients, p-values, and R-squared correctly.

    Statistical Tests: Ensure that ANOVA, Chi-Square tests, etc., return correct p-values based on mock datasets.

    Integration Testing:

    UI Testing: Verify that the file upload, analysis selection, and result display features work smoothly together.

    Visualization Testing: Ensure that scatter plots, heatmaps, and regression plots are rendered correctly with real data.

    Edge Case Testing:

    Empty or invalid datasets (missing values, wrong format).

    Test with different numbers of predictors and targets (e.g., single vs. multiple predictors).

    Large datasets for performance and scalability testing.

    Regression Testing: After each new feature or fix, test the application as a whole to ensure no other functionality has been broken.

Project plan is in the ReadMe file.

# Checkpoints - Updated May 1, 2025

## 1. Technical Requirements
### 1.1 Data Handling and Input Validation 
- The application successfully accepts CSV files as input through the `/upload` endpoint
- File validation is implemented, checking for valid CSV format, non-empty files, and appropriate content
- Clear error messages are provided when uploads fail (invalid format, empty files, etc.)
- Maximum file size limit (50MB) is enforced and properly communicated

### 1.2 Data Preprocessing and Cleaning 
- The application categorizes columns into numeric and categorical automatically upon load
- Handles missing values during analysis by creating clean dataframes
- Provides information about missing values in descriptive statistics
- The memory usage is monitored and memory cleanup functionality is available

### 1.3 Descriptive Statistics Calculation 
- Computes comprehensive descriptive statistics for numeric columns:
  - Central tendency (mean, median, mode)
  - Dispersion (std dev, min, max, range)
  - Distribution characteristics (skewness, kurtosis)
  - Missing data metrics (count, percentage)
- Results are presented in a clear, tabular format with appropriate formatting

### 1.4 Support for Multiple Statistical Tests 
- ANOVA (Analysis of Variance) is fully implemented for comparing means across groups
- Chi-Square Test is implemented for testing association between categorical variables
- Correlation Analysis is implemented with options for different methods (Pearson, Spearman, Kendall)
- Each test provides appropriate statistics, p-values, and interpretation

### 1.5 Regression Analysis (Linear, Multiple, Polynomial) 
- Single Linear Regression is implemented with comprehensive diagnostics
- Multiple Linear Regression is implemented with VIF detection for multicollinearity
- Polynomial Regression is implemented with configurable degree parameter
- Residual diagnostics are available for regression models

### 1.6 Visualization of Results 
- Descriptive statistics include histograms and box plots
- Scatter matrices are available for exploring relationships between multiple variables
- Correlation heatmaps are implemented for visualizing relationship strength
- Regression plots include residual analysis visualizations 
- All visualizations are interactive (expandable) and can be viewed in full-screen mode

### 1.7 Principal Component Analysis (PCA) 
- PCA implementation is complete with explained variance visualization
- Component loadings are displayed in both tabular form and heatmap visualization
- Biplot visualization shows variable contributions to principal components
- Explained variance is plotted in a scree plot to aid in component selection

### 1.8 Clustering (KMeans) 
- KMeans clustering is implemented with configurable number of clusters
- Cluster visualization in 2D space is provided for selected features
- Cluster statistics are calculated and displayed for interpretation
- Parallel coordinates plot shows feature patterns across clusters

### 1.9 Logistic Regression for Classification 
- Logistic Regression is implemented for binary outcome prediction
- ROC curve and AUC calculation for model performance evaluation
- Confusion matrix visualization for classification results
- Coefficient visualization to show feature importance

### 1.10 Model Diagnostics and Residual Analysis 
- Comprehensive residual diagnostics are implemented:
  - Normality testing (Shapiro-Wilk test, Q-Q plot)
  - Homoscedasticity testing (Breusch-Pagan test, residual vs. fitted plot)
  - Residual independence (Durbin-Watson test)
  - Residual distribution visualization

### 1.11 User Interface (UI) 
- User-friendly web interface built using Flask with Bootstrap for styling
- Intuitive file upload mechanism with clear status feedback
- Analysis selection via cards with informative tooltips
- Variable selection with Select2 for enhanced usability
- Results displayed in organized, expandable panels with visualizations

### 1.12 Output and Result Export ⚠️
- Basic download functionality is implemented for some analyses (clustering, PCA)
- Not all analyses support exporting results to CSV/Excel format
- The download button is conditionally displayed based on download availability

### 1.13 Performance and Scalability 
- The application handles datasets with hundreds to thousands of rows
- Memory usage monitoring is implemented
- Memory cleanup functionality allows freeing resources
- Progress bar animation during analysis provides user feedback

### 1.14 Error Handling and Logging 
- Comprehensive error handling throughout the codebase
- Error logging using Python's logging module
- User-friendly error messages displayed in the UI
- Detailed error information available for debugging (expandable technical details)

### 1.15 Testing and Debugging ⚠️
- Error handling and logging are implemented
- No explicit unit tests are visible in the provided code
- Manual testing seems to be the primary verification method

## 2. Software Design
### Libraries 
The application successfully utilizes:
- Flask for the web framework
- Pandas for data manipulation
- NumPy for mathematical operations
- Matplotlib/Seaborn for visualizations
- Scikit-learn for machine learning models
- SciPy for statistical tests
- Statsmodels for advanced statistical modeling

### Functions 
The application has organized functions for:
- Data cleaning and preprocessing
- Descriptive statistics calculation
- Various regression analyses
- Statistical tests implementation
- PCA and clustering algorithms
- Result visualization generation

### UI Components 
The UI includes:
- File upload section with clear instructions
- Analysis type selection with informative cards
- Variable selection forms with appropriate controls
- Progress indication during analysis
- Organized results display with expandable sections

## 3. UI Implementation
The application has successfully implemented:
- Clean, responsive design using Bootstrap
- Card-based analysis selection
- Step indicator for workflow progress
- Interactive visualizations with zoom capability
- Expandable/collapsible sections for detailed results
- Tooltips for enhanced guidance
- Memory usage monitoring
- Full-screen modal for better visualization viewing

## 4. Testing Status
### Unit Testing 
- No explicit unit tests are visible in the provided code
- The application appears to rely on manual testing

### Integration Testing 
- No automated integration tests are visible
- The UI and backend integration appears functional

### Edge Case Handling 
- Empty or invalid datasets are handled with appropriate error messages
- Different variable combinations are supported
- Large dataset handling is implemented with memory management

### Regression Testing 
- No automated regression testing is evident
- Manual testing appears to be the verification method

## Suggested Next Steps
1. Implement comprehensive unit and integration tests
2. Expand export functionality to all analysis types
3. Add batch processing capabilities for multiple analyses
4. Enhance visualization customization options
5. Add user session management for saving/loading analyses
6. Implement caching for improved performance with large datasets