HR Employee Attrition Analysis & Prediction

This project focuses on analyzing employee data to identify key factors contributing to attrition and building a machine learning model to predict whether an employee is likely to leave the company.

📋 Project Overview Employee retention is a critical challenge for HR departments. This project uses the IBM HR Analytics Employee Attrition & Performance dataset to:

Perform Exploratory Data Analysis (EDA) to find patterns in employee turnover.

Build and evaluate predictive models (Logistic Regression, Tree-based models).

Use SHAP (SHapley Additive exPlanations) to interpret model predictions and understand feature importance for individual employees.

📊 Dataset Features The dataset includes 35 columns, such as:

Demographics: Age, Education, Gender, Marital Status.

Job Info: Job Role, Department, Job Level, Total Working Years.

Engagement: Job Satisfaction, Work-Life Balance, Relationship Satisfaction.

Financial: Monthly Income, Stock Option Level, Daily Rate.

Target Variable: Attrition (Yes/No).

🛠️ Tech Stack Language: Python

Libraries: - pandas, numpy for data manipulation.

matplotlib, seaborn for visualization.

scikit-learn for machine learning and hyperparameter tuning.

SHAP for model explainability.

🚀 Key Insights from Analysis Model Interpretation: The project utilizes SHAP force plots to explain specific instances (e.g., explaining why a specific employee at a certain index was predicted to stay or leave).

Hyperparameter Tuning: Systematic tuning of Logistic Regression using GridSearchCV was identified as a critical step to optimize performance.

🔧 How to Use Clone the repository:

Install dependencies:

Bash pip install pandas scikit-learn shap matplotlib seaborn Run the Notebook: Open Hr_attrition.ipynb in Jupyter Lab or Google Colab to see the full analysis and model results.

📈 Future Work Complete the hyperparameter tuning for Logistic Regression to compare final metrics against other models.

Explore advanced ensemble methods like XGBoost or LightGBM for higher accuracy.

Implement a web-based dashboard for HR managers to input employee data and receive attrition risk scores.
