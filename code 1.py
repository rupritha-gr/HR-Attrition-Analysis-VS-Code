import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load your specific dataset
path = r"C:\Users\rupri\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(path)

# 2. Basic Cleaning: Drop columns with zero vhariance (useless for prediction)
cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df.drop(columns=cols_to_drop, inplace=True)

# 3. Exploratory Data Analysis (EDA)
# Let's see how Overtime impacts Attrition
plt.figure(figsize=(8,5))
sns.countplot(x='OverTime', hue='Attrition', data=df, palette='viridis')
plt.title('Attrition vs OverTime')
plt.show()

# 4. Data Preprocessing
# Convert Target 'Attrition' to 0 and 1
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Encode Categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# 5. Split Data
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Feature Importance (The most important part for HR)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 Factors Driving Attrition:")
print(importances.head(5))

# Export results for Power BI
df['Risk_Score'] = model.predict_proba(X)[:, 1] # Probability of leaving
df.to_csv("HR_Attrition_Final_Results.csv", index=False)