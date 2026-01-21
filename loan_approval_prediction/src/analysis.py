import pandas as pd
import numpy as np

df = pd.read_csv("data/loan_data.csv")
print("Dataset loaded")

df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("Missing values handled")

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X = pd.get_dummies(X, drop_first=True)
print("Categorical data converted")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split done")

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
print("Model trained")

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n LOAN APPROVAL MODEL COMPLETED SUCCESSFULLY ")

sample_applicant = X_test.iloc[0:1]

prediction = model.predict(sample_applicant)

if prediction[0] == 'Y' or prediction[0] == 1:
    print("\nLoan Status Prediction: APPROVED ")
else:
    print("\nLoan Status Prediction: REJECTED ")
