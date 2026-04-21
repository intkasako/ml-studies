import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("d:/python-ws/ml-studies/xgboost_salary/job_salary_prediction_dataset.csv")

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nNull values per column:")
print(df.isnull().sum()) 
# 0 null values

# feature encoding
categorical_cols = ["job_title", "education_level", "industry", "company_size", "location", "remote_work"]
label_maps = {} 
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_maps[col] = dict(enumerate(le.classes_))

print(df.head())
print(label_maps)

X = df.drop(columns=['salary'])
y = df['salary']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
X_train, X_cv, y_train, y_cv = train_test_split(X_temp, y_temp, test_size=0.25, random_state=9)

print("\n---- Split ----")
print(f"Train: {X_train.shape[0]} rows ({X_train.shape[0]/len(X):.0%})")
print(f"CV:    {X_cv.shape[0]} rows ({X_cv.shape[0]/len(X):.0%})")
print(f"Test:  {X_test.shape[0]} rows ({X_test.shape[0]/len(X):.0%})")

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=9)
model.fit(X_train, y_train)
print("\nTrain:", model.score(X_train, y_train))
print("CV score:", model.score(X_cv, y_cv))
# OK, no overfitting

# test set 
print("Test score:", model.score(X_test, y_test))