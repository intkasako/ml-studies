from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / 'job_salary_prediction_dataset.csv')

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nNull values per column:")
print(df.isnull().sum()) 
# 0 null values

# feature encoding
# nominals: native categorical support (XGBoost handles optimal partitioning)
nominal_cols = ["job_title", "industry", "location", "remote_work"]
for col in nominal_cols:
    df[col] = df[col].astype('category')

# ordinals: explicit map (real order matters)
education_order = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
company_size_order = {'Startup': 0, 'Small': 1, 'Medium': 2, 'Large': 3, 'Enterprise': 4}

df['education_level'] = df['education_level'].map(education_order)
df['company_size'] = df['company_size'].map(company_size_order)

print(df.head())

X = df.drop(columns=['salary'])
y = df['salary']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
X_train, X_cv, y_train, y_cv = train_test_split(X_temp, y_temp, test_size=0.25, random_state=9)

print("\n---- Split ----")
print(f"Train: {X_train.shape[0]} rows ({X_train.shape[0]/len(X):.0%})")
print(f"CV:    {X_cv.shape[0]} rows ({X_cv.shape[0]/len(X):.0%})")
print(f"Test:  {X_test.shape[0]} rows ({X_test.shape[0]/len(X):.0%})")

model = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    random_state=9, enable_categorical=True
)
model.fit(X_train, y_train)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:>6}  MAE: ${mae:>10,.2f}  RMSE: ${rmse:>10,.2f}  R²: {r2:.4f}")

print("\n---- Evaluation ----")
evaluate("Train", y_train, model.predict(X_train))
evaluate("CV",    y_cv,    model.predict(X_cv))
evaluate("Test",  y_test,  model.predict(X_test))

#tuning
n_estimators_options = [100, 300, 500]
learning_rate = [0.05, 0.1, 0.2]
max_depth = [3, 5, 7]
n_jobs = -1
results = []
print(f"{'n_est':>6}  {'lr':>8}  {'depth':>6}  {'train':>8}  {'cv':>8}")

for n in n_estimators_options:
    for lr in learning_rate:
        for md in max_depth:
            model = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md,
                                 random_state=9, enable_categorical=True, n_jobs=n_jobs)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            cv_acc = model.score(X_cv, y_cv)
            results.append({
                'n_estimators': n,
                'learning_rate': lr,
                'max_depth': md,
                'train_r2': train_acc,
                'cv_r2': cv_acc
            })
            print(f"{n:>6}  {lr:>8.3f}  {md:>6}  {train_acc:>8.4f}  {cv_acc:>8.4f}")

results_df = pd.DataFrame(results)
best = results_df.loc[results_df['cv_r2'].idxmax()]
print(f"\nBest: n_est={int(best['n_estimators'])}, lr={best['learning_rate']}, depth={int(best['max_depth'])}, cv_r2={best['cv_r2']:.4f}")

#final model

final_model = XGBRegressor(
    n_estimators=int(best['n_estimators']),
    learning_rate=best['learning_rate'],
    max_depth=int(best['max_depth']),
    random_state=9,
    enable_categorical=True,
    n_jobs=n_jobs
)
final_model.fit(X_train, y_train)

print("\n---- Final Model (Test Set) ----")
evaluate("Test", y_test, final_model.predict(X_test))

