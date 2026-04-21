from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

IMAGES_DIR = Path(__file__).parent / 'images'

df = sns.load_dataset('penguins')

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nNull values per column:")
print(df.isnull().sum())
print("\nTarget distribution (specie):")
print(df['species'].value_counts(normalize=True))

print(df['species'].value_counts())


# null treatment
#evaluating which possibility to take
print(df[df.isnull().any(axis=1)])

#2 individual have almost all feaetures as null, so it's interesting to drop them
df = df.drop(index=[3, 339]) 

#fill the null values on sex feature as the sex mode of each specie
df['sex'] = df.groupby('species')['sex'].transform(
    lambda x: x.fillna(x.mode()[0])
)

print("\nNulls after treatment:")
print(df.isnull().sum())

# feature encoding

# sex
df['sex'] = df['sex'].map({'Male' : 1, 'Female' : 0})
# island
df = pd.get_dummies(df, columns=['island'], drop_first=True, dtype=int)

#checking changes
print(df.head())

# separating features(x) and target (y)
X = df.drop(columns=['species'])
y = df['species']

# data split
# 1st split 
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=9, stratify=y)
# 2nd split
X_train, X_cv, y_train, y_cv = train_test_split(X_temp, y_temp, test_size=0.25, random_state=9, stratify=y_temp)

print("\n---- Split ----")
print(f"Train: {X_train.shape[0]} rows ({X_train.shape[0]/len(X):.0%})")
print(f"CV:    {X_cv.shape[0]} rows ({X_cv.shape[0]/len(X):.0%})")
print(f"Test:  {X_test.shape[0]} rows ({X_test.shape[0]/len(X):.0%})")

# models
print("---- training ----")
simple_model = DecisionTreeClassifier(random_state=9, criterion='entropy')
forest_model = RandomForestClassifier(random_state=9, criterion='entropy')
simple_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

simple_train_accuracy = simple_model.score(X_train, y_train)
forest_train_accuracy = forest_model.score(X_train, y_train)
simple_cv_accuracy = simple_model.score(X_cv, y_cv)
forest_cv_accuracy = forest_model.score(X_cv, y_cv)

print(f"\nSimple tree    — train: {simple_train_accuracy:.4f}  cv: {simple_cv_accuracy:.4f}")
print(f"Random Forest  — train: {forest_train_accuracy:.4f}  cv: {forest_cv_accuracy:.4f}")

# Hyperparameter grid
n_estimators_list = [50, 100, 200, 500]
max_depth_list = [3, 5, 7, 10, None]

print("\n--- Hyperparameter Tuning ---")
print(f"{'n_est':>6}  {'depth':>6}  {'train':>8}  {'cv':>8}")

results = []
for n in n_estimators_list:
    for d in max_depth_list:
        m = RandomForestClassifier(
            n_estimators=n, max_depth=d,
            random_state=9, criterion='entropy', n_jobs=-1
        )
        m.fit(X_train, y_train)
        train_acc = m.score(X_train, y_train)
        cv_acc = m.score(X_cv, y_cv)
        results.append({
            'n_estimators': n,
            'max_depth': d,
            'train': train_acc,
            'cv': cv_acc
        })
        depth_str = str(d) if d is not None else 'None'
        print(f"{n:>6}  {depth_str:>6}  {train_acc:>8.4f}  {cv_acc:>8.4f}")

results_df = pd.DataFrame(results)

best = results_df.loc[results_df['cv'].idxmax()]
print(f"\nBest: n_estimators={int(best['n_estimators'])}, max_depth={best['max_depth']}, cv={best['cv']:.4f}")
