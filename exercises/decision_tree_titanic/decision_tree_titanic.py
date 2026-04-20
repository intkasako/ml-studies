from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

IMAGES_DIR = Path(__file__).parent / 'images'


df = sns.load_dataset('titanic')

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nNull values per column:")
print(df.isnull().sum())
print("\nTarget distribution (survived):")
print(df['survived'].value_counts(normalize=True))

print(df.columns.tolist())
print("\n--- alive vs survived ---")
print(df[['survived', 'alive']].head(10))

df = df.drop(columns=['alive', 'class', 'embark_town', 'who',
                      'adult_male', 'alone', 'deck'])

df['age_missing'] = df['age'].isnull().astype(int)
df['age'] = df['age'].fillna(df['age'].median())

# sex: binary -> map directly to 0/1
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# embarked: 3 categories (C, Q, S) -> one-hot
df = pd.get_dummies(df, columns=['embarked'], drop_first=True, dtype=int)

#data split
X = df.drop(columns=['survived'])
y = df['survived']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2,random_state=9, stratify=y ) #I like the number 9 :p
X_train, X_crossval, y_train, y_crossval = train_test_split(X_temp, y_temp, test_size=0.25, random_state=9, stratify=y_temp)

print("\n--- Split ---")
print(f"Train: {X_train.shape[0]} rows ({X_train.shape[0]/len(X):.0%})")
print(f"CV:    {X_crossval.shape[0]} rows ({X_crossval.shape[0]/len(X):.0%})")
print(f"Test:  {X_test.shape[0]} rows ({X_test.shape[0]/len(X):.0%})")
print(f"\nSurvived ratio — train: {y_train.mean():.3f}, "
      f"cv: {y_crossval.mean():.3f}, test: {y_test.mean():.3f}")

#baseline -> tree without restriction

model = DecisionTreeClassifier(random_state=9, criterion='entropy') #number 9 again :p
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
crossval_accuracy = model.score(X_crossval, y_crossval)

print(f"train accuracy: {train_accuracy:.4f}")
print(f"cross-validation accuracy: {crossval_accuracy:.4f}")

print(f"Tree depth: {model.get_depth()}")
print(f"leaves_num: {model.get_n_leaves()}")


# --- Tuning max_depth ---
depths = range(1, 16)
train_accs = []
cv_accs = []

for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=9, criterion='entropy')
    m.fit(X_train, y_train)
    train_accs.append(m.score(X_train, y_train))
    cv_accs.append(m.score(X_crossval, y_crossval))

# Find the best max_depth
best_depth = depths[np.argmax(cv_accs)]
best_cv = max(cv_accs)

print("\n--- Tuning max_depth ---")
for d, ta, cv in zip(depths, train_accs, cv_accs):
    mark = "  ←" if d == best_depth else ""
    print(f"depth={d:2d}  train={ta:.4f}  cv={cv:.4f}{mark}")
print(f"\nBest max_depth: {best_depth} (cv acc = {best_cv:.4f})")

# Plot
plt.figure(figsize=(9, 5))
plt.plot(depths, train_accs, 'o-', label='Train')
plt.plot(depths, cv_accs, 's-', label='CV')
plt.axvline(best_depth, color='red', linestyle='--', alpha=0.5,
            label=f'Best: {best_depth}')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Train vs CV accuracy by max_depth')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(IMAGES_DIR / 'depth_tuning.png', dpi=100, bbox_inches='tight')

final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=9, criterion='entropy')
final_model.fit(X_train, y_train)

print("\nFinal model evaluation:")
train_accuracy = final_model.score(X_train, y_train)
crossval_accuracy = final_model.score(X_crossval, y_crossval)
print(f"train accuracy: {train_accuracy:.4f}")
print(f"cross-validation accuracy: {crossval_accuracy:.4f}")

plt.figure(figsize=(36, 12))
plot_tree(
    final_model, feature_names=X.columns, class_names=['died', 'survived'],
    filled=True, rounded=True, fontsize=8)

plt.title(f"Decision Tree (max_depth={best_depth})")
plt.savefig(IMAGES_DIR / 'final_tree.png', dpi=100, bbox_inches='tight')

# --- Feature Importance ---
importances = final_model.feature_importances_
features = X_train.columns
idx = np.argsort(importances)[::-1]

plt.figure(figsize=(9, 5))
plt.bar(range(len(features)), importances[idx])
plt.xticks(range(len(features)), features[idx], rotation=45, ha='right')
plt.ylabel('Importance')
plt.title('Feature Importance — Final Tree')
plt.tight_layout()
plt.savefig(IMAGES_DIR / 'feature_importance.png', dpi=100, bbox_inches='tight')

print("\n--- Feature Importance ---")
for f, imp in zip(features[idx], importances[idx]):
    print(f"  {f:15s} {imp:.4f}")


# probabilities assigned by the model (10 first CV samples)
probs = final_model.predict_proba(X_crossval)
preds = final_model.predict(X_crossval)
print("\n--- Predicted Probabilities (first 10 CV samples) ---")
print(f"  {'died':>8}  {'survived':>10}   prediction")
for p, pred in list(zip(probs, preds))[:10]:
    label = 'survived' if pred == 1 else 'died'
    print(f"  {p[0]:>8.3f}  {p[1]:>10.3f}   {label}")


print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

test_acc = final_model.score(X_test, y_test)
print(f"\nAccuracy on test: {test_acc:.4f}")
print(f"Accuracy on cv:   {crossval_accuracy:.4f}")

y_pred = final_model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['died', 'survived']))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(f"                 pred died  pred survived")
print(f"actual died      {cm[0,0]:>9}  {cm[0,1]:>13}")
print(f"actual survived  {cm[1,0]:>9}  {cm[1,1]:>13}")

plt.show()  # show all open figures at once
