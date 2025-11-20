# -----------------------------------------------
# TASK 5 – Decision Trees & Random Forests
# Using YOUR OWN CSV Dataset
# Target column = LAST COLUMN automatically
# -----------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------
# 1. Load YOUR dataset
# ------------------------------------------------
df = pd.read_csv("C:\\Users\\G HARSHITHA\\Downloads\\archive (6).zip")   # <--- CHANGE ONLY THIS
print("Loaded dataset. Columns:", df.columns.tolist())

# ------------------------------------------------
# 2. Automatically split features & target
# ------------------------------------------------
# Target = LAST COLUMN
X = df.iloc[:, :-1]     # all columns except last
y = df.iloc[:, -1]      # last column as target

# ------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------
# 4. Decision Tree Classifier
# ------------------------------------------------
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n=== Decision Tree Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# ---- Decision Tree Visualization ----
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=X.columns.tolist())
plt.title("Decision Tree Visualization")
plt.show()

# ------------------------------------------------
# 5. Random Forest Classifier
# ------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# ------------------------------------------------
# 6. Feature Importance
# ------------------------------------------------
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(10, 8))
plt.barh(features, importances)
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.show()

# ------------------------------------------------
# 7. Cross-Validation
# ------------------------------------------------
cv_scores = cross_val_score(rf, X, y, cv=5)
print("\nCross-Validation Scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())
