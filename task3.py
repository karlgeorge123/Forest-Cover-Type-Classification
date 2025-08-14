import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.datasets import fetch_covtype

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# 1. Load Dataset

print("Loading Covertype dataset from scikit-learn...")
X, y = fetch_covtype(return_X_y=True)

# Adjust labels for XGBoost (0-based indexing)
y = y - 1  # Now classes are 0 to 6

# Column names for DataFrame
column_names = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
] + [f"Wilderness_Area_{i}" for i in range(4)] \
  + [f"Soil_Type_{i}" for i in range(40)]

df = pd.DataFrame(X, columns=column_names)
df["Cover_Type"] = y.astype(int)

print("Original dataset shape:", df.shape)

# Sample for faster training
df = df.sample(n=20000, random_state=42)
print("Sampled dataset shape:", df.shape)

# 2. Features & Target

X = df.drop(columns=["Cover_Type"])
y = df["Cover_Type"]


# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. Standardize Continuous Features

continuous_cols = [col for col in X.columns if X[col].nunique() > 2]

scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])


# 5. Random Forest (Randomized Search)

print("\n--- Random Forest Hyperparameter Tuning ---")
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_dist_rf = {
    "n_estimators": [100, 150],
    "max_depth": [10, None],
    "min_samples_split": [2, 5]
}

rand_rf = RandomizedSearchCV(
    rf, param_distributions=param_dist_rf, n_iter=3, cv=2,
    scoring="accuracy", n_jobs=-1, random_state=42, verbose=1
)
rand_rf.fit(X_train, y_train)

best_rf = rand_rf.best_estimator_
print("Best RF Params:", rand_rf.best_params_)

rf_pred = best_rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


# 6. XGBoost (Randomized Search)

print("\n--- XGBoost Hyperparameter Tuning ---")
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


param_dist_xgb = {
    "n_estimators": [100, 150],
    "max_depth": [6, 8],
    "learning_rate": [0.1, 0.05],
    "subsample": [0.8, 1.0]
}

rand_xgb = RandomizedSearchCV(
    xgb_model, param_distributions=param_dist_xgb, n_iter=3, cv=2,
    scoring="accuracy", n_jobs=-1, random_state=42, verbose=1
)
rand_xgb.fit(X_train, y_train)

best_xgb = rand_xgb.best_estimator_
print("Best XGB Params:", rand_xgb.best_params_)

xgb_pred = best_xgb.predict(X_test)
print("\nXGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))


# 7. Confusion Matrix (XGBoost Example)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 8. Feature Importance (XGBoost)

xgb.plot_importance(best_xgb, max_num_features=15, importance_type="weight")
plt.title("Top 15 Features - XGBoost")
plt.show()


# 9. Model Comparison

print("\nModel Accuracy Comparison:")
print(f"Random Forest: {accuracy_score(y_test, rf_pred):.4f}")
print(f"XGBoost: {accuracy_score(y_test, xgb_pred):.4f}")

