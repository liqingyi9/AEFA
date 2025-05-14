import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('all_first.csv')

# Separate labels and features (assuming 'y' is the label column)
y = data['y'].values
X = data.drop(columns=['y']).values  # Remove 'y' column, use the rest as features

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to fix invalid values (NaN, inf)
def validate_and_fix_data(X):
    if np.isnan(X).any() or np.isinf(X).any():
        print("Fixing invalid values...")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X

# Apply data validation
X_train = validate_and_fix_data(X_train)
X_test = validate_and_fix_data(X_test)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Denoising with LocalOutlierFactor (LOF)
print("Original training set size:", X_train_scaled.shape[0])

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_labels = lof.fit_predict(X_train_scaled)  # 1: inliers, -1: outliers

# Keep only inlier samples
X_train_clean = X_train_scaled[lof_labels == 1]
y_train_clean = y_train[lof_labels == 1]

print("Training set size after denoising:", X_train_clean.shape[0])
print("Removed", X_train_scaled.shape[0] - X_train_clean.shape[0], "outlier samples")

# 2. Train Random Forest as the primary model using denoised data
rf_model = RandomForestClassifier(criterion='gini', max_depth=200, 
                                   min_samples_split=5, n_estimators=95, 
                                   random_state=42)
rf_model.fit(X_train_clean, y_train_clean)

# 3. Train MLP and XGBoost models for secondary decision-making
mlp_model = MLPClassifier(random_state=42, max_iter=400, activation='relu')
mlp_model.fit(X_train_clean, y_train_clean)

xgb_model = XGBClassifier(random_state=42, max_depth=5, min_child_weight=5)
xgb_model.fit(X_train_clean, y_train_clean)

# Random Forest prediction (threshold 0.2)
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_pred = (rf_probs > 0.2).astype("int32")

# Identify samples predicted as anomalous by RF
anomaly_idx = np.where(rf_pred == 1)[0]
normal_idx = np.where(rf_pred == 0)[0]

# Apply soft voting on anomalous samples using MLP and XGBoost
if len(anomaly_idx) > 0:
    X_anomaly = X_test_scaled[anomaly_idx]
    
    # Predict probabilities using MLP and XGBoost
    mlp_probs = mlp_model.predict_proba(X_anomaly)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_anomaly)[:, 1]
    
    # Soft voting (weighted average)
    ensemble_probs = mlp_probs * 0.1 + xgb_probs * 0.9
    ensemble_pred = (ensemble_probs > 0.2).astype("int32")
    
    # Final prediction results
    final_pred = np.zeros_like(y_test)
    final_pred[normal_idx] = 0  # Keep RF-normal as normal
    final_pred[anomaly_idx] = ensemble_pred  # Apply ensemble results
else:
    final_pred = rf_pred  # If no anomalies detected by RF

# Compute evaluation metrics
recall = recall_score(y_test, final_pred)
precision = precision_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)

print("\nFinal model performance:")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")

# Optional: also show original RF performance
rf_recall = recall_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nOriginal Random Forest model performance (threshold 0.2):")
print(f"Recall: {rf_recall:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"F1-Score: {rf_f1:.4f}")
