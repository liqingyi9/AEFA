# main_pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score

from model_utils import validate_and_fix_data, train_models_with_lof, ensemble_soft_voting

# Load dataset
data = pd.read_csv('all_first.csv')
y = data['y'].values
X = data.drop(columns=['y']).values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fix invalid values
X_train = validate_and_fix_data(X_train)
X_test = validate_and_fix_data(X_test)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models and apply LOF filtering
rf_model, mlp_model, xgb_model, X_train_clean, y_train_clean = train_models_with_lof(X_train_scaled, y_train)

# Predict using Random Forest
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_pred = (rf_probs > 0.2).astype("int32")

# Second-stage soft voting
final_pred = ensemble_soft_voting(rf_pred, X_test_scaled, mlp_model, xgb_model)

# Metrics
recall = recall_score(y_test, final_pred)
precision = precision_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)

print("\nFinal Model Performance:")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")

# RF baseline
rf_recall = recall_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nOriginal Random Forest Model Performance (threshold=0.2):")
print(f"Recall: {rf_recall:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"F1-Score: {rf_f1:.4f}")
