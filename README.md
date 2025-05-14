# Spoofing Behavior Detection via Ensemble Learning

This project performs spoofing behavior detection using a hybrid pipeline of Random Forest, MLP, and XGBoost, with outlier filtering using Local Outlier Factor (LOF).

## 📁 File Structure

- `main_pipeline.py`  
  Main script to run the detection pipeline, train models, apply LOF denoising, and evaluate results.

- `model_utils.py`  
  Utility module containing functions for:
  - Fixing invalid data values
  - LOF-based denoising
  - Model training and soft voting logic

- `all_first.csv`  
  Input CSV file with features and label column `y`.

## 📊 Model Details

- **Random Forest** as the primary detector
- **MLP + XGBoost** as secondary classifiers for anomaly samples
- **Soft voting** to refine predictions from secondary models
- **Threshold**: 0.2 used for binary classification

## 🚀 Run the Code

```bash
python main_pipeline.py
