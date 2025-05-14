# Spoofing Behavior Detection via Ensemble Learning

This project performs spoofing behavior detection using a hybrid pipeline of three modules, Initial Data Optimazation Moduel, Probability Decision Module, and Advanced Voting Integration Module.

## 📁 File Structure

- `main_pipeline.py`  
  Main script to run the detection pipeline, train models, apply first module denoising, and evaluate results.

- `model_utils.py`  
  Utility module containing functions for:
  - Fixing invalid data values
  - LOF-based denoising
  - Model training and soft voting logic

- `all_first.csv`  
  Input CSV file with features and label column `y`.

## 📊 Model Details

- **Initial Data Optimazation Module** as the primary detector
- **Probability Decision Module** as secondary classifiers for anomaly samples
- **Advanced Voting Integration Module** to refine predictions from secondary models

## 🚀 Run the Code

```bash
python main_pipeline.py
