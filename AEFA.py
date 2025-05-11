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

# 加载数据
data = pd.read_csv('all_first.csv')

# 分离标签和特征，假设标签在第一列
y = data['y'].values  # 假设 'y' 列是标签
X = data.drop(columns=['y']).values  # 移除 'account' 和标签列，其余作为特征

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 检查并修复无效值
def validate_and_fix_data(X):
    if np.isnan(X).any() or np.isinf(X).any():
        print("修复无效值...")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X

# 修复数据
X_train = validate_and_fix_data(X_train)
X_test = validate_and_fix_data(X_test)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. 使用 LocalOutlierFactor (LOF) 进行去噪
print("原始训练集大小:", X_train_scaled.shape[0])

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_labels = lof.fit_predict(X_train_scaled)  # 1表示正常，-1表示异常

# 只保留正常样本
X_train_clean = X_train_scaled[lof_labels == 1]
y_train_clean = y_train[lof_labels == 1]

print("去噪后训练集大小:", X_train_clean.shape[0])
print("移除了", X_train_scaled.shape[0] - X_train_clean.shape[0], "个异常样本")

# 2. 定义并训练随机森林模型（主模型）使用去噪后的数据
rf_model = RandomForestClassifier(criterion='gini', max_depth=200, 
                                min_samples_split=5, n_estimators=95, 
                                random_state=42)
rf_model.fit(X_train_clean, y_train_clean)

# 3. 定义并训练 MLP 和 XGBoost 模型（用于二次判断）使用去噪后的数据
mlp_model = MLPClassifier(random_state=42, max_iter=400, activation='relu')
mlp_model.fit(X_train_clean, y_train_clean)

xgb_model = XGBClassifier(random_state=42, max_depth = 5, min_child_weight = 5)
xgb_model.fit(X_train_clean, y_train_clean)

# 随机森林主模型预测（阈值0.2）
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_pred = (rf_probs > 0.2).astype("int32")

# 获取被RF判为异常样本的索引
anomaly_idx = np.where(rf_pred == 1)[0]
normal_idx = np.where(rf_pred == 0)[0]

# 对这些异常样本用MLP和XGBoost进行软投票
if len(anomaly_idx) > 0:
    X_anomaly = X_test_scaled[anomaly_idx]
    
    # 获取各模型预测概率
    mlp_probs = mlp_model.predict_proba(X_anomaly)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_anomaly)[:, 1]
    
    # 软投票（平均概率）
    ensemble_probs = mlp_probs * 0.1 + xgb_probs * 0.9
    ensemble_pred = (ensemble_probs > 0.2).astype("int32")
    
    # 合并结果
    final_pred = np.zeros_like(y_test)
    final_pred[normal_idx] = 0  # RF判为正常的保持
    final_pred[anomaly_idx] = ensemble_pred  # 二次判断结果
else:
    final_pred = rf_pred  # 如果没有被RF判为异常的样本

# 计算并输出指标
recall = recall_score(y_test, final_pred)
precision = precision_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)

print("\n最终模型性能指标:")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")

# 可选：输出原始RF模型的指标对比
rf_recall = recall_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\n原始Random Forest模型性能指标(阈值0.2):")
print(f"Recall: {rf_recall:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"F1-Score: {rf_f1:.4f}")