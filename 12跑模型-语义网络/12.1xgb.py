import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import ast

# === 1. 加载融合特征数据 ===
high_df = pd.read_excel(r"C:\Users\hhhxj\Desktop\专利数据\z12融合后的（技术）\high_value_fusion_supervised.xlsx")
low_df = pd.read_excel(r"C:\Users\hhhxj\Desktop\专利数据\z12融合后的（技术）\low_value_fusion_supervised.xlsx")
high_df["label"] = 1
low_df["label"] = 0
df = pd.concat([high_df, low_df], ignore_index=True)

# === 2. 解析融合特征向量 ===
def parse_vector(vec_str):
    return np.array(ast.literal_eval(vec_str), dtype=np.float32)

df["融合特征"] = df["融合特征"].apply(parse_vector)
X = np.stack(df["融合特征"].values)
y = df["label"].values

# === 3. 数据划分与标准化 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. XGBoost 模型定义与超参数网格搜索 ===
xgb_model = xgb.XGBClassifier(eval_metric='logloss')

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# === 5. 模型评估 ===
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# === 6. 输出结果 ===
print("最佳参数:", best_params)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

