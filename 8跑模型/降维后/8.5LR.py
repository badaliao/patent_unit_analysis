import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb  # 保留导入，但不使用

# 1.加载数据
# positive_df = pd.read_excel(r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\摘要降维\patent_data_reduced3.xlsx')
# negitiva_df = pd.read_excel(r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\摘要降维\patent_data_reduced4.xlsx')
positive_df = pd.read_excel(r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\patent_data_reduced1.xlsx')
negitiva_df = pd.read_excel(r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\patent_data_reduced2.xlsx')

positive_df['label'] = 1
negitiva_df['label'] = 0
# 合并正负样本
data = pd.concat([positive_df, negitiva_df], ignore_index=True)

# 2.特征提取
# 特征列
numeric_features = ['权利要求数量', '独立权利要求数量', '发明人数量', '引证次数', '被引证次数',
                    '简单同族个数', '扩展同族个数', 'IPC类数量', '转让次数']

def parse_vector(vec_str):
    # 移除多余的括号和空格，转换为浮点数数组
    vec_str = vec_str.replace('[', '').replace(']', '').strip()
    return np.array([float(x) for x in vec_str.split() if x], dtype=np.float32)

# data['reduced_摘要向量'] = data['reduced_摘要向量'].apply(parse_vector)
data['reduced_技术知识元向量'] = data['reduced_技术知识元向量'].apply(parse_vector)
data['reduced_应用知识元向量'] = data['reduced_应用知识元向量'].apply(parse_vector)

# 将向量与数值特征合并
X_numeric = data[numeric_features].values
# X_abs_vec = np.stack(data['reduced_摘要向量'].values)
X_tech_vec = np.stack(data['reduced_技术知识元向量'].values)
X_app_vec = np.stack(data['reduced_应用知识元向量'].values)
X = np.hstack([X_numeric, X_tech_vec, X_app_vec])
# X = np.hstack([X_numeric, X_abs_vec])
y = data['label'].values

# 3.数据预处理
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 添加交互项
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# 4. Logistic Regression 模型与超参数调优
lr_model = LogisticRegression()

# 定义网格参数（适用于逻辑回归）
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # 正则化强度的倒数
    'penalty': ['l1', 'l2','elasticnet'], # 正则化类型
    'solver': ['liblinear', 'saga'], # 优化算法，支持 l1 和 l2
    'l1_ratio': [0, 0.5, 1],  # 仅在 elasticnet 时有效
    'max_iter': [100, 200, 500,1000]      # 最大迭代次数
}

# 使用 GridSearchCV 进行超参数调优
grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# 获取最佳模型和参数
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# 5.模型评估
y_pred = best_model.predict(X_test_scaled)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 6.输出结果
print("最佳参数:", best_params)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 特征重要性（逻辑回归使用系数而不是 feature_importances_）
feature_importance = np.abs(best_model.coef_[0])  # 取绝对值表示重要性
feature_names = numeric_features + [f'tech_vec_{i}' for i in range(20)] + [f'app_vec_{i}' for i in range(20)]
# feature_names = numeric_features + [f'abs_vec_{i}' for i in range(X_abs_vec.shape[1])]  # 动态适应向量维度
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
print("\n特征重要性（基于系数的绝对值）:")
print(importance_df.sort_values(by='Importance', ascending=False))