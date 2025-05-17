import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # 导入 SVM 分类器
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb  # 保留导入，但不使用

# 1.加载数据
positive_df = pd.read_excel(r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\patent_data_reduced1.xlsx')
negative_df = pd.read_excel(r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\patent_data_reduced2.xlsx')
# positive_df = pd.read_excel(r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\摘要降维\patent_data_reduced3.xlsx')
# negative_df = pd.read_excel(r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\摘要降维\patent_data_reduced4.xlsx')

positive_df['label'] = 1
negative_df['label'] = 0
# 合并正负样本
data = pd.concat([positive_df, negative_df], ignore_index=True)

# 2.特征提取
# 特征列
numeric_features = ['权利要求数量', '独立权利要求数量', '发明人数量', '引证次数', '被引证次数',
                    '简单同族个数', '扩展同族个数', 'IPC类数量', '转让次数']

def parse_vector(vec_str):
    # 移除多余的括号和空格，转换为浮点数数组
    vec_str = vec_str.replace('[', '').replace(']', '').strip()
    return np.array([float(x) for x in vec_str.split() if x], dtype=np.float32)

# data['reduced_摘要向量'] = data['reduced_摘要向量'].apply(parse_vector)
# data['reduced_技术知识元向量'] = data['reduced_技术知识元向量'].apply(parse_vector)
data['reduced_应用知识元向量'] = data['reduced_应用知识元向量'].apply(parse_vector)

# 将向量与数值特征合并
X_numeric = data[numeric_features].values
# X_abs_vec = np.stack(data['reduced_摘要向量'].values)
# X_tech_vec = np.stack(data['reduced_技术知识元向量'].values)
X_app_vec = np.stack(data['reduced_应用知识元向量'].values)
# X = np.hstack([X_numeric])
X = np.hstack([X_numeric, X_app_vec])
# X = np.hstack([X_numeric, X_abs_vec])
y = data['label'].values

# 3.数据预处理
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（SVM 对特征尺度敏感，标准化很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. SVM 模型与超参数调优
svm_model = SVC(probability=True)  # probability=True 启用概率预测

# 定义网格参数（适用于 SVM）
param_grid = {
    'C': [0.1, 1, 10, 100],          # 正则化参数
    'kernel': ['linear', 'rbf'],      # 核函数：线性核和高斯核
    'gamma': ['scale', 'auto', 0.1, 1],  # 核函数参数（仅对 rbf 有效）
}

# 使用 GridSearchCV 进行超参数调优
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
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

# 特征重要性（SVM 的重要性依赖于核函数）
if best_params['kernel'] == 'linear':
    # 对于线性核，可以直接使用系数
    feature_importance = np.abs(best_model.coef_[0])
    # feature_names = numeric_features + [f'tech_vec_{i}' for i in range(20)] + [f'app_vec_{i}' for i in range(20)]
    feature_names = numeric_features + [f'tech_vec_{i}' for i in range(20)]
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    print("\n特征重要性（基于线性核系数的绝对值）:")
    print(importance_df.sort_values(by='Importance', ascending=False))
else:
    # 对于非线性核（如 rbf），无法直接获取特征重要性
    print("\n注意：使用非线性核（如 rbf），无法直接计算特征重要性。")
    print("若需特征重要性，可尝试线性核或其他方法（如 permutation importance）。")