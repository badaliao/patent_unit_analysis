import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier  # 使用随机森林替代 LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
import ast  # 用于安全地将字符串转换为列表

# 加载正负样本数据
def load_data(positive_file, negative_file):
    # 读取正负样本数据
    pos_data = pd.read_excel(positive_file)
    neg_data = pd.read_excel(negative_file)

    # 给数据添加标签列
    pos_data['ass_not_ass'] = 1  # 正样本标签为1
    neg_data['ass_not_ass'] = 0  # 负样本标签为0

    # 合并数据
    data = pd.concat([pos_data, neg_data], axis=0, ignore_index=True)
    return data

# 处理向量列（通用函数，支持技术知识元向量和应用知识元向量），将字符串转换为数值数组
def preprocess_vector_column(vector_series, column_name):
    # 将字符串形式的向量转换为数值数组
    def parse_vector(vector_str):
        try:
            # 将字符串转换为列表
            vector = ast.literal_eval(vector_str)
            # 确保是256维向量
            if isinstance(vector, list) and len(vector) == 1 and len(vector[0]) == 20:
                return np.array(vector[0], dtype=float)
            else:
                raise ValueError(f"{column_name} 向量维度不符合预期（期望256维）")
        except (ValueError, SyntaxError) as e:
            print(f"警告: 无法解析 {column_name} 向量 {vector_str}, 返回零向量: {e}")
            return np.zeros(20)  # 如果解析失败，返回256维零向量

    # 应用解析函数到每一行
    vector_array = np.stack(vector_series.apply(parse_vector).values)
    return vector_array

# 特征选择
def get_features(data):
    # 普通数值特征列
    numeric_features = [
        '权利要求数量', '独立权利要求数量', '发明人数量', '引证次数', '被引证次数',
        '简单同族个数', '扩展同族个数', 'IPC类数量', '转让次数'
    ]

    # 处理技术知识元向量列
    tech_vector_features = preprocess_vector_column(data['reduced_技术知识元向量'], 'reduced_技术知识元向量')
    tech_vector_df = pd.DataFrame(tech_vector_features, columns=[f'tech_vec_{i}' for i in range(20)])

    # 处理应用知识元向量列
    app_vector_features = preprocess_vector_column(data['reduced_应用知识元向量'], 'reduced_应用知识元向量')
    app_vector_df = pd.DataFrame(app_vector_features, columns=[f'app_vec_{i}' for i in range(20)])

    # 合并数值特征、技术知识元向量和应用知识元向量
    X_numeric = data[numeric_features]
    X = pd.concat([X_numeric, tech_vector_df, app_vector_df], axis=1)

    # 目标变量
    y = data['ass_not_ass']
    return X, y

# 训练模型并进行评估
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # 定义随机森林模型
    rf = RandomForestClassifier(random_state=100)

    # 参数网格（随机森林的常见超参数）
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],  # 树的数量
        'max_depth': [10, 20, 30, 40, None],  # 树的最大深度
        'min_samples_split': [2, 5, 10],  # 节点分裂所需的最小样本数
        'min_samples_leaf': [1, 2, 4],  # 叶子节点最小样本数
        'max_features': ['auto', 'sqrt', 'log2'],  # 每次分裂考虑的特征数量
        'class_weight': [None, 'balanced']  # 处理类别不平衡
    }

    # 使用RandomizedSearchCV进行超参数调优
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=50, cv=10,
                                       n_jobs=-1, verbose=2, random_state=20)
    random_search.fit(X_train, y_train)

    # 输出最佳参数
    best_params = random_search.best_params_
    print("最佳参数:", best_params)

    # 使用最佳参数的模型进行预测
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]

    print(f'混淆矩阵:\n{cm}')
    print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=0)  # 负样本标签
    recall = recall_score(y_test, y_pred, pos_label=0)  # 负样本标签
    f1 = f1_score(y_test, y_pred, pos_label=0)  # 负样本标签

    print(f'准确率: {accuracy}')
    print(f'精确率: {precision}')
    print(f'召回率: {recall}')
    print(f'F1分数: {f1}')
    print(classification_report(y_test, y_pred))

# 主程序
def main():
    # 输入文件路径
    positive_file = r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\patent_data_reduced1.xlsx'
    negative_file = r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\patent_data_reduced2.xlsx'

    # 加载数据
    data = load_data(positive_file, negative_file)

    # 特征和目标变量
    X, y = get_features(data)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # 标准化训练集
    X_test = scaler.transform(X_test)  # 标准化测试集（使用训练集的均值和标准差）

    # 训练模型并评估
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()