import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

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

# 特征选择
def get_features(data):
    # 选择特征列，可以根据需求调整特征列
    features = [
        '权利要求数量', '独立权利要求数量', '发明人数量', '引证次数', '被引证次数', '简单同族个数', '扩展同族个数', '转让次数', 'IPC类数量',
        # '度中心性', '介数中心性', '接近中心性', '特征向量中心性', 'Pagerank'
    ]
    X = data[features]
    y = data['ass_not_ass']
    return X, y

# 训练模型并进行评估
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # 定义决策树模型和参数网格
    dt = DecisionTreeClassifier(random_state=20)

    # 决策树的超参数调优范围
    param_distributions = {
        'criterion': ['gini', 'entropy'],  # 选择划分标准
        'max_depth': [None, 10, 20, 30, 40, 50],  # 最大深度
        'min_samples_split': [2, 5, 10],  # 分裂节点时的最小样本数
        'min_samples_leaf': [1, 2, 5],  # 叶节点的最小样本数
        'max_features': ['auto', 'sqrt', 'log2', None],  # 每次分裂时选择的最大特征数
        'class_weight': [None, 'balanced']  # 类别不平衡时使用加权
    }

    # 使用RandomizedSearchCV进行超参数调优
    random_search = RandomizedSearchCV(estimator=dt, param_distributions=param_distributions, n_iter=50, cv=5,
                                       n_jobs=-1, verbose=2, random_state=20)
    random_search.fit(X_train, y_train)

    # 输出最佳参数
    best_params = random_search.best_params_
    print("Best parameters for Decision Tree:", best_params)

    # 使用最佳参数的模型进行预测
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]

    print(f'Confusion Matrix:\n{cm}')
    print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=0)  # 调整正样本标签
    recall = recall_score(y_test, y_pred, pos_label=0)  # 调整正样本标签
    f1 = f1_score(y_test, y_pred, pos_label=0)  # 调整正样本标签

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(classification_report(y_test, y_pred))

# 主程序
def main():
    # 输入文件路径
    positive_file = r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\传统＋应用网络指标(大于10年)_cleaned.xlsx'  # 替换为正样本的文件路径
    negative_file = r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\传统＋应用网络指标(小于等于10年)_cleaned.xlsx'  # 替换为负样本的文件路径

    # 加载数据
    data = load_data(positive_file, negative_file)

    # 特征和目标变量
    X, y = get_features(data)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练模型并评估
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
