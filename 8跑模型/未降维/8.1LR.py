import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report


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
    # 选择特征列，根据需求调整特征列
    features = [
                '权利要求数量', '独立权利要求数量', '发明人数量', '引证次数', '被引证次数', '简单同族个数', '扩展同族个数', 'IPC类数量',
                '转让次数',
                # '度中心性', '介数中心性', '接近中心性', '特征向量中心性', 'Pagerank',
                # '技术知识元数量',
                # '技术知识元语义多样性',
                # '技术知识元语义集中度'
                ]
    X = data[features]
    y = data['ass_not_ass']
    return X, y


# 训练模型并进行评估
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # 定义模型和参数网格
    lr = LogisticRegression(random_state=100)

    # 参数网格，注意这里的组合，确保不出现不兼容的情况
    param_distributions = {
        'penalty': ['l1', 'l2'],  # 只使用支持的正则化方式
        'C': [0.01, 0.1, 1, 10, 100, 1000],  # 正则化强度范围
        'solver': ['liblinear', 'saga'],  # 只选择liblinear或saga
        'class_weight': [None, 'balanced']  # 是否进行类别不平衡调整
    }

    # 使用RandomizedSearchCV进行超参数调优
    random_search = RandomizedSearchCV(estimator=lr, param_distributions=param_distributions, n_iter=50, cv=10,
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
    positive_file = r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\传统＋技术网络指标(大于10年)_cleaned.xlsx'  # 替换为正样本的文件路径
    negative_file = r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\传统＋技术网络指标(小于等于10年)_cleaned.xlsx'  # 替换为负样本的文件路径

    # 加载数据
    data = load_data(positive_file, negative_file)

    # 特征和目标变量
    X, y = get_features(data)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练模型并评估
    train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
