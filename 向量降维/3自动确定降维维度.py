import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取数据
excel_file = r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\传统＋技术网络指标(小于等于10年)_cleaned.xlsx'
df = pd.read_excel(excel_file)

# 2. 解析字符串转为numpy数组
def parse_vector(vector_str):
    if pd.isna(vector_str):
        return np.array([])
    vector = np.array([float(x) for x in vector_str.strip('[]').split(', ')])
    return vector

# 3. 提取技术知识元向量和应用知识元向量，转换为 NumPy 数组
tech_vectors = np.array([parse_vector(v) for v in df['技术知识元向量']])
app_vectors = np.array([parse_vector(v) for v in df['应用知识元向量']])

# 4. 检查向量的维度，确保一致
if tech_vectors.size == 0 or app_vectors.size == 0:
    print("警告：存在空向量，请检查数据！")
else:
    n_features_tech = tech_vectors.shape[1] if tech_vectors.ndim > 1 else 0
    n_features_app = app_vectors.shape[1] if app_vectors.ndim > 1 else 0

    # 5. 定义函数来选择最佳维度
    def select_n_components(pca, variance_threshold=0.70):
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        return n_components, cumulative_variance

    # 6. 处理技术知识元向量
    if n_features_tech > 0:
        # 先进行全维度 PCA
        pca_tech_full = PCA().fit(tech_vectors)
        # 确定最佳维度（保留70%方差）
        n_components_tech, cum_var_tech = select_n_components(pca_tech_full, 0.70)
        print(f"技术知识元向量保留70%方差所需维度: {n_components_tech}")

        # 使用20维进行降维并计算解释性
        pca_20_tech = PCA(n_components=20)
        reduced_20_tech_vectors = pca_20_tech.fit_transform(tech_vectors)
        explained_variance_ratio_20_tech = pca_20_tech.explained_variance_ratio_
        total_explained_variance_20_tech = sum(explained_variance_ratio_20_tech)
        print(f"技术知识元向量保留20维时的总解释比例: {total_explained_variance_20_tech:.4f}")

        # 使用最佳维度进行降维
        pca_tech = PCA(n_components=n_components_tech)
        reduced_tech_vectors = pca_tech.fit_transform(tech_vectors)

        # 可视化累积方差比例
        plt.figure(figsize=(10, 6))
        plt.plot(cum_var_tech, label='Cumulative Explained Variance')
        plt.axhline(y=0.70, color='r', linestyle='--', label='70% threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Technical Knowledge Vectors PCA')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        reduced_tech_vectors = np.array([])
        print("技术知识元向量数据无效")

    # 7. 处理应用知识元向量
    if n_features_app > 0:
        # 先进行全维度 PCA
        pca_app_full = PCA().fit(app_vectors)
        # 确定最佳维度（保留70%方差）
        n_components_app, cum_var_app = select_n_components(pca_app_full, 0.70)
        print(f"应用知识元向量保留70%方差所需维度: {n_components_app}")

        # 使用20维进行降维并计算解释性
        pca_20_app = PCA(n_components=20)
        reduced_20_app_vectors = pca_20_app.fit_transform(app_vectors)
        explained_variance_ratio_20_app = pca_20_app.explained_variance_ratio_
        total_explained_variance_20_app = sum(explained_variance_ratio_20_app)
        print(f"应用知识元向量保留20维时的总解释比例: {total_explained_variance_20_app:.4f}")

        # 使用最佳维度进行降维
        pca_app = PCA(n_components=n_components_app)
        reduced_app_vectors = pca_app.fit_transform(app_vectors)

        # 可视化累积方差比例
        plt.figure(figsize=(10, 6))
        plt.plot(cum_var_app, label='Cumulative Explained Variance')
        plt.axhline(y=0.70, color='r', linestyle='--', label='70% threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Application Knowledge Vectors PCA')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        reduced_app_vectors = np.array([])
        print("应用知识元向量数据无效")

    # 8. 将降维后的向量转为列表格式，添加到 DataFrame
    df['reduced_技术知识元向量'] = list(reduced_tech_vectors)
    df['reduced_应用知识元向量'] = list(reduced_app_vectors)

    # 9. 保存结果到新 Excel 文件
    output_file = r'C:\Users\hhhxj\Desktop\降维数.xlsx'
    df.to_excel(output_file, index=False)
    print(f"降维后的数据已保存到 {output_file} 文件中！")

    # 10. 输出降维效果
    if n_features_tech > 0:
        print("技术知识元向量解释比例：", pca_tech.explained_variance_ratio_)
        print("技术知识元向量总解释比例：", sum(pca_tech.explained_variance_ratio_))
    if n_features_app > 0:
        print("应用知识元向量解释比例：", pca_app.explained_variance_ratio_)
        print("应用知识元向量总解释比例：", sum(pca_app.explained_variance_ratio_))