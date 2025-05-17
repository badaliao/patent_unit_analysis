import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# 读取数据
excel_file=r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\传统＋技术网络指标(小于等于10年)_cleaned.xlsx'
df = pd.read_excel(excel_file)

#2.解析字符串转为numpy数组
def parse_vector(vector_str):
    if pd.isna(vector_str):
        return np.array([])
    vector = np.array([float(x) for x in vector_str.strip('[]').split(', ')])
    return vector

# 3. 提取技术知识元向量和应用知识元向量，转换为 NumPy 数组
tech_vectors = np.array([parse_vector(v) for v in df['技术知识元向量']])
app_vectors = np.array([parse_vector(v) for v in df['应用知识元向量']])

# 4. 检查向量的维度，确保一致（如果有空值或维度不一致需要处理）
if tech_vectors.size == 0 or app_vectors.size == 0:
    print("警告：存在空向量，请检查数据！")
else:
    # 获取向量的维度（假设所有向量长度相同）
    n_features_tech = tech_vectors.shape[1] if tech_vectors.ndim > 1 else 0
    n_features_app = app_vectors.shape[1] if app_vectors.ndim > 1 else 0

    # 5. 使用 PCA 降维（降到 20 维，可以根据需要调整 n_components）
    if n_features_tech > 0:
        pca_tech = PCA(n_components=20)  # 技术知识元向量降维
        reduced_tech_vectors = pca_tech.fit_transform(tech_vectors)
    else:
        reduced_tech_vectors = np.array([])

    if n_features_app > 0:
        pca_app = PCA(n_components=20)  # 应用知识元向量降维
        reduced_app_vectors = pca_app.fit_transform(app_vectors)
    else:
        reduced_app_vectors = np.array([])

    # 6. 将降维后的向量转为列表格式，添加到 DataFrame
    df['reduced_技术知识元向量'] = list(reduced_tech_vectors)  # 转为列表保存
    df['reduced_应用知识元向量'] = list(reduced_app_vectors)  # 转为列表保存

    # 7. 保存结果到新 CSV 文件
    # df.to_excel(r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\patent_data_reduced.xlsx', index=False)
    df.to_excel(r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\patent_data_reduced2.xlsx', index=False)
    print("降维后的数据已保存到 patent_data_reduced.xlsx 文件中！")

    # 8. 可选：检查降维效果（查看保留的信息比例）
    if n_features_tech > 0:
        explained_variance_ratio_tech = pca_tech.explained_variance_ratio_
        print("技术知识元向量的每个主成分解释比例：", explained_variance_ratio_tech)
        print("技术知识元向量的总解释比例：", sum(explained_variance_ratio_tech))

    if n_features_app > 0:
        explained_variance_ratio_app = pca_app.explained_variance_ratio_
        print("应用知识元向量的每个主成分解释比例：", explained_variance_ratio_app)
        print("应用知识元向量的总解释比例：", sum(explained_variance_ratio_app))