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

# 3. 提取摘要向量，转换为 NumPy 数组
abs_vectors = np.array([parse_vector(v) for v in df['摘要向量']])


# 4. 检查向量的维度，确保一致（如果有空值或维度不一致需要处理）
if abs_vectors.size == 0:
    print("警告：存在空向量，请检查数据！")
else:
    # 获取向量的维度（假设所有向量长度相同）
    n_features_tech = abs_vectors.shape[1] if abs_vectors.ndim > 1 else 0


    # 5. 使用 PCA 降维（降到 20 维，可以根据需要调整 n_components）
    if n_features_tech > 0:
        pca_tech = PCA(n_components=20)  # 技术知识元向量降维
        reduced_abs_vectors = pca_tech.fit_transform(abs_vectors)
    else:
        reduced_abs_vectors = np.array([])


    # 6. 将降维后的向量转为列表格式，添加到 DataFrame
    df['reduced_摘要向量'] = list(reduced_abs_vectors)  # 转为列表保存


    # 7. 保存结果到新 excel 文件
    df.to_excel(r'C:\Users\hhhxj\Desktop\专利数据\10降维向量\摘要降维\patent_data_reduced4.xlsx', index=False)
    print("降维后的数据已保存到文件中！")

    # 8. 可选：检查降维效果（查看保留的信息比例）
    if n_features_tech > 0:
        explained_variance_ratio_tech = pca_tech.explained_variance_ratio_
        print("摘要向量的每个主成分解释比例：", explained_variance_ratio_tech)
        print("摘要向量的总解释比例：", sum(explained_variance_ratio_tech))
