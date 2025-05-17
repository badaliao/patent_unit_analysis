from zhipuai import ZhipuAI
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

api_key1 = 'f04ed8dece0817c5df88425f54de769f.rj1KGfIk4nVOEwyd'  # 请替换为你的APIKey
client = ZhipuAI(api_key=api_key1)


def read_patent_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    ids = df.iloc[:, 0].values.tolist()  # 序号
    abstracts = df.iloc[:, 1].values.tolist()  # 专利摘要
    tech_knowledges = df.iloc[:, 2].values.tolist()  # 技术知识元
    app_knowledges = df.iloc[:, 3].values.tolist()  # 应用知识元
    return ids, abstracts, tech_knowledges, app_knowledges


# 向量化每个知识元
def vectorize_knowledge(knowledge_list):
    response = client.embeddings.create(
        model="embedding-3",  # 填写需要调用的模型编码
        input=knowledge_list,  # 直接传入合并后的文本
        dimensions=256,  # 输出向量维度
    )
    embedding_list = [item.embedding for item in response.data]

    # 检查并处理 NaN 值
    for i, embedding in enumerate(embedding_list):
        if np.any(np.isnan(embedding)):  # 检查向量中是否有 NaN
            print(f"Warning: NaN found in embedding {i}. Replacing with zero vector.")
            embedding_list[i] = [0.0] * 256  # 用零向量替代 NaN

    return embedding_list


# 处理每个知识元，按编号分割
def process_knowledge(knowledge_list):
    # 按照 "1.", "2.", "3." 的模式分割知识元
    knowledge_items = re.split(r'\d+\.', knowledge_list)  # 按照 1. 2. 3. 分割
    knowledge_items = [item.strip() for item in knowledge_items if item.strip()]  # 去掉空项
    return knowledge_items


# 计算向量间的距离，求多样性和集中度
def calculate_vector_metrics(vectors):
    # 计算每个知识元的L2范数
    norms = np.linalg.norm(vectors, axis=1)

    # 计算最大/最小L2范数
    max_norm = np.max(norms)
    min_norm = np.min(norms)

    # 计算语义多样性 (平均距离)
    similarities = cosine_similarity(vectors)  # 计算余弦相似度
    diversity = 1 - np.mean(similarities)  # 多样性可以通过1减去平均相似度来衡量

    # 计算语义集中度 (计算各个向量的平均方向差异)
    centroid = np.mean(vectors, axis=0)  # 向量中心
    centroid_norm = np.linalg.norm(centroid)
    concentratedness = np.mean(np.dot(vectors, centroid) / (np.linalg.norm(vectors, axis=1) * centroid_norm))

    return norms, max_norm, min_norm, diversity, concentratedness


def main(file_path):
    ids, abstracts, tech_knowledges, app_knowledges = read_patent_data(file_path)

    tech_vectors = []
    app_vectors = []

    # 单独存储每个指标的列表
    tech_norms = []
    tech_max_norms = []
    tech_min_norms = []
    tech_diversities = []
    tech_concentratednesses = []

    app_norms = []
    app_max_norms = []
    app_min_norms = []
    app_diversities = []
    app_concentratednesses = []

    # 向量化技术知识元
    for idx, tech_knowledge in enumerate(tech_knowledges, start=1):
        print(f"正在处理第 {idx} 个专利的技术知识元")  # 打印正在处理的摘要
        tech_item_list = process_knowledge(tech_knowledge)  # 按照 1. 2. 3. 分割技术知识元
        tech_item_vectors = vectorize_knowledge(tech_item_list)  # 向量化每个知识元
        tech_vectors.append(tech_item_vectors)  # 保存向量

        # 计算技术知识元指标
        norms, max_norm, min_norm, diversity, concentratedness = calculate_vector_metrics(tech_item_vectors)
        tech_norms.append(norms)
        tech_max_norms.append(max_norm)
        tech_min_norms.append(min_norm)
        tech_diversities.append(diversity)
        tech_concentratednesses.append(concentratedness)

    # 向量化应用知识元
    for idx, app_knowledge in enumerate(app_knowledges, start=1):
        print(f"正在处理第 {idx} 个专利的应用知识元")  # 打印正在处理的摘要
        app_item_list = process_knowledge(app_knowledge)  # 按照 1. 2. 3. 分割应用知识元
        app_item_vectors = vectorize_knowledge(app_item_list)  # 向量化
        app_vectors.append(app_item_vectors)  # 保存向量

        # 计算应用知识元指标
        norms, max_norm, min_norm, diversity, concentratedness = calculate_vector_metrics(app_item_vectors)
        app_norms.append(norms)
        app_max_norms.append(max_norm)
        app_min_norms.append(min_norm)
        app_diversities.append(diversity)
        app_concentratednesses.append(concentratedness)

    # 保存结果到新的 DataFrame
    result_df = pd.DataFrame({
        '序号': ids,
        '专利摘要': abstracts,
        '技术知识元': tech_knowledges,
        '技术知识元向量': tech_vectors,  # 每个知识元有一个列表
        '技术知识元数量': [len(item) for item in tech_vectors],
        '技术知识元L2范数': tech_norms,
        '技术知识元最大L2范数': tech_max_norms,
        '技术知识元最小L2范数': tech_min_norms,
        '技术知识元语义多样性': tech_diversities,
        '技术知识元语义集中度': tech_concentratednesses,
        '应用知识元': app_knowledges,
        '应用知识元向量': app_vectors,  # 每个知识元有一个列表
        '应用知识元数量': [len(item) for item in app_vectors],
        '应用知识元L2范数': app_norms,
        '应用知识元最大L2范数': app_max_norms,
        '应用知识元最小L2范数': app_min_norms,
        '应用知识元语义多样性': app_diversities,
        '应用知识元语义集中度': app_concentratednesses,
    })

    # 保存结果为 Excel 文件
    result_df.to_excel(r"C:\Users\hhhxj\Desktop\专利数据\语义特征（改进）\2指标计算\语义指标_less_than_10_years_unit(10)(去除空值).xlsx",
                       index=False)
    print("处理完成，结果已保存！")


if __name__ == '__main__':
    file_path = r"C:\Users\hhhxj\Desktop\专利数据\4抽取的知识元\less_than_10_years_unit(10)(去除空值).xlsx"
    main(file_path)
