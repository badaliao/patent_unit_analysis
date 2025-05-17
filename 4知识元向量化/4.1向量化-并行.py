from zhipuai import ZhipuAI
import pandas as pd
import numpy as np
import re

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
    knowledge_items = re.split(r'\d+\.', knowledge_list)  # 按照 1. 2. 3. 分割
    knowledge_items = [item.strip() for item in knowledge_items if item.strip()]  # 去掉空项
    return knowledge_items


def main(file_path):
    ids, abstracts, tech_knowledges, app_knowledges = read_patent_data(file_path)

    tech_vectors = []
    app_vectors = []

    # 向量化技术知识元
    for idx, tech_knowledge in enumerate(tech_knowledges, start=1):
        print(f"正在处理第 {idx} 个摘要")  # 打印正在处理的摘要
        tech_item_vectors = vectorize_knowledge([tech_knowledge])  # 向量化每个知识元
        tech_vectors.append(tech_item_vectors)  # 保存向量

    # 向量化应用知识元
    for idx, app_knowledge in enumerate(app_knowledges, start=1):
        print(f"正在处理第 {idx} 个摘要")  # 打印正在处理的摘要
        app_item_vectors = vectorize_knowledge([app_knowledge])  # 向量化
        app_vectors.append(app_item_vectors)  # 保存向量

    # 保存结果到新的 DataFrame
    result_df = pd.DataFrame({
        '序号': ids,
        '专利摘要': abstracts,
        '技术知识元': tech_knowledges,
        '技术知识元向量': tech_vectors,
        '应用知识元': app_knowledges,
        '应用知识元向量': app_vectors,
    })

    # 保存结果为 Excel 文件
    result_df.to_excel(r"C:\Users\hhhxj\Desktop\专利数据\5向量化\vectorized_less_than_or_equal_10_years_unit(10)(去除空值).xlsx",
                       index=False)
    print("处理完成，结果已保存！")


if __name__ == '__main__':
    file_path = r"C:\Users\hhhxj\Desktop\专利数据\4抽取的知识元\less_than_10_years_unit(10)(去除空值).xlsx"
    main(file_path)
