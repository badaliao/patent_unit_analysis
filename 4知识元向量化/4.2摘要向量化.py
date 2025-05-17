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
    return ids, abstracts


# 向量化每个知识元
def vectorize_knowledge(abstracts):
    response = client.embeddings.create(
        model="embedding-3",  # 填写需要调用的模型编码
        input=abstracts,  # 直接传入合并后的文本
        dimensions=256,  # 输出向量维度
    )
    embedding_list = [item.embedding for item in response.data]

    # 检查并处理 NaN 值
    for i, embedding in enumerate(embedding_list):
        if np.any(np.isnan(embedding)):  # 检查向量中是否有 NaN
            print(f"Warning: NaN found in embedding {i}. Replacing with zero vector.")
            embedding_list[i] = [0.0] * 256  # 用零向量替代 NaN

    return embedding_list



def main(file_path):
    ids, abstracts = read_patent_data(file_path)

    abs_vectors = []


    # 向量化摘要
    for idx, abstract in enumerate(abstracts, start=1):
        print(f"正在处理第 {idx} 个摘要")  # 打印正在处理的摘要
        abs_item_vectors = vectorize_knowledge([abstract])  # 向量化每个知识元
        abs_vectors.append(abs_item_vectors)  # 保存向量



    # 保存结果到新的 DataFrame
    result_df = pd.DataFrame({
        '序号': ids,
        '专利摘要': abstracts,
        '摘要向量': abs_vectors,
    })

    # 保存结果为 Excel 文件
    result_df.to_excel(r"C:\Users\hhhxj\Desktop\专利数据\5向量化\5.1摘要向量化\vectorized_less_than_or_equal_10_years_unit(10)(去除空值)(摘要向量化).xlsx",
                       index=False)
    print("处理完成，结果已保存！")


if __name__ == '__main__':
    file_path = r"C:\Users\hhhxj\Desktop\专利数据\4抽取的知识元\less_than_10_years_unit(10)(去除空值).xlsx"
    main(file_path)
