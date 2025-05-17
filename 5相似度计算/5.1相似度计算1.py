import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def read_patent_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    ids = df['序号'].values.tolist()  # 获取专利序号
    # 将技术知识元向量转换为列表，去掉额外的嵌套层级
    tech_knowledges = df['技术知识元向量'].apply(eval).apply(
        lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x).values.tolist()
    app_knowledges = df['应用知识元向量'].apply(eval).apply(
        lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x).values.tolist()
    return ids, tech_knowledges, app_knowledges


def calculate_cosine_similarity(ids, tech_knowledges):
    similarity_results = []
    total_pairs = len(tech_knowledges) * (len(tech_knowledges) - 1) // 2  # 总配对数
    current_pair = 0  # 当前处理的配对数
    # 遍历每对技术知识元
    for i in range(len(tech_knowledges)):
        for j in range(i + 1, len(tech_knowledges)):  # 避免与自己匹配
            similarity = cosine_similarity([tech_knowledges[i]], [tech_knowledges[j]])[0][0]
            similarity_results.append([ids[i], ids[j], similarity])  # 保存序号和相似度
            current_pair += 1
            # 每处理 10000 对打印一次进度
            if current_pair % 100000 == 0 or current_pair == total_pairs:
                print(
                    f"技术知识元相似度计算进度: {current_pair}/{total_pairs} ({(current_pair / total_pairs) * 100:.2f}%)")
    return similarity_results


def save_similarity_to_csv(results, output_path):
    results_df = pd.DataFrame(results, columns=['序号1', '序号2', '相似度'])
    results_df.to_csv(output_path, index=False)  # 保存为CSV文件
    print(f"相似度计算完成，结果已保存至: {output_path}")


def merge_and_calculate_similarity(file1, file2, tech_output_file, app_output_file):
    ids1, tech_knowledge1, app_knowledge1 = read_patent_data(file1)
    ids2, tech_knowledge2, app_knowledge2 = read_patent_data(file2)

    ids = ids1 + ids2
    tech_knowledges = tech_knowledge1 + tech_knowledge2
    app_knowledges = app_knowledge1 + app_knowledge2

    # 计算技术知识元相似度
    print("开始计算技术知识元相似度...")
    similarity_results = calculate_cosine_similarity(ids, tech_knowledges)
    save_similarity_to_csv(similarity_results, tech_output_file)

    # 计算应用知识元相似度
    print("开始计算应用知识元相似度...")
    similarity_results = calculate_cosine_similarity(ids, app_knowledges)
    save_similarity_to_csv(similarity_results, app_output_file)


def main(input_file1, input_file2, tech_output_file, app_output_file):
    merge_and_calculate_similarity(input_file1, input_file2, tech_output_file, app_output_file)


if __name__ == '__main__':
    input_file1 = r'C:\Users\hhhxj\Desktop\专利数据\5向量化\vectorized_greater_than_or_equal_10_years_unit(10)(去除空值).xlsx'  # 输入文件路径
    input_file2 = r'C:\Users\hhhxj\Desktop\专利数据\5向量化\vectorized_less_than_or_equal_10_years_unit(10)(去除空值).xlsx'  # 输入文件路径
    tech_output_file = r'C:\Users\hhhxj\Desktop\专利数据\6相似度计算\技术知识元向量相似度计算结果.csv'  # 输出文件路径
    app_output_file = r'C:\Users\hhhxj\Desktop\专利数据\6相似度计算\应用知识元向量相似度计算结果.csv'  # 输出文件路径
    main(input_file1, input_file2, tech_output_file, app_output_file)
