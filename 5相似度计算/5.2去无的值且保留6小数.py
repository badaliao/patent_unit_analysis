import pandas as pd


def process_similarity_column(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 处理相似度列：将大于 1 的值归为 1，其他四舍五入保留 6 位小数
    df['相似度'] = df['相似度'].apply(lambda x: 1 if x > 1 else round(x, 6))

    return df


def delete_rows(df, ids_to_delete, id_column1='序号1', id_column2='序号2'):
    # 删除特定行
    return df[~((df[id_column1].isin(ids_to_delete)) | (df[id_column2].isin(ids_to_delete)))]


def main():
    # 输入文件路径
    tech_file_path = r'C:\Users\hhhxj\Desktop\专利数据\6相似度计算\技术知识元向量相似度计算结果.csv'
    app_file_path = r'C:\Users\hhhxj\Desktop\专利数据\6相似度计算\应用知识元向量相似度计算结果.csv'

    # 处理技术知识元文件
    tech_df = process_similarity_column(tech_file_path)
    # 删除技术知识元文件中的特定行
    tech_df = delete_rows(tech_df, [37993, 37278])

    # 处理应用知识元文件
    app_df = process_similarity_column(app_file_path)
    # 删除应用知识元文件中的特定行
    app_df = delete_rows(app_df, [9816, 34939, 30168, 35833])

    # 保存处理后的文件为新的CSV
    new_tech_file_path = r'C:\Users\hhhxj\Desktop\专利数据\6相似度计算\处理后的技术知识元向量相似度计算结果.csv'
    new_app_file_path = r'C:\Users\hhhxj\Desktop\专利数据\6相似度计算\处理后的应用知识元向量相似度计算结果.csv'

    tech_df.to_csv(new_tech_file_path, index=False)
    app_df.to_csv(new_app_file_path, index=False)

    # 统计处理后的数据量
    print(f"技术知识元相似度文件数据量: {len(tech_df)}")
    print(f"应用知识元相似度文件数据量: {len(app_df)}")


if __name__ == '__main__':
    main()
