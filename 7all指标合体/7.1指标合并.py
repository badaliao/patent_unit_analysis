import pandas as pd

#读取xlsx和csv文件
# xlsx_file_path=r"C:\Users\hhhxj\Desktop\专利数据\2抽取数据\less_than_or_equal_10_years.xlsx"
xlsx_file_path=r"C:\Users\hhhxj\Desktop\专利数据\2抽取数据\greater_than_10_years.xlsx"
# csv_file_path=r"C:\Users\hhhxj\Desktop\专利数据\7网络制作\技术大于0.85_网络指标.csv"
csv_file_path=r"C:\Users\hhhxj\Desktop\专利数据\7网络制作\应用大于0.85_网络指标.csv"

xlsx_df=pd.read_excel(xlsx_file_path)

csv_df=pd.read_csv(csv_file_path)
# csv_df2=pd.read_csv(csv_file_path2)

xlsx_first_column=xlsx_df.columns[0]
csv_first_column=csv_df.columns[0]

csv_df.set_index(csv_first_column,inplace=True)

for index, row in xlsx_df.iterrows():
    match_value = row[xlsx_first_column]  # 获取当前行的第一列值

    # 如果csv中有对应的值，则取csv中该行的所有列数据
    if match_value in csv_df.index:
        matched_row = csv_df.loc[match_value]
        # 将匹配到的数据合并到xlsx行
        xlsx_df.loc[index, matched_row.index] = matched_row.values
    else:
        # 如果没有匹配到数据，则设置为0
        xlsx_df.loc[index, matched_row.index] = 0

output_file_path=r"C:\Users\hhhxj\Desktop\专利数据\8数据合并\传统＋应用网络指标(大于10年).xlsx"
xlsx_df.to_excel(output_file_path,index=False)

print(f'数据合并完成，输出文件路径：{output_file_path}')
