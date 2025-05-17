import pandas as pd
import os

# 设置包含 Excel 文件的文件夹路径
folder_path = r'C:\Users\hhhxj\Desktop\专利数据\1原始数据'

# 用于存储寿命大于10年和小于等于10年的专利数量
count_greater_10_years = 0
count_less_equal_10_years = 0

# 遍历文件夹中的所有 Excel 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        # 读取 Excel 文件
        file_path = os.path.join(folder_path, filename)
        xls = pd.ExcelFile(file_path)

        # 遍历文件中的每个 sheet
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)

            # 处理 "专利寿命（月）" 列的数据
            df["专利寿命（月）"] = pd.to_numeric(df["专利寿命（月）"], errors='coerce')

            # 统计寿命大于 10 年（即大于 120 个月或为空的专利）
            count_greater_10_years += df[(df["专利寿命（月）"] > 120) | (df["专利寿命（月）"].isna())].shape[0]

            # 统计寿命小于等于 10 年（即小于等于 120 个月的专利）
            count_less_equal_10_years += df[df["专利寿命（月）"] <= 120].shape[0]

# 输出统计结果
print(f"寿命大于10年的专利总数: {count_greater_10_years}")
print(f"寿命小于等于10年的专利总数: {count_less_equal_10_years}")
