import pandas as pd
import os

# 设置包含 Excel 文件的文件夹路径
folder_path = r'C:\Users\hhhxj\Desktop\专利数据\1原始数据'

# 用于存储仅包含"发明申请"和"发明授权"的专利数据
df_invention_application = pd.DataFrame()
df_invention_granted = pd.DataFrame()

# 遍历文件夹中的所有 Excel 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        # 读取 Excel 文件
        file_path = os.path.join(folder_path, filename)
        xls = pd.ExcelFile(file_path)

        # 遍历文件中的每个 sheet
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)

            # 确保 "专利寿命（月）" 为数值
            df["专利寿命（月）"] = pd.to_numeric(df["专利寿命（月）"], errors='coerce')

            # 只保留 "发明申请" 和 "发明授权" 的数据
            df_filtered = df[df["专利类型"].isin(["发明申请", "发明授权"])]

            # 分别存入两个 DataFrame
            df_invention_application = pd.concat(
                [df_invention_application, df_filtered[df_filtered["专利类型"] == "发明申请"]], ignore_index=True)
            df_invention_granted = pd.concat(
                [df_invention_granted, df_filtered[df_filtered["专利类型"] == "发明授权"]], ignore_index=True)

# 统计发明申请和发明授权的总数量
total_invention_application = len(df_invention_application)
total_invention_granted = len(df_invention_granted)

# 按寿命筛选
df_greater_10_years = pd.concat(
    [df_invention_application[df_invention_application["专利寿命（月）"] > 120],
     df_invention_granted[df_invention_granted["专利寿命（月）"] > 120]], ignore_index=True)

df_less_equal_10_years = pd.concat(
    [df_invention_application[df_invention_application["专利寿命（月）"] <= 120],
     df_invention_granted[df_invention_granted["专利寿命（月）"] <= 120]], ignore_index=True)

# 随机抽取 2000 条寿命大于 10 年的专利
sample_greater_10 = df_greater_10_years.sample(n=min(2000, len(df_greater_10_years)), random_state=42)

# 随机抽取 2000 条寿命小于等于 10 年的专利
sample_less_equal_10 = df_less_equal_10_years.sample(n=min(2000, len(df_less_equal_10_years)), random_state=42)

# 将抽取的 4000 条数据合并
df_combined_4000 = pd.concat([sample_greater_10, sample_less_equal_10], ignore_index=True)

# 从剩下的专利中随机抽取 100 条（不区分寿命）
df_remaining = pd.concat([df_greater_10_years, df_less_equal_10_years]).drop(df_combined_4000.index, axis=0)
sample_remaining_100 = df_remaining.sample(n=min(100, len(df_remaining)), random_state=42)

# 统计抽取后的专利类型数量
count_application_in_sample = df_combined_4000[df_combined_4000["专利类型"] == "发明申请"].shape[0]
count_granted_in_sample = df_combined_4000[df_combined_4000["专利类型"] == "发明授权"].shape[0]

# 保存 Excel 文件
output_folder = r'C:\Users\hhhxj\Desktop\专利数据\2抽取数据'
os.makedirs(output_folder, exist_ok=True)

sample_greater_10.to_excel(os.path.join(output_folder, 'greater_than_10_years.xlsx'), index=False)
sample_less_equal_10.to_excel(os.path.join(output_folder, 'less_than_or_equal_10_years.xlsx'), index=False)
sample_remaining_100.to_excel(os.path.join(output_folder, 'remaining_100.xlsx'), index=False)

# 输出统计信息
print("统计结果：")
print(f"总 '发明申请' 数量: {total_invention_application}")
print(f"总 '发明授权' 数量: {total_invention_granted}")
print(f"抽取后 '发明申请' 数量: {count_application_in_sample}")
print(f"抽取后 '发明授权' 数量: {count_granted_in_sample}")

print("\n已保存 3 个 Excel 文件：")
print("1. greater_than_10_years.xlsx")
print("2. less_than_or_equal_10_years.xlsx")
print("3. remaining_100.xlsx")
