import pandas as pd
import os
def read_multiple_excel(file_paths):
    dfs = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        dfs.append(df)
    return dfs

def drop_coulums(df, columns_to_drop):
    df.drop(columns=columns_to_drop, inplace=True)

# 处理IPC列，生成IPC类数量列
def add_ipc_count_column(df):
    df['IPC类数量']=df['IPC'].apply(lambda x: len(str(x).split(';')) if isinstance(x, str) else 0)
    return df

# 填充空值为0
def fill_empty_values(df, columns):
    for col in columns:
        df[col].fillna(0, inplace=True)
    return df

# 保存清洗后的数据到新的Excel文件
def save_to_excel(df, output_file):
    df.to_excel(output_file, index=False)

def main():
    file_paths = [r'C:\Users\hhhxj\Desktop\专利数据\8数据合并\传统＋应用网络指标(大于10年).xlsx',
                  r'C:\Users\hhhxj\Desktop\专利数据\8数据合并\传统＋应用网络指标(小于等于10年).xlsx',
                  r'C:\Users\hhhxj\Desktop\专利数据\8数据合并\传统＋技术网络指标(小于等于10年).xlsx',
                  r'C:\Users\hhhxj\Desktop\专利数据\8数据合并\传统＋技术网络指标(大于10年).xlsx',
                  ]
    columns_to_drop=['标题 (中文)','标题 (英文)','摘要 (中文)','摘要 (英文)','申请人','公开（公告）号','公开（公告）日','申请号',
                     '申请日','公开类型','专利类型','公开国别','链接到incoPat','首次公开日','失效日']
    ipc_column = 'IPC'  # IPC列的列名
    fill_columns = ['引证次数', '被引证次数','转让次数']  # 需要填充空值的列

    dfs=read_multiple_excel(file_paths)
    for i,df in enumerate(dfs):

        drop_coulums(df, columns_to_drop)

        df = add_ipc_count_column(df)
        # 填充空值
        df = fill_empty_values(df, fill_columns)

        # 获取原文件名并去除扩展名
        original_filename = os.path.basename(file_paths[i])  # 获取原文件名
        file_name_without_extension = os.path.splitext(original_filename)[0]  # 去掉扩展名

        # 构建输出文件名（以原文件名为基础，添加 "_cleaned" 后缀）
        output_file = f"C:/Users/hhhxj/Desktop/专利数据/9数据清洗/{file_name_without_extension}_cleaned.xlsx"

        # 保存处理后的数据到新的Excel文件
        save_to_excel(df, output_file)
        print(f"Processed {file_paths[i]} and saved as {output_file}")

if __name__ == '__main__':
    main()