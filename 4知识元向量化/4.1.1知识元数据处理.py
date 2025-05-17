import pandas as pd

def check_and_fill_empty_values(file_path,output_path):
    # 读取数据
    df = pd.read_excel(file_path,engine="openpyxl")
    # 初始化
    empty_count_c=0
    empty_count_d=0
    empty_rows_c=[]
    empty_rows_d=[]
    # 遍历每一行
    for index,(c_value,d_value) in enumerate(zip(df.iloc[:,2],df.iloc[:,3]),start=1):
        if pd.isna(c_value):
            empty_count_c+=1
            empty_rows_c.append(index)
            df.loc[index-1,df.columns[2]] = '无'
        if pd.isna(d_value):
            empty_count_d+=1
            empty_rows_d.append(index)
            df.loc[index-1,df.columns[3]] = '无'
        # 输出统计结果
    print(f"C列空值数量: {empty_count_c}, 空值行号: {empty_rows_c}")
    print(f"D列空值数量: {empty_count_d}, 空值行号: {empty_rows_d}")

    # 保存结果
    df.to_excel(output_path,index=False)
    print(f"结果已保存至{output_path}")

if __name__ == '__main__':
    #file_path = r"C:\Users\hhhxj\Desktop\专利数据\4抽取的知识元\less_than_or_equal_10_years_unit(10).xlsx"  # 原始文件路径
    file_path = r"C:\Users\hhhxj\Desktop\专利数据\4抽取的知识元\greater_than_10_years_unit(10).xlsx"  # 原始文件路径
    #output_path = r"C:\Users\hhhxj\Desktop\专利数据\4抽取的知识元\less_than_10_years_unit(10)(去除空值).xlsx"  # 输出文件路径
    output_path = r"C:\Users\hhhxj\Desktop\专利数据\4抽取的知识元\greater_than_10_years_unit(10)(去除空值).xlsx"  # 输出文件路径
    check_and_fill_empty_values(file_path, output_path)