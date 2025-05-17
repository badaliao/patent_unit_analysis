import pandas as pd


# 检查缺失值函数
def check_missing_values(file_path):
    # 读取数据
    data = pd.read_excel(file_path)

    # 找出包含缺失值的行
    missing_data = data[data.isnull().any(axis=1)]

    # 检查是否有缺失值
    if not missing_data.empty:
        print("以下行包含缺失值：")
        print(missing_data)  # 输出包含缺失值的行
    else:
        print("数据中没有缺失值。")


# 主程序
def main():
    # 输入文件路径
    positive_file = r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\传统＋技术网络指标(大于10年)_cleaned.xlsx'  # 替换为正样本的文件路径
    negative_file = r'C:\Users\hhhxj\Desktop\专利数据\9数据清洗\传统＋技术网络指标(小于等于10年)_cleaned.xlsx'  # 替换为负样本的文件路径

    print("检查正样本数据中的缺失值：")
    check_missing_values(positive_file)

    print("\n检查负样本数据中的缺失值：")
    check_missing_values(negative_file)


if __name__ == '__main__':
    main()
