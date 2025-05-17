import json

def split_jsonl(jsonl_file, train_file, test_file, split_ratio=0.8):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 计算分割点
    split_point = int(len(lines) * split_ratio)

    # 分割数据
    train_data = lines[:split_point]
    test_data = lines[split_point:]

    # 写入训练数据
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_data)

    # 写入测试数据
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_data)

# 文件路径
jsonl_file = r'C:\Users\hhhxj\Desktop\专利数据\3知识元标注\converted_data_formatted(100).jsonl'
train_file = r'C:\Users\hhhxj\Desktop\专利数据\3知识元标注\train_data.jsonl'
test_file = r'C:\Users\hhhxj\Desktop\专利数据\3知识元标注\test_data.jsonl'

# 按 4:1 的比例分割数据
split_jsonl(jsonl_file, train_file, test_file, split_ratio=0.8)