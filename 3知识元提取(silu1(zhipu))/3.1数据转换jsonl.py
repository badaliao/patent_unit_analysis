import json
def json_to_jsonl(json_file,jsonl_file):
     with open(json_file, 'r', encoding='utf-8') as f:
         data = json.load(f)
     with open(jsonl_file, 'w', encoding='utf-8') as f:
         for item in data:
             f.write(json.dumps(item, ensure_ascii=False) + '\n')

json_file = r'C:\Users\hhhxj\Desktop\专利数据\3知识元标注\converted_data_formatted(100).json'
output_file = r'C:\Users\hhhxj\Desktop\专利数据\3知识元标注\converted_data_formatted(100).jsonl'
json_to_jsonl(json_file,output_file)