import json
import pandas as pd

# 加载数据
file_path = r'C:\Users\hhhxj\Desktop\专利数据\3知识元标注\知识元标注.xlsx'
data = pd.read_excel(file_path)

# 创建一个空列表来存储转换后的数据
converted_data = []

# 提示词
prompt = """
# Role:
Semiconductor Patent Knowledge Extractor
专注于半导体领域专利，熟悉了解半导体领域的技术与应用知识。
# Goals:
从半导体领域的专利摘要中提取技术知识元和应用知识元，帮助识别和整理专利中的技术创新和应用场景。
# Constraints:
回复中的每个知识元应清晰、简洁、明确，直接呈现技术知识元或应用知识元的核心要点，不需要添加总结或说明性文字，避免包含额外的文字说明。
# Skills:
请精准分析专利文本，区分不同技术和应用知识元，并从中提炼出核心的技术信息与市场应用信息，避免抽象或模糊的描述。
# 知识元：
知识元是不可再分的独立知识单元，具备完备的知识表达能力、稳定性和完整性。
# 技术知识元：
技术知识元描述了技术系统的重要组成部分和技术实施的主要方式，常用名词或名词性短语表示。
# 应用知识元：
应用知识元主要描述专利技术在实际世界中的具体应用，它不仅包括技术所涉及的行业或领域，还包括该技术如何在这些领域中被实际使用、解决了哪些问题。
# Output Format:
请直接输出知识元，每类知识元前使用【技术知识元】和【应用知识元】标记，同类多个知识元用序号区分。输出的结果不需额外添加任何解释、词语。
"""

# 遍历DataFrame的每一行
for index, row in data.iterrows():
    # 每一行的数据
    input_text = row['摘要 (中文)']
    tech_knowledge = row['技术知识元']
    app_knowledge = row['应用知识元']

    # 创建符合要求的格式
    entry = {
        "messages": [
            {"role": "system", "content": prompt},  # 将提示词放在系统消息中
            {"role": "user", "content":  f"专利摘要：{input_text}\n请提取以下两类知识元：1.技术知识元，2.应用知识元。\n不需生成额外字词。"},
            {"role": "assistant", "content": f"技术知识元：{tech_knowledge} 应用知识元：{app_knowledge}"}
        ]
    }

    # 将字典添加到结果列表中
    converted_data.append(entry)

# 将转换后的数据保存为JSON文件
output_file_path = r'C:\Users\hhhxj\Desktop\专利数据\3知识元标注\converted_data_formatted(100).json'
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(converted_data, json_file, ensure_ascii=False, indent=4)

print(f"数据已成功保存到 {output_file_path}")
