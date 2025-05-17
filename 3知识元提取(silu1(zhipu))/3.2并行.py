from zhipuai import ZhipuAI
import pandas as pd

api_key1 = 'f04ed8dece0817c5df88425f54de769f.rj1KGfIk4nVOEwyd'  # 请替换为你的APIKey

# 请填写您自己的APIKey
client = ZhipuAI(api_key=api_key1)

# 读取Excel文件中的专利摘要 (从你提供的读取函数中来)
def read_patent_abstracts(file_path):
    # 使用 read_excel 读取 Excel 文件
    df = pd.read_excel(file_path, header=0, engine='openpyxl')  # 使用 openpyxl 引擎来读取 .xlsx 文件
    # 获取第F列的数据（第6列，索引为5）
    abstracts = df.iloc[:, 3].tolist()
    ids = df.iloc[:, 0].tolist()  # 获取专利ID
    return ids,abstracts


# 调用智谱API来识别专利中的方案、功效、应用知识元
def get_patent_knowledge(abstract):
    response = client.chat.completions.create(
        model="glm-4-flash:1653547437:patent10:ihljpf7s",  # 使用你希望的模型
        messages=[
            {"role": "system",
             "content": """
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
             },
            {"role": "user",
             "content": f"专利摘要：\n{abstract}\n请提取以下两类知识元：1.技术知识元，2.应用知识元。请用【技术知识元】和【应用知识元】标记每类知识元。"
                        }
        ],
    )

    # 检查请求是否成功并返回结果
    if response and response.choices:
        return response.choices[0].message.content
    else:
        print(f"请求失败，响应内容: {response}")
        return None


# 分离出方案、功效和应用知识元
def separate_knowledge(knowledge_text):
    tech_knowledge = ""
    application_knowledge = ""

    if knowledge_text:
        # 按照标记拆分知识元
        if "【技术知识元】" in knowledge_text:
            tech_knowledge = knowledge_text.split("【技术知识元】")[1].split("【应用知识元】")[0].strip()
        if "【应用知识元】" in knowledge_text:
            application_knowledge = knowledge_text.split("【应用知识元】")[1].strip()

    return tech_knowledge, application_knowledge


# 主函数，读取Excel并逐条处理专利摘要
def main(file_path):
    ids,abstracts = read_patent_abstracts(file_path)
    tech_knowledges = []
    application_knowledges = []

    for idx, abstract in enumerate(abstracts):
        print(f"正在处理第 {idx + 1} 条摘要...")  # 只输出处理的进度
        knowledge = get_patent_knowledge(abstract)

        if knowledge:
            tech_knowledge, application_knowledge = separate_knowledge(knowledge)
            tech_knowledges.append(tech_knowledge)
            application_knowledges.append(application_knowledge)
        else:
            tech_knowledges.append("无")
            application_knowledges.append("无")

    # 将识别出的知识元分列保存回Excel，按照图片中的格式
    df = pd.DataFrame({
        '序号':ids,
        "专利摘要": abstracts,
        "技术知识元": tech_knowledges,
        "应用知识元": application_knowledges
    })

    df.to_excel(r"C:\Users\hhhxj\Desktop\专利数据\4抽取的知识元\less_than_or_equal_10_years_unit(10).xlsx", index=False)
    print("所有专利摘要处理完成，结果已保存")


# 运行主函数，传入Excel文件路径
if __name__ == "__main__":
    file_path = r"C:\Users\hhhxj\Desktop\专利数据\2抽取数据\less_than_or_equal_10_years.xlsx"  # 替换为你的Excel文件路径
    main(file_path)
