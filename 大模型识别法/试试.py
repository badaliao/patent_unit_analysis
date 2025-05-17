import pandas as pd
import requests
from zhipuai import ZhipuAI
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 智谱API配置（替换为你的API密钥）
API_KEY = "f04ed8dece0817c5df88425f54de769f.rj1KGfIk4nVOEwyd"
API_URL = "	https://open.bigmodel.cn/api/paas/v4/chat/completions"

# System Prompt：定义模型角色和特征解释
system_prompt = """
你是一个专利价值评估专家，任务是根据给定的专利特征判断其是否为高价值专利（1=高价值，0=低价值）。请基于以下特征和额外指标进行推理，并提供分类结果和简要理由。

### 特征解释：
1. **权利要求数**：专利的权利要求总数，通常越多表示保护范围广，可能价值更高。
2. **独立权利要求数量**：独立权利要求的数量，越多可能表明技术创新点多，价值较高。
3. **简单同族个数**：同一专利的简单同族数量，反映专利在单一技术方向的扩展，数量适中可能价值高。
4. **扩展同族个数**：扩展同族数量，反映专利在多技术领域的应用，越多可能价值越高。
5. **引证数量**：专利引用的其他专利数量，较多可能表明技术基础扎实。
6. **发明人数量**：参与发明的人数，较多可能表示团队协作和技术复杂度高。
7. **IPC数量**：国际专利分类（IPC）的数量，越多表示技术覆盖面广，价值可能更高。
8. **转让次数**：专利被转让的次数，频繁转让可能反映市场需求高，价值较高。
9. **被引证数量**：专利被其他专利引用的次数，越多表示技术影响力大，通常价值高。

### 网络指标（基于技术知识元语义相似度构建的网络）：
1. **度中心性（Degree Centrality）**：表示技术知识元在网络中的连接数量，值越高说明该专利涉及的技术节点在语义网络中更活跃，可能反映技术广泛性或重要性，预示价值较高。
2. **接近中心性（Closeness Centrality）**：衡量技术知识元到网络中其他节点的平均距离，值越高说明知识元更接近网络中心，可能表示技术整合能力强，价值可能更高。
3. **介数中心性（Betweenness Centrality）**：表示技术知识元在网络中连接不同群组的桥梁作用，值越高说明该专利可能连接不同技术领域，体现创新交叉性，预示价值较高。
4. **特征向量中心性（Eigenvector Centrality）**：衡量技术知识元与重要节点的连接质量，值越高说明连接的知识元更重要，可能反映技术核心性，预示价值更高。
5. **PageRank**：基于网络结构评估技术知识元的重要性，值越高说明该专利的技术知识元在语义网络中影响力大，通常与高价值专利相关。

### 额外指标（用于对比实验）：
1. **技术突破性**：专利是否涉及新颖技术（如新材料、新工艺），通过摘要或知识元判断，突破性强可能价值高。
2. **应用场景广度**：专利应用领域的多样性，覆盖多个场景（如5G、AI）可能价值更高。

### 判断标准：
- 根据高价值专利的特点进行判断，如权利要求数、同族数量、引证数量、发明人数量、转让次数、被引证数量等。
- 请尽量结合所有特征综合判断，避免只依赖单一指标。

### 输出格式：
分类：1 或 0
"""
# 理由：简要说明你的判断依据

# 函数：调用智谱API
def call_zhipu_api(features):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    user_prompt = f"""
专利信息：
- 权利要求数量：{features['权利要求数量']}
- 独立权利要求数量：{features['独立权利要求数量']}
- 简单同族个数：{features['简单同族个数']}
- 扩展同族个数：{features['扩展同族个数']}
- 引证次数：{features['引证次数']}
- 发明人数量：{features['发明人数量']}
- IPC类数量：{features['IPC类数量']}
- 转让次数：{features['转让次数']}
- 被引证次数：{features['被引证次数']}
- 度中心性：{features['度中心性']}
- 接近中心性：{features['接近中心性']}
- 介数中心性：{features['介数中心性']}
- 特征向量中心性：{features['特征向量中心性']}
- PageRank：{features['Pagerank']}

请根据以上信息判断该专利是否为高价值专利（1=高价值，0=低价值），并说明理由。
"""
    payload = {
        "model": "GLM-Zero-Preview",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7  # 控制随机性，可调
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
        # 解析结果
        classification = 1 if "分类：1" in result else 0
        # reason = result.split("理由：")[-1] if "理由：" in result else "无理由"
        return classification
    else:
        print(f"API调用失败：{response.text}")
        return None, None


# 读取Excel模板（请补充路径和列名）
def load_excel_data(positive_path, negative_path):
    # 正样本和高价值标签（1）
    positive_df = pd.read_excel(positive_path)
    positive_df['True_Label'] = 1
    # 负样本和低价值标签（0）
    negative_df = pd.read_excel(negative_path)
    negative_df['True_Label'] = 0
    # 合并数据
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    return combined_df


# 主函数：处理数据并调用API
def process_patent_value(positive_path, negative_path, output_path):
    # 读取数据
    df = load_excel_data(positive_path, negative_path)

    # 假设Excel列名与特征一致，若不同请调整
    feature_columns = [
        "权利要求数量", "独立权利要求数量", "简单同族个数", "扩展同族个数",
        "引证次数", "发明人数量", "IPC类数量", "转让次数", "被引证次数",
        "度中心性", "接近中心性", "介数中心性", "特征向量中心性","Pagerank"
    ]

    # 添加预测结果列
    df['Predicted_Label'] = None
    # df['Reason'] = None

    # 对每行数据调用API
    for index, row in df.iterrows():
        features = row[feature_columns].to_dict()
        # abstract = row.get('摘要', '')  # 若Excel有摘要列
        pred_label= call_zhipu_api(features)
        df.at[index, 'Predicted_Label'] = pred_label
        # df.at[index, 'Reason'] = reason
        print(f"Processed row {index + 1}/{len(df)}")

    # 计算指标
    y_true = df['True_Label'].astype(int)
    y_pred = df['Predicted_Label'].astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 导出结果
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


# 示例运行（请替换路径）
if __name__ == "__main__":
    positive_excel = r"C:\Users\hhhxj\Desktop\专利数据\大模型判断\test\1.xlsx"  # 正样本Excel路径
    negative_excel = r"C:\Users\hhhxj\Desktop\专利数据\大模型判断\test\0.xlsx"   # 负样本Excel路径
    # positive_excel = r"C:\Users\hhhxj\Desktop\专利数据\语义特征（改进）\2指标计算\语义指标_greater_than_10_years_unit(10)(去除空值).xlsx"  # 正样本Excel路径
    # negative_excel = r"C:\Users\hhhxj\Desktop\专利数据\语义特征（改进）\2指标计算\语义指标_less_than_10_years_unit(10)(去除空值).xlsx"  # 负样本Excel路径
    output_excel = r"C:\Users\hhhxj\Desktop\专利数据\大模型判断\test.xlsx"  # 输出路径
    process_patent_value(positive_excel, negative_excel, output_excel)