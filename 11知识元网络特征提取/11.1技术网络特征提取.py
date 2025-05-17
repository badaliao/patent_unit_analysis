import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx

# === 参数设置 ===
EMBED_DIM = 256  # 输出网络特征维度
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. 加载数据 ===
high_df = pd.read_excel(r"C:\Users\hhhxj\Desktop\专利数据\2抽取数据\greater_than_10_years.xlsx")         # 列名为“序号”
low_df = pd.read_excel(r"C:\Users\hhhxj\Desktop\专利数据\2抽取数据\less_than_or_equal_10_years.xlsx")      # 列名为“序号”
edge_df = pd.read_csv(r"C:\Users\hhhxj\Desktop\专利数据\7网络制作\技术大于0.85.csv")           # 列名为“序号1”、“序号2”、“相似度”

# === 2. 构建全节点集合并编码为连续ID ===
all_nodes = pd.concat([high_df, low_df], ignore_index=True).drop_duplicates()
all_nodes['node_id'] = range(len(all_nodes))
node_id_map = dict(zip(all_nodes['序号'], all_nodes['node_id']))

# === 3. 构建边列表 ===
edge_index = []
for _, row in edge_df.iterrows():
    if row['序号1'] in node_id_map and row['序号2'] in node_id_map:
        s = node_id_map[row['序号1']]
        t = node_id_map[row['序号2']]
        edge_index.append((s, t))
        edge_index.append((t, s))  # 无向图双边

# === 4. 创建图结构 ===
G = nx.Graph()
G.add_nodes_from(node_id_map.values())
G.add_edges_from(edge_index)
pyg_data = from_networkx(G)

# === 5. 初始化节点特征（one-hot，保证结构主导）===
x = torch.eye(len(G)).to(DEVICE)
pyg_data.x = x
pyg_data.edge_index = pyg_data.edge_index.to(DEVICE)

# === 6. 定义 GAT 模型 ===
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, 128, heads=4, concat=True)
        self.gat2 = GATConv(128 * 4, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

model = GAT(in_channels=x.size(1), out_channels=EMBED_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# === 7. 无监督训练模型（loss = 嵌入范数）===
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(pyg_data.x, pyg_data.edge_index)
    loss = torch.mean(out.norm(dim=1))  # 无监督loss：保持嵌入规范性
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === 8. 获取节点嵌入 ===
model.eval()
with torch.no_grad():
    node_embeddings = model(pyg_data.x, pyg_data.edge_index).cpu().numpy()

# === 9. 构建嵌入DataFrame，并转为一列字符串格式 ===
embedding_df = pd.DataFrame(node_embeddings)
embedding_df['node_id'] = embedding_df.index
inverse_map = {v: k for k, v in node_id_map.items()}
embedding_df['序号'] = embedding_df['node_id'].map(inverse_map)

# 转为字符串格式的嵌入列
embedding_df['网络特征'] = embedding_df.drop(columns=['node_id', '序号']).apply(
    lambda row: str(list(row)), axis=1)

embedding_df_final = embedding_df[['序号', '网络特征']]

# === 10. 合并输出Excel（一个单元格存一行嵌入）===
def merge_and_save_single_col(orig_df, name):
    df = pd.merge(orig_df, embedding_df_final, on='序号', how='left')
    df.to_excel(f"{name}_with_embedding.xlsx", index=False)

merge_and_save_single_col(high_df, "high_value")
merge_and_save_single_col(low_df, "low_value")
