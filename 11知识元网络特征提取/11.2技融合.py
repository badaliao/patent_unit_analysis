import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === 1. 加载数据 ===
high_df = pd.read_excel(r"C:\Users\hhhxj\Desktop\专利数据\z11技术网络特征\high_value_with_embedding.xlsx")
low_df = pd.read_excel(r"C:\Users\hhhxj\Desktop\专利数据\z11技术网络特征\low_value_with_embedding.xlsx")
high_df["标签"] = 1
low_df["标签"] = 0
df = pd.concat([high_df, low_df], ignore_index=True)

# === 2. 设置数值列名 ===
numeric_columns = ['权利要求数量', '独立权利要求数量', '发明人数量', '引证次数', '被引证次数',
                   '简单同族个数', '扩展同族个数', 'IPC类数量', '转让次数']

# === 3. 处理特征 ===
df["网络特征"] = df["网络特征"].apply(ast.literal_eval)
net_feat = np.vstack(df["网络特征"].values)
num_feat = df[numeric_columns].values

# 用 0 填充数值特征中的 NaN
num_feat = np.nan_to_num(num_feat, nan=0.0)

labels = df["标签"].values

# 标准化
sc_net = StandardScaler()
sc_num = StandardScaler()
net_feat_scaled = sc_net.fit_transform(net_feat)
num_feat_scaled = sc_num.fit_transform(num_feat)

# 检查数据合法性
assert not np.isnan(net_feat_scaled).any(), "网络特征包含NaN"
assert not np.isnan(num_feat_scaled).any(), "数值特征包含NaN"

# === 4. 构造 PyTorch 数据集 ===
X_net = torch.tensor(net_feat_scaled, dtype=torch.float32)
X_num = torch.tensor(num_feat_scaled, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_net, X_num, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# === 5. 定义模型 ===
class SupervisedAttentionFusion(nn.Module):
    def __init__(self, net_dim, num_dim, fusion_dim=64):
        super().__init__()
        self.att_net = nn.Sequential(
            nn.Linear(net_dim + num_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.reduce = nn.Linear(net_dim + num_dim, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, 1)

    def forward(self, net, num):
        combined = torch.cat([num, net], dim=1)
        att_weights = self.att_net(combined)
        w_num = att_weights[:, 0].unsqueeze(1)
        w_net = att_weights[:, 1].unsqueeze(1)
        fused = torch.cat([w_num * num, w_net * net], dim=1)
        reduced = self.reduce(fused)
        logits = self.classifier(reduced)
        return reduced, logits

model = SupervisedAttentionFusion(net_dim=256, num_dim=9, fusion_dim=64)

# 权重初始化（避免训练不稳定）
for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()

# === 6. 训练模型 ===
model.train()
for epoch in range(30):
    total_loss = 0
    for xb_net, xb_num, yb in loader:
        fused, pred = model(xb_net, xb_num)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 避免梯度爆炸
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# === 7. 获取融合特征（64维） ===
model.eval()
with torch.no_grad():
    fused_all, _ = model(X_net, X_num)
    fused_all = fused_all.numpy()

# === 8. 保存结果 ===
df["融合特征"] = [list(x) for x in fused_all]
high_df_out = df[df["标签"] == 1].drop(columns=["标签"])
low_df_out = df[df["标签"] == 0].drop(columns=["标签"])

high_df_out.to_excel(r"C:\Users\hhhxj\Desktop\专利数据\z12融合后的（技术）\high_value_fusion_supervised.xlsx", index=False)
low_df_out.to_excel(r"C:\Users\hhhxj\Desktop\专利数据\z12融合后的（技术）\low_value_fusion_supervised.xlsx", index=False)
