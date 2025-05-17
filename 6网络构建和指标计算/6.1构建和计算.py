import pandas as pd
import networkx as nx

file_path = r"C:\Users\hhhxj\Desktop\专利数据\7网络制作\技术大于0.85.csv"
df = pd.read_csv(file_path)

# 构建网络
G = nx.Graph()

for _, row in df.iterrows():
    G.add_edge(row['序号1'], row['序号2'], weight=row['相似度'])

# 1度中心性
degree_centrality = nx.degree_centrality(G)

# 2介数中心性
betweenness_centrality = nx.betweenness_centrality(G)

# 3接近中心性
closeness_centrality = nx.closeness_centrality(G)

# 4特征向量中心性
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

# 5Pagerank
pagerank = nx.pagerank(G, weight='weight')

centrality_df = pd.DataFrame({
    '节点': list(degree_centrality.keys()),
    '度中心性': pd.Series(degree_centrality),
    '介数中心性': pd.Series(betweenness_centrality),
    '接近中心性': pd.Series(closeness_centrality),
    '特征向量中心性': pd.Series(eigenvector_centrality),
    'Pagerank': pd.Series(pagerank)
})

output_path = r"C:\Users\hhhxj\Desktop\专利数据\7网络制作\技术大于0.85_网络指标.csv"
centrality_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f'网络指标计算完成，结果已保存至{output_path}')