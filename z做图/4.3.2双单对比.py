import matplotlib.pyplot as plt
import numpy as np

# 数据准备
models = ['XGBoost', 'SVM', 'MLP', 'RF', 'KNN']

# 实验二、三、四的四个指标数据
exp2_accuracy = [0.8387, 0.9000, 0.9125, 0.7638, 0.8425]
exp3_accuracy = [0.8888, 0.9163, 0.9500, 0.8237, 0.8788]
exp4_accuracy = [0.9325, 0.9475, 0.9600, 0.8562, 0.9387]

exp2_precision = [0.8728, 0.8958, 0.9151, 0.8042, 0.8957]
exp3_precision = [0.9012, 0.9871, 0.9548, 0.8557, 0.9333]
exp4_precision = [0.9319, 0.9524, 0.9599, 0.8717, 0.9794]

exp2_recall = [0.8128, 0.9171, 0.9194, 0.7299, 0.7938]
exp3_recall = [0.8863, 0.9502, 0.9502, 0.8009, 0.8294]
exp4_recall = [0.9408, 0.9479, 0.9645, 0.8531, 0.9028]

exp2_f1 = [0.8417, 0.9063, 0.9173, 0.7652, 0.8417]
exp3_f1 = [0.8937, 0.9229, 0.9525, 0.8274, 0.8783]
exp4_f1 = [0.9363, 0.9501, 0.9622, 0.8623, 0.9396]

# 设置柱状图
x = np.arange(len(models))
width = 0.25

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig.suptitle('Performance Metrics Comparison Across Experiments', fontsize=16)

# Accuracy
axs[0, 0].bar(x - width, exp2_accuracy, width, label='Exp 2 (Tech)', color='skyblue')
axs[0, 0].bar(x, exp3_accuracy, width, label='Exp 3 (App)', color='lightgreen')
axs[0, 0].bar(x + width, exp4_accuracy, width, label='Exp 4 (Tech+App)', color='salmon')
axs[0, 0].set_title('Accuracy', fontsize=12)
axs[0, 0].set_ylim(0.7, 1.0)
axs[0, 0].grid(True, linestyle='--', alpha=0.7)

# Precision
axs[0, 1].bar(x - width, exp2_precision, width, label='Exp 2 (Tech)', color='skyblue')
axs[0, 1].bar(x, exp3_precision, width, label='Exp 3 (App)', color='lightgreen')
axs[0, 1].bar(x + width, exp4_precision, width, label='Exp 4 (Tech+App)', color='salmon')
axs[0, 1].set_title('Precision', fontsize=12)
axs[0, 1].set_ylim(0.7, 1.0)
axs[0, 1].grid(True, linestyle='--', alpha=0.7)

# Recall
axs[1, 0].bar(x - width, exp2_recall, width, label='Exp 2 (Tech)', color='skyblue')
axs[1, 0].bar(x, exp3_recall, width, label='Exp 3 (App)', color='lightgreen')
axs[1, 0].bar(x + width, exp4_recall, width, label='Exp 4 (Tech+App)', color='salmon')
axs[1, 0].set_title('Recall', fontsize=12)
axs[1, 0].set_ylim(0.7, 1.0)
axs[1, 0].grid(True, linestyle='--', alpha=0.7)

# F1 Score
axs[1, 1].bar(x - width, exp2_f1, width, label='Exp 2 (Tech)', color='skyblue')
axs[1, 1].bar(x, exp3_f1, width, label='Exp 3 (App)', color='lightgreen')
axs[1, 1].bar(x + width, exp4_f1, width, label='Exp 4 (Tech+App)', color='salmon')
axs[1, 1].set_title('F1 Score', fontsize=12)
axs[1, 1].set_ylim(0.7, 1.0)
axs[1, 1].grid(True, linestyle='--', alpha=0.7)

# 设置x轴标签和图例
for ax in axs.flat:
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend()

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()