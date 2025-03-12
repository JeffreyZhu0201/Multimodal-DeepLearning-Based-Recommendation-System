'''
Author: Jeffrey Zhu 1624410543@qq.com
Date: 2025-03-09 12:54:31
LastEditors: Jeffrey Zhu 1624410543@qq.com
LastEditTime: 2025-03-09 12:54:43
FilePath: \SimpleRecommendationSystem\OnePlusOne\analysis.py
Description: File Description Here...

Copyright (c) 2025 by JeffreyZhu, All Rights Reserved. 
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取训练过程的损失数据
loss_data = pd.read_csv("OnePlusOne/loss_data.csv")[1:]
test_results = pd.read_csv("OnePlusOne/test_results.csv")

# 获取测试RMSE
test_rmse = test_results[test_results['metric'] == 'test_rmse']['value'].iloc[0]

# 创建图表
plt.figure(figsize=(12, 6))

# 计算并绘制训练和验证的RMSE
epochs = range(1, len(loss_data) + 1)
train_rmse = np.sqrt(loss_data['train_loss'] * 5.0**2)  # 使用sum_dis进行反归一化
val_rmse = np.sqrt(loss_data['val_loss'] * 5.0**2)

# 绘制训练和验证RMSE
plt.plot(epochs, train_rmse, 'b-', label='train RMSE', linewidth=1)
plt.plot(epochs, val_rmse, 'r-', label='val RMSE', linewidth=2)

# 绘制测试RMSE（水平线）
plt.axhline(y=test_rmse, color='g', linestyle='--', 
           label=f'test RMSE: {test_rmse:.4f}', linewidth=2)

# 自定义图表
plt.title('模型训练过程RMSE变化', fontsize=14)
plt.xlabel('训练轮次', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# 设置y轴范围，略微扩展以便更好地显示
y_min = min(min(train_rmse), min(val_rmse), test_rmse) * 0.9
y_max = max(max(train_rmse), max(val_rmse), test_rmse) * 1.1
plt.ylim(y_min, y_max)

# 添加网格和边距
plt.tight_layout()

# 保存图表
plt.savefig('OnePlusOne/training_process.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

# 打印训练过程的统计信息
print("\n训练统计信息：")
print(f"最终训练RMSE: {train_rmse.iloc[-1]:.4f}")
print(f"最终验证RMSE: {val_rmse.iloc[-1]:.4f}")
print(f"最终测试RMSE: {test_rmse:.4f}")
print(f"\n最佳验证RMSE: {min(val_rmse):.4f} (第{np.argmin(val_rmse)+1}轮)")
