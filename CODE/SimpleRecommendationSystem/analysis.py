'''
Author: Jeffrey Zhu 1624410543@qq.com
Date: 2025-03-07 12:58:08
LastEditors: Jeffrey Zhu 1624410543@qq.com
LastEditTime: 2025-03-07 13:25:50
FilePath: \SimpleRecommendationSystem\analysis.py
Description: File Description Here...

Copyright (c) 2025 by JeffreyZhu, All Rights Reserved. 
import matplotlib.pyplot as plt
'''
import matplotlib.pyplot as plt
import  csv
import os
# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'loss_data.csv')

# Read train_loss and test_loss from loss_data.csv
train_loss = []
val_loss = []
with open(csv_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        train_loss.append(float(row['train_loss']))
        val_loss.append(float(row['val_loss']))

# Create epochs array
epochs = range(1, len(train_loss) + 1)

# Create the plot
plt.figure(figsize=(10,6))
# plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs,[tl* 5.0**2 for tl in train_loss],'p-',label="Train RMSE")

# plt.plot(epochs, test_loss, 'r-',label='val Loss')
plt.plot(epochs,[tl* 5.0**2 for tl in val_loss],'p-',label="Validating RMSE")


# Customize the plot
plt.title('Training and Validating Loss Over Epochs')
plt.xlabel('Epochs')
# plt.ylabel('Loss')
plt.ylabel(ylabel='RMSE')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
