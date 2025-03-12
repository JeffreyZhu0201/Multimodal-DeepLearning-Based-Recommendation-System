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

# Sample data (replace with your actual data)
train_loss = [0.0269, 0.0266,0.0263,0.0260, 0.0258]
test_loss = [0.0278, 0.0277, 0.0275, 0.0273, 0.0272]

# Create epochs array
epochs = range(1, len(train_loss) + 1)

# Create the plot
plt.figure(figsize=(10,6))
# plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs,[tl* 5.0**2 for tl in train_loss],'p-',label="Train RMSE")

# plt.plot(epochs, test_loss, 'r-',label='val Loss')
plt.plot(epochs,[tl* 5.0**2 for tl in test_loss],'p-',label="Validating RMSE")

plt.plot(epochs,[0.8255 for i in range(5)],'p-',label="Testing RMSE")

# Customize the plot
plt.title('Training and Validating Loss Over Epochs')
plt.xlabel('Epochs')
# plt.ylabel('Loss')
plt.ylabel(ylabel='RMSE')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
