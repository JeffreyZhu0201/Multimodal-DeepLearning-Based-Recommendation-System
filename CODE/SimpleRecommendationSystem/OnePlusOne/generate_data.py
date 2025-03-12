'''
Author: Jeffrey Zhu 1624410543@qq.com
Date: 2025-03-09 16:16:23
LastEditors: Jeffrey Zhu 1624410543@qq.com
LastEditTime: 2025-03-12 21:12:00
FilePath: \SimpleRecommendationSystem\OnePlusOne\generate_data.py
Description: File Description Here...

Copyright (c) 2025 by JeffreyZhu, All Rights Reserved. 
'''
import pandas as pd
import numpy as np

def generate_plus_data():
    """生成加法训练数据，增加小数值样本"""
    data = []
    
    # 生成小数值区间的密集样本
    for a in np.linspace(1, 10, 100):
        for b in np.linspace(1, 10, 100):
            data.append([a, b, a + b])
    
    # 生成其他区间的稀疏样本
    for a in range(10, 1830, 2):
        for b in range(10, 1830, 2):
            data.append([a, b, a + b])
    
    df = pd.DataFrame(data, columns=['a', 'b', 'sum'])
    df.to_csv('OnePlusOne/plus_data.csv', index=False)

if __name__ == '__main__':
    generate_plus_data()