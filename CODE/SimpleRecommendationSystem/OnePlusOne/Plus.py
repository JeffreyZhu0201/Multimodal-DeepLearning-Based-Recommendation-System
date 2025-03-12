'''
Author: Jeffrey Zhu 1624410543@qq.com
Date: 2025-03-08 23:50:16
LastEditors: Jeffrey Zhu 1624410543@qq.com
LastEditTime: 2025-03-09 16:28:24
FilePath: \SimpleRecommendationSystem\OnePlusOne\Plus.py
Description: File Description Here...

Copyright (c) 2025 by JeffreyZhu, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 读取数据
data = pd.read_csv('OnePlusOne/plus_data.csv')

# 计算归一化参数
a_dis = float(data['a'].max() - data['a'].min())
b_dis = float(data['b'].max() - data['b'].min())
sum_dis = float(data['sum'].max() - data['sum'].min())
a_min = data['a'].min()
b_min = data['b'].min()
sum_min = data['sum'].min()

class PlusDataset(Dataset):
    """加法数据集"""
    def __init__(self, a, b, sum_val):
        self.a = torch.FloatTensor(a)
        self.b = torch.FloatTensor(b)
        self.sum = torch.FloatTensor(sum_val)
        
    def __len__(self):
        return len(self.a)
    
    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.sum[idx]

class PlusModel(nn.Module):
    """简单的加法模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, a, b):
        x = torch.stack([a, b], dim=1)
        return self.fc(x)

def train_model():
    """训练模型"""
    # 创建或清空loss_data.csv文件
    with open('OnePlusOne/loss_data.csv', 'w', newline='') as f:
        writer = pd.DataFrame({'epoch': [], 'train_loss': [], 'val_loss': []}).to_csv(f, index=False)
    
    # 数据预处理
    data['a_norm'] = (data['a'] - a_min) / a_dis
    data['b_norm'] = (data['b'] - b_min) / b_dis
    data['sum_norm'] = (data['sum'] - sum_min) / sum_dis

    # 划分数据集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # 创建数据加载器
    batch_size = 64
    train_dataset = PlusDataset(
        train_data['a_norm'].values,
        train_data['b_norm'].values,
        train_data['sum_norm'].values
    )
    val_dataset = PlusDataset(
        val_data['a_norm'].values,
        val_data['b_norm'].values,
        val_data['sum_norm'].values
    )
    test_dataset = PlusDataset(
        test_data['a_norm'].values,
        test_data['b_norm'].values,
        test_data['sum_norm'].values
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = PlusModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # 训练循环
    epochs = 300
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for a, b, sum_true in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            a, b, sum_true = a.to(device), b.to(device), sum_true.to(device)
            
            optimizer.zero_grad()
            sum_pred = model(a, b)
            loss = criterion(sum_pred, sum_true.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for a, b, sum_true in val_loader:
                a, b, sum_true = a.to(device), b.to(device), sum_true.to(device)
                sum_pred = model(a, b)
                val_loss += criterion(sum_pred, sum_true.unsqueeze(1)).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # 保存训练和验证损失
        with open('OnePlusOne/loss_data.csv', 'a', newline='') as f:
            pd.DataFrame({
                'epoch': [epoch + 1],
                'train_loss': [train_loss],
                'val_loss': [val_loss]
            }).to_csv(f, header=False, index=False)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'OnePlusOne/best_model.pth')
    
    # 测试阶段
    model.load_state_dict(torch.load('OnePlusOne/best_model.pth'))
    model.eval()
    test_loss = 0.0
    test_rmse = 0.0
    
    with torch.no_grad():
        for a, b, sum_true in test_loader:
            a, b, sum_true = a.to(device), b.to(device), sum_true.to(device)
            sum_pred = model(a, b)
            test_loss += criterion(sum_pred, sum_true.unsqueeze(1)).item()
    
    test_loss /= len(test_loader)
    test_rmse = np.sqrt(test_loss * sum_dis**2)
    
    # 保存测试误差
    test_results = pd.DataFrame({
        'metric': ['test_loss', 'test_rmse'],
        'value': [test_loss, test_rmse]
    })
    test_results.to_csv('OnePlusOne/test_results.csv', index=False)
    
    print(f'\nTest Loss: {test_loss:.6f}')
    print(f'Test RMSE: {test_rmse:.4f}')

if __name__ == '__main__':
    train_model()
