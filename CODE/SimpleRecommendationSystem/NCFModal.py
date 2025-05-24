import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv

# 数据预处理
ratings = pd.read_csv('../Dataset/ml-32m/ratings.csv')
csv_file_path = 'loss_data.csv'
print("数据读取成功")
# 创建用户和电影映射字典
user_ids = ratings['userId'].unique()
user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
movie_ids = ratings['movieId'].unique()
movie_to_idx = {movie: idx for idx, movie in enumerate(movie_ids)}

# 转换为连续索引
ratings['user_idx'] = ratings['userId'].map(user_to_idx)
ratings['movie_idx'] = ratings['movieId'].map(movie_to_idx)

# 归一化评分到0-1范围
ratings['rating'] = ratings['rating'] / 5.0

# 数据集划分
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)


# 定义Dataset类
class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.movies[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float)
        )

# 创建数据加载器
batch_size = 2048

train_dataset = MovieLensDataset(train_df['user_idx'].values, 
                               train_df['movie_idx'].values,
                               train_df['rating'].values)
val_dataset = MovieLensDataset(val_df['user_idx'].values,
                             val_df['movie_idx'].values,
                             val_df['rating'].values)
test_dataset = MovieLensDataset(test_df['user_idx'].values,
                              test_df['movie_idx'].values,
                              test_df['rating'].values)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print("数据加载成功")

# 定义推荐模型
class Recommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(2*embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user, movie):
        user_emb = self.user_emb(user)
        movie_emb = self.movie_emb(movie)
        x = torch.cat([user_emb, movie_emb], dim=1)
        return self.fc(x).squeeze()

# 初始化模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

n_users = len(user_ids)
n_movies = len(movie_ids)

model = Recommender(n_users, n_movies).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
epochs = 20
best_val_loss = float('inf')
loss_array = []

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
        user, movie, rating = [x.to(device) for x in batch]
        optimizer.zero_grad()
        pred = model(user, movie)
        loss = criterion(pred, rating)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * user.size(0)
    train_loss /= len(train_loader.dataset)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoch {epoch+1} Validation'):
            user, movie, rating = [x.to(device) for x in batch]
            pred = model(user, movie)
            val_loss += criterion(pred, rating).item() * user.size(0)
    val_loss /= len(val_loader.dataset)
    
    loss_array.append([train_loss,val_loss])

    print(f'Epoch {epoch+1}:')
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['train_loss', 'val_loss'])  # 添加表头
        csv_writer.writerows(loss_array)

print("训练完成，开始测试阶段")

# 测试阶段
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        user, movie, rating = [x.to(device) for x in batch]
        pred = model(user, movie)
        test_loss += criterion(pred, rating).item() * user.size(0)
test_loss /= len(test_loader.dataset)

test_loss_csv = 'test_loss.csv'

with open(test_loss_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['test_loss'])  # 添加表头
    csv_writer.writerow([test_loss])  # 写入测试损失

print(loss_array)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test RMSE: {np.sqrt(test_loss * 5.0**2):.4f}')  # 反归一化后计算RMSE