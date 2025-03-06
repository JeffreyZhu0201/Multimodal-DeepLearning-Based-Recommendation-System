
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 加载评分数据
ratings = pd.read_csv('ml-latest/ratings.csv')
# 创建用户和电影ID的映射
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_to_idx = {old: new for new, old in enumerate(user_ids)}
movie_to_idx = {old: new for new, old in enumerate(movie_ids)}
ratings['userId'] = ratings['userId'].map(user_to_idx)
ratings['movieId'] = ratings['movieId'].map(movie_to_idx)

# 划分训练集和测试集
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)


class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['userId'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'movie': self.movies[idx],
            'rating': self.ratings[idx]
        }

# 创建数据加载器
batch_size = 1024
train_dataset = MovieLensDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MovieLensDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=batch_size)