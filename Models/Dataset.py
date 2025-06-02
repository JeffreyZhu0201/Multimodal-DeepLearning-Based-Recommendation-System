
import torch
from torch.utils.data import Dataset

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
    