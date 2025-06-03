import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings,movie_ids, movie_bert_dict):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        self.movie_ids = movie_ids
        self.movie_bert_dict = movie_bert_dict

    def __getitem__(self, idx):
        user = torch.tensor(self.users[idx], dtype=torch.long)
        movie = torch.tensor(self.movies[idx], dtype=torch.long)
        rating = torch.tensor(self.ratings[idx], dtype=torch.float)
        # 获取 bert+genre 向量
        feature_vec = torch.tensor(self.movie_bert_dict[int(self.movie_ids[int(self.movies[idx])])], dtype=torch.float)
        return user, movie, feature_vec, rating
    
    def __len__(self):
        return len(self.users)