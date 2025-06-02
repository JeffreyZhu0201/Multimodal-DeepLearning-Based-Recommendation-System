import torch
import torch.nn as nn

# 定义推荐模型
class Recommender(nn.Module):
    def __init__(self, num_users, num_movies, device, embedding_dim, hidden_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.device = device

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
        x = torch.cat([user_emb, movie_emb], dim=1).to(self.device)
        return self.fc(x).squeeze()