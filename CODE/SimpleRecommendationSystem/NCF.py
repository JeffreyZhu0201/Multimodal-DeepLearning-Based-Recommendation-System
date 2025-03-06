
import torch.nn as nn
import torch
class NCF(nn.Module):
    def __init__(self, num_users, num_movies, emb_dim=64, hidden_layers=[128, 64]):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.movie_emb = nn.Embedding(num_movies, emb_dim)
        
        # 构建MLP层
        layers = []
        input_dim = emb_dim * 2  # 用户和物品嵌入拼接
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user, movie):
        u_emb = self.user_emb(user)
        m_emb = self.movie_emb(movie)
        concat = torch.cat([u_emb, m_emb], dim=-1)
        return self.mlp(concat).squeeze()