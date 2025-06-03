import torch
import torch.nn as nn
class Recommender(nn.Module):
    def __init__(self, num_users, num_movies, device, embedding_dim, hidden_dim, bert_dim, genre_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.device = device

        # 输入维度变为 user_emb + movie_emb + bert_vec + genre_vec
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dim + bert_dim + genre_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, user, movie, movie_feature_vec):
        user_emb = self.user_emb(user)
        movie_emb = self.movie_emb(movie)
        # 拼接 user_emb, movie_emb, bert+genre_vec
        x = torch.cat([user_emb, movie_emb, movie_feature_vec], dim=1).to(self.device)
        return self.fc(x).squeeze()