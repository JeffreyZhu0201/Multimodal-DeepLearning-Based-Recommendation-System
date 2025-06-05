import torch
import numpy as np
import pandas as pd
from Models.Recommender import Recommender

def load_recommendation_data():
    # 读取 ratings 数据
    ratings = pd.read_csv('dataset/ml-1m/ratings.dat', sep='::', engine='python',
                          names=['userId', 'movieId', 'rating', 'timestamp'])
    # 用户和电影稠密映射
    user_ids = ratings['userId'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    movie_ids = ratings['movieId'].unique()
    movie_to_idx = {movie: idx for idx, movie in enumerate(movie_ids)}
    # 加载 BERT+Genre 特征
    movie_bert_dict = np.load("dataset/movie_bert_dict.npy", allow_pickle=True).item()
    return ratings, user_to_idx, movie_to_idx, user_ids, movie_ids, movie_bert_dict

def generate_recommendations(user_id, model, movie_idx_list, movie_bert_dict, movie_ids, top_n=10, device='cpu'):
    model.eval()
    user_tensor = torch.tensor([user_id] * len(movie_idx_list), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(movie_idx_list, dtype=torch.long).to(device)
    feature_vecs = torch.stack([
        torch.tensor(movie_bert_dict[mid], dtype=torch.float) for mid in movie_ids
    ]).to(device)
    with torch.no_grad():
        scores = model(user_tensor, movie_tensor, feature_vecs)
    top_indices = torch.topk(scores, top_n).indices.cpu().numpy()
    recommended_movie_idxs = [movie_idx_list[i] for i in top_indices]
    return recommended_movie_idxs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用统一的数据读取函数
    ratings, user_to_idx, movie_to_idx, user_ids, movie_ids, movie_bert_dict = load_recommendation_data()
    num_users = len(user_ids)
    num_movies = len(movie_ids)
    embedding_dim = 32
    hidden_dim = 128
    bert_dim = 768
    genre_dim = len(next(iter(movie_bert_dict.values()))) - bert_dim
    model = Recommender(num_users, num_movies, device, embedding_dim, hidden_dim, bert_dim, genre_dim).to(device)
    model.load_state_dict(torch.load("best_model/best_model.pth", map_location=device))
    # 推荐时用稠密 user/movie 索引
    user_id = 1  # 原始 userId
    user_idx = user_to_idx[user_id]
    # movie_ids: 原始movieId列表
    # movie_idx_list: 稠密索引列表
    movie_idx_list = [movie_to_idx[mid] for mid in movie_ids]
    recommendations = generate_recommendations(user_idx, model, movie_idx_list, movie_bert_dict, movie_ids, top_n=10, device=device)
    # 将稠密索引映射回原始movieId
    inv_movie_to_idx = {v: k for k, v in movie_to_idx.items()}
    recommended_movie_ids = [inv_movie_to_idx[idx] for idx in recommendations]
    print(f"Top 10 recommended movie IDs for user {user_id}: {recommended_movie_ids}")