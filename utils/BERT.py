import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def transferToBERT():
    movie_df = pd.read_csv('dataset/ml-1m/movies.dat', sep='::', engine='python',
                    names=['movieId','movieName','movieGenres'],encoding='latin1')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 获取所有可能的genres
    all_genres = set()
    for genres in movie_df['movieGenres']:
        all_genres.update(genres.split('|'))
    all_genres = sorted(list(all_genres))
    genre2idx = {g: i for i, g in enumerate(all_genres)}
    genre_dim = len(all_genres)

    movie_bert_dict = {}

    for _, row in movie_df.iterrows():
        movie_id = int(row['movieId'])
        movie_name = row['movieName']
        genres = row['movieGenres']

        # BERT向量
        inputs = tokenizer(movie_name, return_tensors='pt', truncation=True, max_length=32)
        with torch.no_grad():
            outputs = model(**inputs)
            bert_vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 平均池化

        # Genres多热编码
        genre_vec = np.zeros(genre_dim, dtype=np.float32)
        for g in genres.split('|'):
            if g in genre2idx:
                genre_vec[genre2idx[g]] = 1.0

        # 拼接BERT和genre向量
        feature_vec = np.concatenate([bert_vec, genre_vec], axis=0)
        movie_bert_dict[movie_id] = feature_vec

    # 保存genre2idx以便后续使用
    np.save('dataset/movie_bert_dict.npy', movie_bert_dict)
    np.save('dataset/genre2idx.npy', genre2idx)
    return movie_bert_dict,genre2idx