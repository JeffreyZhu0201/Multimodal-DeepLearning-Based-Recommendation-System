
import pandas as pd
from sklearn.model_selection import train_test_split

from Models.Dataset import MovieLensDataset
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from utils.BERT import transferToBERT
def loadDataset():
    bert_dict_path = 'dataset/movie_bert_dict.npy'
    genre2idx_path = 'dataset/genre2idx.npy'
    # 数据预处理
    # 读取ratings.dat
    ratings = pd.read_csv('dataset/ml-1m/ratings.dat', sep='::', engine='python',
                    names=['userId', 'movieId', 'rating', 'timestamp'])
    
    # csv_file_path = 'loss_data.csv'
    print("数据读取成功")
    # 创建用户和电影映射字典,将稀疏数据稠密化
    user_ids = ratings['userId'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    movie_ids = ratings['movieId'].unique()
    movie_to_idx = {movie: idx for idx, movie in enumerate(movie_ids)}

    # 转换为连续索引
    ratings['user_idx'] = ratings['userId'].map(user_to_idx)
    ratings['movie_idx'] = ratings['movieId'].map(movie_to_idx)
    # 归一化评分到0-1范围
    ratings['rating'] = ratings['rating'] / 5.0

    # 检查 movie_bert_dict.npy 是否存在
    
    if os.path.exists(bert_dict_path):
        movie_bert_dict = np.load(bert_dict_path, allow_pickle=True).item()
        genre2idx = np.load(genre2idx_path,allow_pickle=True).item()
        len_genre2idx = len(genre2idx)
        print("movie_bert_dict 加载成功")
    else:
        print("movie_bert_dict.npy未找到,正在创建")
        movie_bert_dict,genre2idx = transferToBERT()
        len_genre2idx = len(genre2idx)
        if movie_bert_dict == None:
            print("movie_bert_dict.npy 创建失败")
        else:
            print("BERT向量创建成功")

    return ratings,user_to_idx,movie_to_idx,user_ids,movie_ids,movie_bert_dict,len_genre2idx


def transferToDataLoader(batch_size,train_slide,val_slide):
    # 数据集划分
    ratings,user_to_idx,movie_to_idx,user_ids,movie_ids,movie_bert_dict,len_genre2idx = loadDataset()

    train_df, test_df = train_test_split(ratings, test_size=1-train_slide-val_slide, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=1-(val_slide)/(train_slide+val_slide), random_state=42)

    # 创建数据加载器

    train_dataset = MovieLensDataset(
        train_df['user_idx'].values,
        train_df['movie_idx'].values,
        train_df['rating'].values,
        movie_ids,
        movie_bert_dict
    )
    val_dataset = MovieLensDataset(
        val_df['user_idx'].values,
        val_df['movie_idx'].values,
        val_df['rating'].values,
        movie_ids,
        movie_bert_dict,
    )
    test_dataset = MovieLensDataset(
        test_df['user_idx'].values,
        test_df['movie_idx'].values,
        test_df['rating'].values,
        movie_ids,
        movie_bert_dict
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,pin_memory=True)
    print("数据加载成功")

    return train_loader,val_loader,test_loader,user_ids,movie_ids,len_genre2idx