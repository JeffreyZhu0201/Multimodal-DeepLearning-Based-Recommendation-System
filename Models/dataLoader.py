
import pandas as pd
from sklearn.model_selection import train_test_split

from Dataset import MovieLensDataset
from torch.utils.data import Dataset, DataLoader

def loadDataset():
    # 数据预处理
    ratings = pd.read_csv('../dataset/ml-1m/ratings.csv')
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
    return ratings,user_to_idx,movie_to_idx,user_ids,movie_ids



def transferToDataLoader(batch_size):
    # 数据集划分
    ratings,user_to_idx,movie_to_idx,user_ids,movie_ids = loadDataset()

    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    # 创建数据加载器

    train_dataset = MovieLensDataset(train_df['user_idx'].values, 
                                train_df['movie_idx'].values,
                                train_df['rating'].values)
    val_dataset = MovieLensDataset(val_df['user_idx'].values,
                                val_df['movie_idx'].values,
                                val_df['rating'].values)
    test_dataset = MovieLensDataset(test_df['user_idx'].values,
                                test_df['movie_idx'].values,
                                test_df['rating'].values)
    
    print(train_dataset[0])  # 打印第一个样本以验证数据集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,pin_memory=True)
    print("数据加载成功")

    return train_loader,val_loader,test_loader,user_ids,movie_ids