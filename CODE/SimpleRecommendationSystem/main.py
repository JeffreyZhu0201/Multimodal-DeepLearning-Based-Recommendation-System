'''
Author: Jeffrey Zhu 1624410543@qq.com
Date: 2025-03-06 23:49:22
LastEditors: Jeffrey Zhu 1624410543@qq.com
LastEditTime: 2025-03-07 00:37:38
FilePath: \SimpleRecommendationSystem\main.py
Description: File Description Here...

Copyright (c) 2025 by JeffreyZhu, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from MovieLensDataset import user_to_idx,movie_to_idx,train_loader,test_loader
from NCF import NCF
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_users = len(user_to_idx)
num_movies = len(movie_to_idx)
model = NCF(num_users, num_movies).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="train", unit="epoch"):
        user = batch['user'].to(device)
        movie = batch['movie'].to(device)
        rating = batch['rating'].to(device)
        
        optimizer.zero_grad()
        pred = model(user, movie)
        loss = criterion(pred, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 验证步骤
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test", unit="epoch"):
            user = batch['user'].to(device)
            movie = batch['movie'].to(device)
            rating = batch['rating'].to(device)
            pred = model(user, movie)
            test_loss += criterion(pred, rating).item()
    
    print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}')