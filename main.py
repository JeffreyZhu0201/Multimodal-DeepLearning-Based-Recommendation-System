

from resultAnalyze.functions import loss_writter
from Models.eval import evalProcess
from Models.train import trainProcess
from Models.test import testProcess
from utils.loadDevice import loadDevice
from Models.dataLoader import transferToDataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from Models.Recommender import Recommender
import os
import pandas as pd



def main():

    batch_size = 256
    embedding_dim = 32
    hidden_dim = 32
    learning_rate = 0.001
    epochs = 50
    best_val_loss = float('inf')
    criterion = nn.MSELoss()
    loss_path = 'resultAnalyze/loss_data.csv'
    best_model_path = 'best_model/best_model.pth'
    # Load Device
    device = loadDevice()
    
    train_loader,val_loader,test_loader,user_ids,movie_ids = transferToDataLoader(batch_size,train_slide=0.5,val_slide=0.3)


    model = Recommender(len(user_ids),len(movie_ids),device,embedding_dim=embedding_dim,hidden_dim=hidden_dim).to(device)
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("已加载保存的模型参数")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        train_loss = trainProcess(model,train_loader,device,optimizer,criterion,epoch)
        val_loss = evalProcess(model,val_loader,device,criterion,epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists('best_model'):
                os.makedirs('best_model')
            torch.save(model.state_dict(), best_model_path)

        loss_writter(epoch,train_loss,val_loss,loss_path)

    testProcess(model,test_loader,device,criterion)

if __name__ == "__main__":
    main()
    
