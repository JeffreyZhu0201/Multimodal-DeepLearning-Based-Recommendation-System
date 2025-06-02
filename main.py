

from resultAnalyze.functions import loss_writter
from Models.eval import evalProcess
from Models.train import trainProcess
from Models.test import testProcess
from utils.loadDevice import loadDevice
from utils.downloadMovieLens import transferToDataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from Models.Recommender import Recommender
import os
import pandas as pd

batch_size = 2048
embedding_dim = 32
hidden_dim = 32
learning_rate = 0.001
epochs = 20
criterion = nn.MSELoss()
loss_path = 'resultAnalyze/loss_data.csv'

def main():
    device = loadDevice()
    
    train_loader,val_loader,test_loader,user_ids,movie_ids = transferToDataLoader(batch_size,slide=0.1)


    model = Recommender(len(user_ids),len(movie_ids),device,embedding_dim=32,hidden_dim=32).to(device)
    
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("已加载保存的模型参数")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss = trainProcess(model,train_loader,device,optimizer,criterion,epoch)
        val_loss = evalProcess(model,val_loader,device,criterion,epoch)

        loss_writter(epoch,train_loss,val_loss,loss_path)

    testProcess(model,test_loader,device,criterion)

if __name__ == "__main__":
    main()
