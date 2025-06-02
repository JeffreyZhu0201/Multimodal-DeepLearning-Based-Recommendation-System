
import torch
import tqdm
import csv
import numpy as np

def testProcess(model,test_loader,device,criterion):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            user, movie, rating = [x.to(device) for x in batch]
            pred = model(user, movie)
            test_loss += criterion(pred, rating).item() * user.size(0)
    test_loss /= len(test_loader.dataset)

    print(test_loss)
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test RMSE: {np.sqrt(test_loss * 5.0**2):.4f}')  # 反归一化后计算RMSE