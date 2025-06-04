
import torch
from tqdm import tqdm
import csv
import numpy as np

def testProcess(model,test_loader,device,criterion):
    model.load_state_dict(torch.load('best_model/best_model.pth'))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            user, movie,bert_vec,rating = [x.to(device) for x in batch]
            pred = model(user, movie,bert_vec)
            test_loss += criterion(pred, rating).item() * user.size(0)
    test_loss /= len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f}')