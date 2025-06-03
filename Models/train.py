
from tqdm import tqdm
import pandas as pd

def trainProcess(model, train_loader, device, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
        user, movie, bert_vec, rating = [x.to(device) for x in batch]
        optimizer.zero_grad()
        pred = model(user, movie, bert_vec)
        loss = criterion(pred, rating)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * user.size(0)

    train_loss /= len(train_loader.dataset)
    return train_loss
