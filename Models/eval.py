
import torch
from tqdm import tqdm

def evalProcess(model,val_loader,device,criterion,epoch):
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoch {epoch+1} Validation'):
            user, movie,bert_vec,rating = [x.to(device) for x in batch]
            pred = model(user, movie,bert_vec)
            val_loss += criterion(pred, rating).item() * user.size(0)
    val_loss /= len(val_loader.dataset)
    
    return val_loss