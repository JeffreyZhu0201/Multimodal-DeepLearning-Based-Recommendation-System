
import torch

def loadDevice():
    # 初始化模型和优化器
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if(torch.cuda.is_available()):
        torch.cuda.empty_cache()
        print("使用GPU加速")
    print(device)
    return device