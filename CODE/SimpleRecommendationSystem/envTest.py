
import torch

# 检查PyTorch和CUDA信息
print("="*40)
print("PyTorch Environment Summary")
print("="*40)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA不可用，当前运行在CPU模式")

# 检查cuDNN版本（需CUDA支持）
if torch.cuda.is_available():
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")

# 检查默认数据类型（CPU/GPU）
print(f"默认张量类型: {torch.get_default_dtype()}")