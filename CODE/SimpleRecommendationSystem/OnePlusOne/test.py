import torch
import numpy as np
from Plus import PlusModel, a_dis, b_dis, sum_dis, a_min, b_min, sum_min

def load_model():
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlusModel()  # 使用简单模型
    model.load_state_dict(torch.load('OnePlusOne/best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict_sum(model, device, a: float, b: float) -> float:
    """
    预测两个数字的和
    Args:
        model: 加载的模型
        device: 计算设备
        a: 第一个数字
        b: 第二个数字
    Returns:
        预测的和
    """
    # 数据归一化
    a_normalized = np.clip((a - a_min) / a_dis, 0, 1)
    b_normalized = np.clip((b - b_min) / b_dis, 0, 1)
    
    # 转换为tensor并移至设备
    a_tensor = torch.tensor(a_normalized, dtype=torch.float32).to(device)
    b_tensor = torch.tensor(b_normalized, dtype=torch.float32).to(device)
    
    # 预测
    with torch.no_grad():
        pred_normalized = model(a_tensor.unsqueeze(0), b_tensor.unsqueeze(0))
        
    # 反归一化
    predicted_sum = pred_normalized.item() * sum_dis + sum_min
    
    return predicted_sum

def main():
    # 加载模型
    model, device = load_model()
    print(f"使用设备: {device}")
    print(f"输入范围: [{a_min:.1f}, {a_min + a_dis:.1f}]")
    
    while True:
        try:
            # 获取用户输入
            print("\n请输入两个数字(输入q退出):")
            user_input = input()
            
            if user_input.lower() == 'q':
                break
                
            # 解析输入
            a, b = map(float, user_input.split())
            
            # 验证输入范围
            if a < a_min or a > a_min + a_dis or b < b_min or b > b_min + b_dis:
                print(f"警告：输入值超出训练范围！")
            
            # 预测结果
            predicted = predict_sum(model, device, a, b)
            actual = a + b
            
            # 打印结果
            print(f"\n输入: {a} + {b}")
            print(f"预测值: {predicted:.4f}")
            print(f"实际值: {actual:.4f}")
            error = abs(predicted - actual)
            print(f"绝对误差: {error:.4f}")
            print(f"相对误差: {(error/actual)*100:.2f}%")
            
        except ValueError:
            print("输入格式错误！请输入两个数字，用空格分隔")
        except Exception as e:
            print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()