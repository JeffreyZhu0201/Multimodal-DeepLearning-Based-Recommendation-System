import pandas as pd


def loss_writter(epoch,train_loss,val_loss,csv_path):
    # 保存训练损失和验证损失到同一个csv
    df = pd.DataFrame({
        'epoch': [epoch + 1],
        'train_loss': [train_loss],
        'val_loss': [val_loss] if val_loss is not None else [None]
    })
    # 如果文件不存在则写入表头，否则追加
    if epoch == 0:
        df.to_csv(csv_path, index=False, mode='w')
    else:
        df.to_csv(csv_path, index=False, mode='a', header=False)

def loss_reader():
    pass