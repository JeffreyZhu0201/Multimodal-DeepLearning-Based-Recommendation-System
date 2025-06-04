
import matplotlib.pyplot as plt
import  csv
import pandas as pd

# 直接使用当前工作目录

def main():
    csv_path = 'resultAnalyze/loss_data_bert.csv'

    # Read train_loss and test_loss from loss_data.csv
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    train_loss = df['train_loss']
    val_loss = df['val_loss']

    # Create the plot
    plt.figure(figsize=(8,6))
    # plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs,[tl for tl in train_loss],'p-',label="Train Loss")

    # plt.plot(epochs, test_loss, 'r-',label='val Loss')
    plt.plot(epochs,[tl for tl in val_loss],'p-',label="Validating Loss")
    plt.axhline(0.0337, color='g', linestyle='--', label=f'Test Loss: 0.0336')

    # Customize the plot
    plt.title('Training and Validating Loss Over Epochs')
    plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    plt.ylabel(ylabel='Loss')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()