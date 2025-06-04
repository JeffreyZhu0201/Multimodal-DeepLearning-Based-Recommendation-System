
import matplotlib.pyplot as plt
import  csv
import pandas as pd

# 直接使用当前工作目录

def main():
<<<<<<< HEAD
    csv_path = 'resultAnalyze/loss_data_main.csv'
=======
    csv_path = 'resultAnalyze/loss_data_bert.csv'
>>>>>>> BERT

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
<<<<<<< HEAD
    # Add a horizontal line for Test Loss
    test_loss_value = 0.0341
    # Test Loss: 0.0341
    # Test RMSE: 0.9234
    plt.axhline(y=test_loss_value, color='g', linestyle='--', label='Test Loss: {:.4f}'.format(test_loss_value))
=======
    plt.axhline(0.0337, color='g', linestyle='--', label=f'Test Loss: 0.0336')
>>>>>>> BERT

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