import urllib.request
import zipfile
import os

url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
dataset_dir = './dataset'
extract_path = os.path.join(dataset_dir, 'ml-1m')
save_path = os.path.join(dataset_dir, 'ml-1m.zip')

# 创建dataset文件夹
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

if not os.path.exists(extract_path):
    urllib.request.urlretrieve(url, save_path)
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("下载并解压完成")
else:
    print("数据已存在")
    