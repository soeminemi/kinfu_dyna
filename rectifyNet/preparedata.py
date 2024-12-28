from random import sample
import numpy as np
import torch
import os
import logging
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt

class DataNormalizer:
    """数据标准化处理类"""
    def __init__(self):
        self.data_mean = None
        self.data_std = None
    
    def fit(self, data):
        """计算标准化参数"""
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)
        # 防止除零
        self.data_std = np.where(self.data_std < 1e-6, 1.0, self.data_std)
    
    def transform(self, data):
        """标准化数据"""
        if self.data_mean is None or self.data_std is None:
            raise ValueError("必须先调用fit方法计算标准化参数")
        return (data - self.data_mean) / self.data_std
    
    def inverse_transform(self, normalized_data):
        """反标准化数据"""
        return normalized_data * self.data_std + self.data_mean
    
    def save(self, path):
        """保存标准化参数"""
        np.savez(path, data_mean=self.data_mean, data_std=self.data_std)
    
    def load(self, path):
        """加载标准化参数"""
        params = np.load(path)
        self.data_mean = params['data_mean']
        self.data_std = params['data_std']

def prepare_dataset(data_dir, output_dir, test_size=0.2, val_size=0.2, random_state=42):
    """准备数据集"""
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载原始数据
        input_data = np.load(os.path.join(data_dir, 'input_data.npy'))
        target_data = np.load(os.path.join(data_dir, 'target_data.npy'))
        
        logging.info(f"加载数据: 输入形状 {input_data.shape}, 目标形状 {target_data.shape}")
        
        # 创建标准化器
        normalizer = DataNormalizer()
        
        # 首先分割出测试集
        train_val_data, test_data = train_test_split(input_data, test_size=test_size,
                                                    random_state=random_state)
        train_val_target, test_target = train_test_split(target_data, test_size=test_size,
                                                        random_state=random_state)
        
        # 然后分割训练集和验证集
        train_data, val_data = train_test_split(train_val_data,
                                               test_size=val_size/(1-test_size),
                                               random_state=random_state)
        train_target, val_target = train_test_split(train_val_target,
                                                   test_size=val_size/(1-test_size),
                                                   random_state=random_state)
        
        # 使用训练集计算标准化参数
        normalizer.fit(train_data)
        
        # 标准化所有数据集
        train_data_norm = normalizer.transform(train_data)
        val_data_norm = normalizer.transform(val_data)
        test_data_norm = normalizer.transform(test_data)
        
        # 保存标准化参数
        normalizer.save(os.path.join(output_dir, 'normalization_params.npz'))
        
        # 保存处理后的数据集
        np.save(os.path.join(output_dir, 'train_data.npy'), train_data_norm)
        np.save(os.path.join(output_dir, 'train_labels.npy'), train_target)
        np.save(os.path.join(output_dir, 'val_data.npy'), val_data_norm)
        np.save(os.path.join(output_dir, 'val_labels.npy'), val_target)
        np.save(os.path.join(output_dir, 'test_data.npy'), test_data_norm)
        np.save(os.path.join(output_dir, 'test_labels.npy'), test_target)
        
        logging.info(f"数据集准备完成:")
        logging.info(f"训练集大小: {len(train_data)}")
        logging.info(f"验证集大小: {len(val_data)}")
        logging.info(f"测试集大小: {len(test_data)}")
        
    except Exception as e:
        logging.error(f"数据准备过程中出错: {str(e)}")
        raise

def split_data_by_name(input_datas, output_datas, param_names,person_names):
    """根据名字分割数据，带G的作为测试集，带E的作为验证集"""
    test_indices = []
    val_indices = []
    train_indices = []
    import random
    # Set random seed to ensure different randomness each time
    import time
    random.seed(int(time.time()))

    # 从person_names随机选择10个名字
    selected_names = random.sample(person_names, 15)
    
    # 为每个名字添加'-'后缀
    modified_names = [name + '-' for name in selected_names]
    
    # 将修改后的名字分为两个集合
    test_names = modified_names[:7]
    val_names = modified_names[7:]
    
    # test_names = ["G1-","G2-","G8-","G9-","G10-","G11-"]
    # val_names = ["E1-","E2-","E3-","E4-","E5-","E6-"]
    flag_apd = False
    for i, name in enumerate(param_names):
        flag_apd = False
        for val_name in val_names:
            if val_name in name:
                print('val_name: ', val_name, name)
                val_indices.append(i)
                flag_apd = True
                break
        for test_name in test_names:
            if test_name in name:
                test_indices.append(i)
                flag_apd = True
                break
        
        if not flag_apd:
            train_indices.append(i)
    
    # 转换为numpy数组
    test_indices = np.array(test_indices)
    val_indices = np.array(val_indices)
    train_indices = np.array(train_indices)
    
    print(f"数据集划分:")
    print(f"测试集（带G样本）: {len(test_indices)} 个样本")
    print(f"验证集（带E样本）: {len(val_indices)} 个样本")
    print(f"训练集: {len(train_indices)} 个样本")
    
    logging.info(f"数据集划分:")
    logging.info(f"测试集（带G样本）: {len(test_indices)} 个样本")
    logging.info(f"验证集（带E样本）: {len(val_indices)} 个样本")
    logging.info(f"训练集: {len(train_indices)} 个样本")
    
    return train_indices, val_indices, test_indices

# Configure logging to only show warnings and errors
logging.basicConfig(level=logging.WARNING)

# 检查文件是否存在
print("Checking data files...")
if not os.path.exists('./data/export_s.csv'):
    print("Error: export.csv not found!")
if not os.path.exists('./data/alldata.txt'):
    print("Error: alldata.txt not found!")

ignore_list = ["E2","E4","E6","E11","E12","E22","F1","F6","F8","F10","F13","F14","F24","F31","F41","G5","H5","H7","H13","H20","H25","H26"]
# 读取目标数据
print("Reading export.csv...")
with open('./data/export.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    param_names = header[3:-1]
    target_data = {}
    measure_data = {}
    person_names = []
    for row in reader:
        if '*' in row[0]:
            continue
        if row[0].strip() in ignore_list:
            print('ignore: ', row[0].strip())
            continue
        if int(row[1]) != 0:
            measure_data[row[0].strip()+'-'+row[1].strip()] = [float(x.strip()) for x in row[3:-1]]
            print('add measure_data: ', row[0].strip())
            continue
        person_names.append(row[0].strip())
        target_data[row[0].strip()] = [float(x.strip()) for x in row[3:-1]]
        
print(f"Found {len(target_data)} entries in target_data")
print("First key in target_data:", next(iter(target_data)) if target_data else "None")
print(target_data)
input("Press Enter to continue...")
# 读取体重数据
print("Reading weights.txt...")
weight_data = {}
try:
    with open('./data/weights.txt', 'r') as f:
        for line in f:
            print(line)
            name, weight = line.strip().split(' ')
            print(name, weight)
            weight_data[name] = float(weight)
    print(f"Found {len(weight_data)} entries in weight_data")
    print("First key in weight_data:", next(iter(weight_data)) if weight_data else "None")
except FileNotFoundError:
    print("Error: weights.txt not found!")
except ValueError:
    print("Error: Invalid format in weights.txt")
print('measure_data element dimension: ', len(next(iter(measure_data.values()))))
input("Press Enter to continue...")
# 读取输入数据
print("\nReading alldata.txt...")
with open('./data/alldata.txt') as f:
    lines = f.readlines()
print(f"Found {len(lines)} lines in alldata.txt")

input_datas = []
output_datas = []
matches = 0
sample_names = []
for line in lines:
    data = line.strip().split(' ', 1)
    name = data[0].split('-')[0]
    tdata = data[1].split(' ')
    print('datasize: ', len(tdata))
    # if "G" in name:
    #     print('ignore: ', name)
    #     continue
    if name in target_data and data[0] in measure_data:
        matches += 1
        indata = measure_data[data[0]] + [float(tdata[i]) for i in range(len(tdata))]
        indata.append(target_data[name][-1])
        indata.append(weight_data[name] if name in weight_data else 0.0)
        print('append indata shape: ', len(indata))
        print('weight: ', weight_data[name] if name in weight_data else 0.0)

        input_datas.append(indata)
        output_datas.append(target_data[name][:])
        sample_names.append(data[0])
        print("append data: ",data[0])
    else:
        print(f"No match found for key: {data[0]}")
input("Press any key to continue")
print(f"\nMatched {matches} entries between files")

input_datas = np.array(input_datas)
output_datas = np.array(output_datas)
print("\nArray shapes:")
print("input_data shape:", input_datas.shape)
print("output_data shape:", output_datas.shape)

if len(input_datas) == 0:
    print("\nPossible issues:")
    print("1. No matching keys between export.csv and alldata.txt")
    print("2. Data format mismatch")
    print("3. Empty or corrupted input files")
    exit(1)

input("Press Enter to continue...")
# 计算每个样本的数据和标签值的偏差，统计每个维度的MAE
mae_per_dimension = []
for i in range(22):  # 前22维为测量值
    errors = np.abs(input_datas[:, i] - output_datas[:, i])
    mae = np.mean(errors)
    mae_per_dimension.append(mae)

print("\nMAE for each dimension:")
for i, mae in enumerate(mae_per_dimension):
    print(f"Dimension {i}: {mae:.4f}")

# 计算总体MAE
overall_mae = np.mean(mae_per_dimension)
print(f"\nOverall MAE: {overall_mae:.4f}")

# 计算每个样本的MAE
mae_per_sample = np.abs(input_datas[:,0:22] - output_datas).mean(axis=1)

# 绘制MAE直方图
plt.figure(figsize=(10, 5))  # 设置图形大小
plt.hist(mae_per_sample, bins=np.arange(0, np.max(mae_per_sample) + 0.5, 0.5), edgecolor='black')
plt.title('Histogram of MAE per Sample')
plt.xlabel('MAE (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)


# 计算每个维度的误差
errors_per_dimension = (input_datas[:, 0:22] - output_datas)

# 计算每个维度的均值
mean_errors = np.mean(errors_per_dimension, axis=0)

# 计算每个维度的误差均值
errors_per_dimension_centered = errors_per_dimension - mean_errors
mean_mae = np.mean(np.abs(errors_per_dimension_centered), axis=0)

# 打印每个维度的均值和误差均值
print("\nMean Error for each dimension:")
for i, mean_error in enumerate(mean_errors):
    print(f"Dimension {i}: Mean Error = {mean_error:.4f}, Mean MAE = {mean_mae[i]:.4f}")

from pypinyin import pinyin, Style

def convert_to_pinyin(sample_names):
    pinyin_names = []
    for name in sample_names:
        # 将汉字转换为拼音，使用普通风格（不带声调）
        py = pinyin(name, style=Style.NORMAL)
        # 将拼音列表展平并用空格连接
        pinyin_name = ' '.join([item[0] for item in py])
        pinyin_names.append(pinyin_name)
    return pinyin_names

# 将样本名称转换为拼音
pinyin_param_names = convert_to_pinyin(param_names)

# 绘制每个维度的误差直方图
for i in range(errors_per_dimension.shape[1]):
    plt.figure(figsize=(10, 5))
    plt.hist(errors_per_dimension[:, i], bins=np.arange(-10, 10, 0.5), edgecolor='black')
    plt.title(f'Histogram of Errors for {pinyin_param_names[i]}')
    plt.xlabel('Error (cm)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'./results/errors_histogram_{i}.png')
    plt.close()
    
output_datas = output_datas[:, :-1]
print("output_data shape:", output_datas.shape)

# 根据名字分割数据
train_indices, val_indices, test_indices = split_data_by_name(
    input_datas, output_datas, sample_names,person_names
)

# 创建输出目录
os.makedirs('./data', exist_ok=True)
# 输出训练数据的第一个样本的数据
print("\nFirst sample of training data:")
print(input_datas[train_indices[0]])
print("\nCorresponding label:")
print(output_datas[train_indices[0]])

# 保存数据集
np.save('./data/train_data.npy', input_datas[train_indices][:, :])
np.save('./data/train_labels.npy', output_datas[train_indices])
np.save('./data/val_data.npy', input_datas[val_indices][:, :])
np.save('./data/val_labels.npy', output_datas[val_indices])
np.save('./data/test_data.npy', input_datas[test_indices][:, :])
np.save('./data/test_labels.npy', output_datas[test_indices])
np.save('./data/test_names.npy', np.array(sample_names)[test_indices])

# 保存维度名字
np.save('./data/measurement_names.npy', np.array(param_names))
