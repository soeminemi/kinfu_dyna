import numpy as np
from scipy.stats import pearsonr

def calculate_correlation():
    # 直接从npy文件加载数据和标签
    train_data = np.load('./data/train_data.npy')
    train_labels = np.load('./data/train_labels.npy')
    
    measures = train_data[:, :21]
    groundtruth = train_labels[:, :21]
    errors = measures - groundtruth
    errors_per_dimension = np.mean(errors, axis=0)
    print('errors per dimension',errors_per_dimension)
    # 从measures中减去每个维度的平均误差
    corrected_measures = measures - errors_per_dimension
    cerror = corrected_measures-groundtruth
    # 计算cerror的MAE
    mae = np.mean(np.mean(np.abs(cerror), axis=0))
    print('MAE of cerror:', mae)
    
    
    # 计算相关性
    
    # 读取测量名称并找到腰围对应的索引
    measurement_names = np.load('./data/measurement_names.npy')
    waist_idx = np.where(measurement_names == '裤腰围')[0][0]
    
    # 加载需要计算的点的序号
    point_indices = np.loadtxt('../data/waist.mat', dtype=int)
    
    # 重塑数据以便于访问每个点的信息 (N, 点数, 4)
    num_points = train_data.shape[1] // 4
    reshaped_data = train_data.reshape(-1, num_points, 4)
    
    # 提取指定点的法向量偏差（第4个维度）
    normal_deviations = reshaped_data[:, point_indices, 3]
    
    # 计算每个样本的平均偏差 (N,)
    sample_mean_deviations = np.mean(normal_deviations, axis=1)
    
    # 获取体重数据（训练数据的最后一个维度）
    weights = train_data[:, -1]
    
    # 将法向均值除以体重
    normalized_deviations = sample_mean_deviations / weights
    
    # 获取腰围数据
    waist_train = train_data[:, waist_idx]  # 训练数据中的腰围值
    waist_label = train_labels[:, waist_idx]  # 标签中的腰围值
    waist_diff = waist_label - waist_train  # 计算差值
    
    # 计算相关性
    correlation, p_value = pearsonr(normalized_deviations, waist_diff)
    
    print(f"归一化法向量偏差（除以体重）与腰围差值的相关性: {correlation:.4f}")
    print(f"P值: {p_value:.4f}")
    
    # 输出一些基本统计信息
    print(f"\n归一化法向量偏差的统计信息:")
    print(f"平均值: {np.mean(normalized_deviations):.4f}")
    print(f"标准差: {np.std(normalized_deviations):.4f}")
    print(f"最小值: {np.min(normalized_deviations):.4f}")
    print(f"最大值: {np.max(normalized_deviations):.4f}")
    
    print(f"\n腰围差值的统计信息:")
    print(f"平均值: {np.mean(waist_diff):.4f}")
    print(f"标准差: {np.std(waist_diff):.4f}")
    print(f"最小值: {np.min(waist_diff):.4f}")
    print(f"最大值: {np.max(waist_diff):.4f}")
    
    # # 并列输出每个样本的数据
    # print("\n样本数据对比:")
    # print("样本序号   归一化法向偏差   腰围差值")
    # print("-" * 40)
    # for i in range(len(waist_diff)):
    #     print(f"{i:6d}   {normalized_deviations[i]:10.4f}   {waist_diff[i]:10.4f}")

if __name__ == '__main__':
    calculate_correlation()
