import torch
import numpy as np
import os
import logging
import csv
import sys
from predictHW import PredictionModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_new_model(input_size, output_size, device):
    """创建新模型"""
    model = PredictionModel(input_size, output_size).to(device)
    return model

def load_or_create_model(model_path, input_size, output_size, device):
    """加载或创建模型"""
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = PredictionModel(input_size, output_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    else:
        print("Creating new model")
        return PredictionModel(input_size, output_size)

def load_param_names(csv_file):
    """加载参数名称"""
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            # 跳过ID、类型和名称列
            param_names = header[3:-1]  # 不包括最后一列
        return param_names
    except Exception as e:
        logging.error(f"Error loading parameter names from {csv_file}: {e}")
        return None

def load_data(file_path):
    """从alldata.txt加载数据"""
    # 读取真值数据和参数名
    target_data = {}
    param_names = []
    measure_data = {}
    
    print("Reading export.csv...")
    with open('./data/export.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        param_names = header[3:-1]  # 不包括最后一列
        for row in reader:
            if '*' in row[0]:
                continue
            if int(row[1]) != 0:
                measure_data[row[0].strip()+'-'+row[1].strip()] = [float(x.strip()) for x in row[3:-1]]
                print('add measure_data: ', row[0].strip())
                continue
            target_data[row[0].strip()] = [float(x.strip()) for x in row[3:-1]]
            
    print(f"Found {len(target_data)} entries in target_data")
    
    # 读取weights.txt
    logging.info("Reading weights.txt...")
    weight_data = {}
    try:
        with open('./data/weights.txt', 'r') as f:
            for line in f:
                name, weight = line.strip().split(' ')
                weight_data[name] = float(weight)
        logging.info(f"Found {len(weight_data)} entries in weight_data")
    except Exception as e:
        logging.error(f"Error reading weights.txt: {e}")
    
    # 读取需要预测的数据
    with open(file_path) as f:
        lines = f.readlines()
    
    input_datas = []
    output_datas = []
    matches = 0
    for line in lines:
        data = line.strip().split(' ', 1)
        name = data[0].split('-')[0]
        tdata = data[1].split(' ')
        print('datasize: ', len(tdata))
        if not "G" in name:
            print('ignore: ', name)
            continue
        if name in target_data and data[0] in measure_data:
            matches += 1
            indata = measure_data[data[0]] + [float(tdata[i]) for i in range(3, len(tdata), 4)]
            indata.append(target_data[name][-1])
            indata.append(weight_data[name] if name in weight_data else 0.0)
            print('append indata shape: ', len(indata))
            print('weight: ', weight_data[name] if name in weight_data else 0.0)
            input_datas.append(indata)
            output_datas.append(target_data[name][:])
            names.append(data[0])
        else:
            print(f"No match found for key: {data[0]}")

    print(f"\nMatched {matches} entries between files")
    
    logging.info(f"\nLoaded {len(input_data)} samples")
    if len(input_data) > 0:
        logging.info(f"Input data shape: {len(input_data[0])} features")
        logging.info(f"Ground truth shape: {len(output_datas[0])} measurements")
    
    return np.array(input_datas), names, np.array(output_datas), param_names

def predict_measurements(model, input_data, device, input_mean, input_std, label_mean, label_std):
    """使用模型预测测量值"""
    model.eval()
    with torch.no_grad():
        # 确保数据是tensor并在正确的设备上
        if isinstance(input_data, np.ndarray):
            input_data = torch.FloatTensor(input_data)
        input_data = input_data.to(device)
        
        # 标准化输入数据
        input_normalized = (input_data - input_mean) / (input_std + 1e-6)
        
        # 预测修正量
        corrections = model(input_normalized)
        
        # 反标准化
        corrections = corrections * label_std + label_mean
        
def format_prediction_table(param_names, predictions, ground_truth=None, sample_names=None):
    """格式化预测结果表格"""
    table_lines = []
    table_lines.append("-" * 90)
    total_mae = 0
    total_num = 0
    if ground_truth is not None:
        for sample_idx in range(len(predictions)):
            header = f"{'参数名称':20s}  {'预测值':>10s}  {'真实值':>10s}  {'误差':>10s}"
            table_lines.append(header)
            table_lines.append("-" * 90)
            sample_name = sample_names[sample_idx] if sample_names is not None else f"Sample {sample_idx + 1}"
            table_lines.append(f"Sample: {sample_name}")
            mse = 0
            mae = 0
            for i, param_name in enumerate(param_names):
                pred_val = float(predictions[sample_idx][i])
                true_val = float(ground_truth[sample_idx][i])
                error = abs(pred_val - true_val)
                mse += error ** 2
                mae += error
                table_lines.append(f"{param_name:20s}  {pred_val:10.2f}  {true_val:10.2f}  {error:10.2f}")
            mse /= len(param_names)
            mae /= len(param_names)
            table_lines.append(f"{'MSE':>74s}: {mse:.4f}")
            table_lines.append(f"{'MAE':>74s}: {mae:.4f}")
            table_lines.append("-" * 90)
            total_mae += mae
            total_num += 1
    else:
        header = f"{'参数名称':20s}  {'预测值':>10s}"
    table_lines.append(header)
    table_lines.append("-" * 90)
    
    for sample_idx in range(len(predictions)):
        sample_name = sample_names[sample_idx] if sample_names is not None else f"Sample {sample_idx + 1}"
        table_lines.append(f"Sample: {sample_name}")
        for i, param_name in enumerate(param_names):
            pred_val = float(predictions[sample_idx][i])
            table_lines.append(f"{param_name:20s}  {pred_val:10.2f}")
        table_lines.append("-" * 90)
    
    average_mae = total_mae / total_num
    table_lines.append(f"{'平均MAE':>74s}: {average_mae:.4f}")
    table_lines.append("-" * 90)
    return "\n".join(table_lines)

def main():
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 加载归一化参数
        norm_params = torch.load('./data/normalization_params.pt', map_location=device)
        data_mean = norm_params['data_mean'].to(device)
        data_std = norm_params['data_std'].to(device)
        label_mean = norm_params['label_mean'].to(device)
        label_std = norm_params['label_std'].to(device)
        
        # 加载或创建模型
        input_size = 41342  # 22 + 41318 + 2 (测量值 + 点云 + 身高体重)
        output_size = 22    # 22个测量值的修正量
        model = load_or_create_model('best_model_checkpoint.pth', input_size, output_size, device)
        model = model.to(device)
        model.eval()  # 设置为评估模式
        
        # 加载测试数据
        test_data, names, ground_truth, param_names = load_data('./data/alldata.txt')
        if len(test_data) == 0:
            logging.error("No test data found!")
            return
            
        print(f"Test data shape: {test_data.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
        print(f"Number of parameters: {len(param_names)}")
        
        # 数据预处理 - 确保所有数据都在正确的设备上
        test_data = torch.FloatTensor(test_data).to(device)
        test_data = (test_data - data_mean) / data_std
        
        if ground_truth is not None:
            ground_truth = torch.FloatTensor(ground_truth).to(device)
        
        # 进行预测
        with torch.no_grad():
            main_predictions = model(test_data)
            # 反归一化预测结果
            predictions = main_predictions * label_std + label_mean
            predictions = predictions.cpu().numpy()
            if ground_truth is not None:
                ground_truth = ground_truth.cpu().numpy()
                
        table = format_prediction_table(param_names, predictions, ground_truth, names)
        print(table)
        
        # 保存预测结果
        output_file = './data/predictions.npy'
        np.save(output_file, predictions)
        print(f"\nPredictions saved to: {output_file}")
        
        # 输出统计信息
        print("\nPrediction Statistics:")
        print(f"Number of samples: {len(predictions)}")
        print(f"Average correction: {np.mean(predictions):.4f}")
        print(f"Max correction: {np.max(predictions):.4f}")
        print(f"Min correction: {np.min(predictions):.4f}")
        print(f"Correction std: {np.std(predictions):.4f}")
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return

if __name__ == '__main__':
    main()
