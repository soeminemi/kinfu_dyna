import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
import pandas as pd
import torch.nn.functional as F
import copy
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BodyMeasurementRectifier(nn.Module):
    def __init__(self, measurement_dim=22, mesh_dim=27560, basic_dim=2):
        super(BodyMeasurementRectifier, self).__init__()
        self.measurement_dim = measurement_dim
        self.mesh_dim = mesh_dim
        self.basic_dim = basic_dim
        
        # 1. 点云特征提取（逐步降维）
        self.mesh_net = nn.Sequential(
            nn.Linear(mesh_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 2. 身高体重特征
        self.basic_net = nn.Sequential(
            nn.Linear(basic_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        
        # 3. 测量值处理
        self.measurement_net = nn.Sequential(
            nn.Linear(measurement_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # 4. 预测网络
        predict_input = 64 + 16 + 32  # 特征 + 原始测量值
        self.predict_net = nn.Sequential(
            nn.Linear(predict_input, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(64, measurement_dim),
            nn.Tanh()  # 限制偏差范围
        )
        
        # 可学习的缩放因子（每个维度独立）
        self.scale = nn.Parameter(torch.ones(measurement_dim) * 0.1)
        
        # L2正则化
        self.l2_reg = 0.001
    
    def forward(self, x):
        # 分离输入数据
        measurements = x[:, :self.measurement_dim]
        mesh_features = x[:, self.measurement_dim:-self.basic_dim]
        basic_features = x[:, -self.basic_dim:]
        
        # 1. 特征提取
        mesh_features = self.mesh_net(mesh_features)
        basic_features = self.basic_net(basic_features)
        measurement_features = self.measurement_net(measurements)
        
        # 2. 特征组合
        combined = torch.cat([
            mesh_features,
            basic_features,
            measurement_features
        ], dim=1)
        
        # 3. 预测偏差
        delta = self.predict_net(combined)
        
        # 4. 应用缩放因子并添加残差连接
        return measurements + delta * self.scale

class RegressionLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(RegressionLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, outputs, targets, delta, constraint_weights, model=None):
        # 基础损失
        mse_loss = self.mse(outputs, targets)
        mae_loss = self.mae(outputs, targets)
        
        # 偏差正则化（鼓励小的偏差）
        delta_reg = torch.mean(torch.abs(delta))
        
        # 组合损失
        loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss + 0.1 * delta_reg
        
        # L2正则化
        if model is not None:
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += model.l2_reg * l2_reg
        
        return loss

class DataNormalizer:
    """数据标准化处理类"""
    def __init__(self):
        self.data_mean = None
        self.data_std = None
        self.label_mean = None
        self.label_std = None
    
    def fit(self, data, labels):
        """计算标准化参数"""
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)
        self.label_mean = np.mean(labels, axis=0)
        self.label_std = np.std(labels, axis=0)
        
        # 防止除零
        self.data_std = np.where(self.data_std < 1e-6, 1.0, self.data_std)
        self.label_std = np.where(self.label_std < 1e-6, 1.0, self.label_std)
    
    def transform(self, data, labels=None):
        """标准化数据"""
        if self.data_mean is None or self.data_std is None:
            raise ValueError("必须先调用fit方法计算标准化参数")
        
        normalized_data = (data - self.data_mean) / self.data_std
        if labels is not None:
            normalized_labels = (labels - self.label_mean) / self.label_std
            return normalized_data, normalized_labels
        return normalized_data
    
    def inverse_transform_labels(self, normalized_labels):
        """反标准化标签"""
        return normalized_labels * self.label_std + self.label_mean
    
    def save(self, path):
        """保存标准化参数"""
        np.savez(path,
                 data_mean=self.data_mean,
                 data_std=self.data_std,
                 label_mean=self.label_mean,
                 label_std=self.label_std)
    
    def load(self, path):
        """加载标准化参数"""
        params = np.load(path)
        self.data_mean = params['data_mean']
        self.data_std = params['data_std']
        self.label_mean = params['label_mean']
        self.label_std = params['label_std']

class BodyParamDataset(Dataset):
    """人体参数数据集"""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        print(f"\n数据集信息:")
        print(f"样本数量: {len(self.data)}")
        print(f"输入维度: {self.data.shape[1]}")
        print(f"标签维度: {self.labels.shape[1]}")
        print("-" * 30)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def init_weights(m):
    """初始化网络权重"""
    if isinstance(m, nn.Linear):
        # He初始化
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """创建带预热的余弦学习率调度器"""
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦衰减阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(model, train_loader, val_loader, num_epochs=500, device='cuda'):
    """训练模型"""
    model = model.to(device)
    model.apply(init_weights)  # 初始化权重
    
    # 组合损失函数
    mse_criterion = RegressionLoss(0.9).to(device)
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
    
    # 计算总步数并设置预热步数
    total_steps = len(train_loader) * num_epochs
    warmup_steps = total_steps // 10  # 10%的步数用于预热
    
    # 使用带预热的余弦学习率调度器
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    best_val_loss = float('inf')
    best_model = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # 计算组合损失
            delta = outputs - batch_data[:, :22]
            constraint_weights = torch.ones_like(delta)  # 使用全1权重
            loss = mse_criterion(outputs, batch_labels, delta, constraint_weights, model)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_metrics = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                
                # 计算验证损失
                delta = outputs - batch_data[:, :22]
                constraint_weights = torch.ones_like(delta)  # 使用全1权重
                val_batch_loss = mse_criterion(outputs, batch_labels, delta, constraint_weights, model)
                
                val_loss += val_batch_loss.item()
                
                # 计算每个维度的误差
                dim_errors = torch.abs(outputs - batch_labels).mean(dim=0)
                val_metrics.append(dim_errors.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dim_errors = np.mean(val_metrics, axis=0)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        if epoch % 10 == 0:
            logging.info(f'Epoch {epoch}/{num_epochs}')
            logging.info(f'Train Loss: {avg_train_loss:.6f}')
            logging.info(f'Val Loss: {avg_val_loss:.6f}')
            logging.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            # 显示模型中的scale参数
            for name, param in model.named_parameters():
                if 'scale' in name:
                    if param.numel() == 1:
                        logging.info(f'{name}: {param.item():.6f}')
                    else:
                        logging.info(f'{name}: {param.mean().item():.6f} (mean)')
            
    
    # 训练结束后，在训练集上进行最终评估
    logging.info("Final Evaluation on Training Set:")
    train_final_loss = 0.0
    train_metrics = []
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            
            # 计算损失
            delta = outputs - batch_data[:, :22]
            constraint_weights = torch.ones_like(delta)
            train_batch_loss = mse_criterion(outputs, batch_labels, delta, constraint_weights, model)
            train_final_loss += train_batch_loss.item()
            
            # 计算每个维度的误差
            dim_errors = torch.abs(outputs - batch_labels).mean(dim=0)
            train_metrics.append(dim_errors.cpu().numpy())
    
    avg_train_final_loss = train_final_loss / len(train_loader)
    avg_train_dim_errors = np.mean(train_metrics, axis=0)
    
    # 在验证集上进行最终评估
    logging.info("Final Evaluation on Validation Set:")
    val_final_loss = 0.0
    val_metrics = []
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            
            # 计算损失
            delta = outputs - batch_data[:, :22]
            constraint_weights = torch.ones_like(delta)
            val_batch_loss = mse_criterion(outputs, batch_labels, delta, constraint_weights, model)
            val_final_loss += val_batch_loss.item()
            
            # 计算每个维度的误差
            dim_errors = torch.abs(outputs - batch_labels).mean(dim=0)
            val_metrics.append(dim_errors.cpu().numpy())
    
    avg_val_final_loss = val_final_loss / len(val_loader)
    avg_val_dim_errors = np.mean(val_metrics, axis=0)
    
    # 输出最终评估结果
    logging.info("\n" + "=" * 50)
    logging.info('Final Evaluation Results:')
    logging.info("=" * 50)
    logging.info(f'Final Training Loss: {avg_train_final_loss:.6f}')
    logging.info(f'Final Validation Loss: {avg_val_final_loss:.6f}')
    
    logging.info("\nTraining Set Errors by Dimension:")
    for i, error in enumerate(avg_train_dim_errors):
        logging.info(f'Dim {i}: {error:.4f}')
    
    logging.info("\nValidation Set Errors by Dimension:")
    for i, error in enumerate(avg_val_dim_errors):
        logging.info(f'Dim {i}: {error:.4f}')
    logging.info("=" * 50)
    
    return best_model, best_val_loss

def evaluate_model(model, data_loader, normalizer, device='cuda'):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_original = []  # 存储原始测量数据
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            # 保存原始测量数据（前22维）
            original_measurements = batch_data[:, :22]
            all_original.append(original_measurements.cpu().numpy())
            
            # 获取模型预测
            outputs = model(batch_data)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
    # 转换为numpy数组
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    original = np.concatenate(all_original)
    
    # 反标准化
    predictions = normalizer.inverse_transform_labels(predictions)
    labels = normalizer.inverse_transform_labels(labels)
    original = normalizer.inverse_transform_labels(original)
    
    # 计算网络修正后的误差
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    relative_error = np.mean(np.abs((predictions - labels) / labels)) * 100
    
    # 计算原始测量数据的误差
    original_mae = np.mean(np.abs(original - labels))
    original_rmse = np.sqrt(np.mean((original - labels) ** 2))
    original_relative_error = np.mean(np.abs((original - labels) / labels)) * 100
    
    # 计算每个维度的误差改善程度
    dimension_improvement = []
    for i in range(22):
        original_dim_mae = np.mean(np.abs(original[:, i] - labels[:, i]))
        corrected_dim_mae = np.mean(np.abs(predictions[:, i] - labels[:, i]))
        improvement = ((original_dim_mae - corrected_dim_mae) / original_dim_mae) * 100
        dimension_improvement.append(improvement)
    
    # 加载维度名称
    try:
        measurement_names = np.load('./data/measurement_names.npy').tolist()
    except:
        measurement_names = [f"维度{i+1}" for i in range(22)]
    
    logging.info("\n=== 评估结果（反标准化后的实际单位） ===")
    logging.info("\n原始测量数据误差：")
    logging.info(f"MAE: {original_mae:.4f}")
    logging.info(f"RMSE: {original_rmse:.4f}")
    logging.info(f"相对误差: {original_relative_error:.2f}%")
    
    logging.info("\n网络修正后误差：")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"相对误差: {relative_error:.2f}%")
    
    logging.info("\n误差改善：")
    logging.info(f"MAE改善: {((original_mae - mae) / original_mae * 100):.2f}%")
    logging.info(f"RMSE改善: {((original_rmse - rmse) / original_rmse * 100):.2f}%")
    logging.info(f"相对误差改善: {((original_relative_error - relative_error) / original_relative_error * 100):.2f}%")
    
    logging.info("\n各维度MAE改善：")
    for name, imp in zip(measurement_names, dimension_improvement):
        logging.info(f"{name}: {imp:.2f}%")
    
    # 保存详细结果到CSV
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建详细的DataFrame
    results_df = pd.DataFrame({
        '测量项': measurement_names,
        '原始MAE': [np.mean(np.abs(original[:, i] - labels[:, i])) for i in range(22)],
        '修正后MAE': [np.mean(np.abs(predictions[:, i] - labels[:, i])) for i in range(22)],
        '改善百分比': dimension_improvement,
        '原始RMSE': [np.sqrt(np.mean((original[:, i] - labels[:, i]) ** 2)) for i in range(22)],
        '修正后RMSE': [np.sqrt(np.mean((predictions[:, i] - labels[:, i]) ** 2)) for i in range(22)]
    })
    
    # 保存结果
    results_df.to_csv(os.path.join(results_dir, 'evaluation_results.csv'), 
                     index=False, encoding='utf-8-sig')
    
    return mae, rmse, relative_error, original_mae, original_rmse, original_relative_error

def visualize_model(model, save_path='model_architecture.png'):
    """可视化模型结构"""
    try:
        from torchviz import make_dot
        import torch
        
        # 创建示例输入
        batch_size = 1
        measurement_dim = 22
        mesh_dim = 27560
        basic_dim = 2
        x = torch.randn(batch_size, measurement_dim + mesh_dim + basic_dim)
        
        # 获取模型输出
        y = model(x)
        
        # 创建计算图可视化
        dot = make_dot(y, params=dict(model.named_parameters()))
        
        # 设置图形属性
        dot.attr(rankdir='TB')  # 从上到下的布局
        dot.attr('node', shape='box')
        
        # 保存图形
        dot.render(save_path, format='png', cleanup=True)
        logging.info(f"模型结构图已保存到 {save_path}.png")
        
        # 打印模型结构摘要
        logging.info("\n模型结构摘要:")
        logging.info("=" * 50)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"总参数量: {total_params:,}")
        logging.info(f"可训练参数量: {trainable_params:,}")
        logging.info("=" * 50)
        
        # 打印每层的参数量
        logging.info("\n每层参数统计:")
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                params = sum(p.numel() for p in module.parameters())
                logging.info(f"{name}: {params:,} 参数")
    
    except ImportError:
        logging.warning("请安装 torchviz: pip install torchviz")
        logging.warning("以及 graphviz: pip install graphviz")

def test_network_dimensions():
    """测试网络维度匹配"""
    try:
        # 创建模型
        model = BodyMeasurementRectifier()
        model.eval()
        
        # 创建测试输入
        batch_size = 2
        total_dim = 22 + 27560 + 2  # measurement_dim + mesh_dim + basic_dim
        x = torch.randn(batch_size, total_dim)
        
        # 分离输入并检查维度
        measurements = x[:, :22]
        mesh_features = x[:, 22:-2]
        basic_features = x[:, -2:]
        
        print("\n=== 维度检查 ===")
        print(f"输入维度: {x.shape}")
        print(f"测量值维度: {measurements.shape}")
        print(f"点云特征维度: {mesh_features.shape}")
        print(f"身高体重维度: {basic_features.shape}")
        
        # 测试各个网络分支
        mesh_out = model.mesh_net(mesh_features)
        basic_out = model.basic_net(basic_features)
        measurement_out = model.measurement_net(measurements)
        
        print("\n=== 特征提取输出维度 ===")
        print(f"点云特征输出: {mesh_out.shape}")
        print(f"身高体重输出: {basic_out.shape}")
        print(f"测量值输出: {measurement_out.shape}")
        
        # 测试特征组合
        combined = torch.cat([
            mesh_out,
            basic_out,
            measurement_out,
            measurements
        ], dim=1)
        print(f"\n组合特征维度: {combined.shape}")
        
        # 测试预测网络
        delta = model.predict_net(combined)
        print(f"预测偏差维度: {delta.shape}")
        
        # 测试最终输出
        output = model(x)
        print(f"\n最终输出维度: {output.shape}")
        print("维度检查通过！")
        return True
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        return False

def main():
    """主函数"""
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # 加载数据
        train_data = np.load('./data/train_data.npy')
        train_labels = np.load('./data/train_labels.npy')
        val_data = np.load('./data/val_data.npy')
        val_labels = np.load('./data/val_labels.npy')
        test_data = np.load('./data/test_data.npy')
        test_labels = np.load('./data/test_labels.npy')
        test_names = np.load('./data/test_names.npy')
        
        # 打印数据维度信息
        print("\n数据维度信息:")
        print(f"训练数据: {train_data.shape}")
        print(f"训练标签: {train_labels.shape}")
        print(f"验证数据: {val_data.shape}")
        print(f"验证标签: {val_labels.shape}")
        print(f"测试数据: {test_data.shape}")
        print(f"测试标签: {test_labels.shape}")
        print("-" * 50)
        
        # 标准化数据
        normalizer = DataNormalizer()
        normalizer.fit(train_data, train_labels)
        
        # 保存标准化参数
        os.makedirs('./data', exist_ok=True)
        normalizer.save('./data/normalization_params.npz')
        
        # 标准化训练集、验证集和测试集
        train_data_norm, train_labels_norm = normalizer.transform(train_data, train_labels)
        val_data_norm, val_labels_norm = normalizer.transform(val_data, val_labels)
        test_data_norm, test_labels_norm = normalizer.transform(test_data, test_labels)
        
        # 创建数据加载器
        train_dataset = BodyParamDataset(train_data_norm, train_labels_norm)
        val_dataset = BodyParamDataset(val_data_norm, val_labels_norm)
        test_dataset = BodyParamDataset(test_data_norm, test_labels_norm)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        test_loader = DataLoader(test_dataset, batch_size=8)
        
        # 创建模型
        model = BodyMeasurementRectifier()
        
        # 可视化模型结构
        visualize_model(model)
        
        # 测试网络维度
        test_network_dimensions()
        
        # 训练模型
        logging.info("开始训练...")
        model, best_val_loss = train_model(model, train_loader, val_loader, num_epochs=600, device=device)
        logging.info(f"训练完成！最佳验证损失: {best_val_loss:.6f}")
        
        # 在验证集上评估
        logging.info("\n进行验证集评估...")
        val_mae, val_rmse, val_rel_err, original_mae, original_rmse, original_rel_err = evaluate_model(model, val_loader, normalizer, device)
        
        # 在测试集上评估
        logging.info("\n进行测试集评估...")
        test_mae, test_rmse, test_rel_err, original_mae, original_rmse, original_rel_err = evaluate_model(model, test_loader, normalizer, device)
        
        # 计算并显示所有维度的平均MAE
        val_overall_mae = np.mean(val_mae)
        test_overall_mae = np.mean(test_mae)
        
        logging.info("\n" + "=" * 50)
        logging.info("所有维度平均MAE:")
        logging.info("-" * 30)
        logging.info(f"验证集: {val_overall_mae:.4f}")
        logging.info(f"测试集: {test_overall_mae:.4f}")
        logging.info("=" * 50)
        
        # 保存总体指标
        metrics_summary = {
            '数据集': ['验证集', '测试集'],
            '所有维度平均MAE': [val_overall_mae, test_overall_mae]
        }
        pd.DataFrame(metrics_summary).to_csv('./results/overall_metrics.csv', index=False, encoding='utf-8-sig')
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()