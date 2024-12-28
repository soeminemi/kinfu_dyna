import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BodyParamDataset(Dataset):
    def __init__(self, data_path, label_path):
        # 使用numpy加载数据，然后转换为tensor
        self.data = torch.from_numpy(np.load(data_path)).float()
        self.labels = torch.from_numpy(np.load(label_path)).float()
        
        # 分离不同类型的输入数据
        self.initial_measurements = self.data[:, :22]  # 初始测量值
        self.point_cloud = self.data[:, 22:-2]  # 点云偏差
        self.height_weight = self.data[:, -2:]  # 身高体重
        
        # 计算测量值的误差（目标修正量）
        self.measurement_errors = self.labels - self.initial_measurements
        
        # 保存原始数据的副本用于验证
        self._original_data = self.data.clone()
        self._original_errors = self.measurement_errors.clone()
    
    def normalize_data(self, m_mean, m_std, p_mean, p_std, h_mean, h_std, e_mean, e_std):
        # 分别对三种输入数据进行标准化
        self.initial_measurements = (self.initial_measurements - m_mean) / (m_std + 1e-6)
        self.point_cloud = (self.point_cloud - p_mean) / (p_std + 1e-6)
        self.height_weight = (self.height_weight - h_mean) / (h_std + 1e-6)
        
        # 标准化误差
        self.measurement_errors = (self.measurement_errors - e_mean) / (e_std + 1e-6)
        
        # 重新组合数据
        self.data = torch.cat([self.initial_measurements, self.point_cloud, self.height_weight], dim=1)
    
    def get_original_data(self):
        return self._original_data
    
    def get_original_errors(self):
        return self._original_errors
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.measurement_errors[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features)
        )
        self.relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class FeatureExtractor(nn.Module):
    def __init__(self, in_features, dims, use_residual=True):
        super(FeatureExtractor, self).__init__()
        layers = []
        current_dim = in_features
        
        for dim in dims:
            # 降维层
            layers.extend([
                nn.Linear(current_dim, dim),
                nn.LayerNorm(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            
            # 添加残差块进行特征提取
            if use_residual:
                layers.append(ResidualBlock(dim))
            
            current_dim = dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class AttentionLayer(nn.Module):
    def __init__(self, dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class BodyEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(BodyEstimator, self).__init__()
        
        # 测量值处理分支 - 增强特征提取
        self.measurement_extractor = FeatureExtractor(
            22, 
            dims=[64, 128, 256, 512, 256],
            use_residual=True
        )
        
        # 点云处理分支 - 更深的特征提取
        point_cloud_dim = input_size - 24
        self.point_extractor = FeatureExtractor(
            point_cloud_dim,
            dims=[1024, 512, 256, 128, 256],
            use_residual=True
        )
        
        # 身高体重处理分支
        self.hw_extractor = FeatureExtractor(
            2,
            dims=[32, 64, 128, 64],
            use_residual=False
        )
        
        # 特征融合网络
        fusion_dim = 256 + 256 + 64  # 各分支输出维度之和
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            ResidualBlock(512),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            ResidualBlock(256),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128)
        )
        
        # 注意力层
        self.attention = AttentionLayer(128)
        
        # 预测误差修正量
        self.error_predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
            
            nn.Linear(256, output_size),
            nn.Tanh()  # 限制修正量的范围
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用He初始化，但稍微增大初始值
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        # 分离输入数据
        measurements = x[:, :22]
        point_cloud = x[:, 22:-2]
        height_weight = x[:, -2:]
        
        # 特征提取
        m_features = self.measurement_extractor(measurements)
        p_features = self.point_extractor(point_cloud)
        h_features = self.hw_extractor(height_weight)
        
        # 特征融合
        combined = torch.cat([m_features, p_features, h_features], dim=1)
        fusion_features = self.fusion_net(combined)
        
        # 应用注意力机制
        attended_features = self.attention(fusion_features)
        
        # 预测修正量
        corrections = self.error_predictor(attended_features)
        
        return corrections

class WeightedLoss(nn.Module):
    def __init__(self, measurement_weights=None):
        super(WeightedLoss, self).__init__()
        self.measurement_weights = measurement_weights if measurement_weights is not None else torch.ones(22)
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        # 计算基本误差
        l1_loss = torch.abs(pred - target)
        l2_loss = self.mse(pred, target)
        
        # 计算相对误差
        rel_error = l1_loss / (torch.abs(target) + 1e-6)
        
        # 计算Huber损失（结合L1和L2的优点）
        delta = 1.0
        huber_loss = torch.where(
            l1_loss < delta,
            0.5 * l2_loss,
            delta * l1_loss - 0.5 * delta**2
        )
        
        # 组合损失
        weighted_loss = (
            0.4 * huber_loss + 
            0.4 * l2_loss + 
            0.2 * rel_error
        ) * self.measurement_weights.to(pred.device)
        
        return weighted_loss.mean()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, error_mean, error_std):
    best_val_loss = float('inf')
    patience = 20
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target_errors) in enumerate(train_loader):
            data, target_errors = data.to(device), target_errors.to(device)
            
            optimizer.zero_grad()
            predicted_errors = model(data)
            
            # 计算损失
            loss = criterion(predicted_errors, target_errors)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 打印每个batch的损失
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        abs_errors = []
        rel_errors = []
        
        with torch.no_grad():
            for data, target_errors in val_loader:
                data, target_errors = data.to(device), target_errors.to(device)
                predicted_errors = model(data)
                
                # 计算验证损失
                val_loss += criterion(predicted_errors, target_errors).item()
                
                # 反标准化预测的误差
                pred_errors_denorm = predicted_errors * (error_std.to(device) + 1e-6) + error_mean.to(device)
                target_errors_denorm = target_errors * (error_std.to(device) + 1e-6) + error_mean.to(device)
                
                # 计算绝对误差和相对误差
                abs_err = torch.abs(pred_errors_denorm - target_errors_denorm)
                rel_err = abs_err / (torch.abs(target_errors_denorm) + 1e-6)
                
                abs_errors.append(abs_err.cpu())
                rel_errors.append(rel_err.cpu())
        
        val_loss /= len(val_loader)
        
        # 计算平均误差
        abs_errors = torch.cat(abs_errors, dim=0)
        rel_errors = torch.cat(rel_errors, dim=0)
        
        mean_abs_error = abs_errors.mean(dim=0)
        mean_rel_error = rel_errors.mean(dim=0)
        
        print(f'Epoch {epoch}:')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')
        print(f'Mean Absolute Error: {mean_abs_error.mean():.4f}')
        print(f'Mean Relative Error: {mean_rel_error.mean():.4f}')
        
        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    return model.state_dict()

def evaluate_model(model, test_loader, device, label_mean, label_std):
    model.eval()
    predictions = []
    actuals = []
    
    # 确保标准化参数在正确的设备上
    label_mean = label_mean.to(device)
    label_std = label_std.to(device)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # 将标签也移到正确的设备上
            outputs = model(inputs)
            
            # 转换回原始范围
            outputs = outputs * label_std + label_mean
            labels = labels * label_std + label_mean
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 计算各种评估指标
    mae = np.mean(np.abs(predictions - actuals), axis=0)
    mse = np.mean((predictions - actuals)**2, axis=0)
    rmse = np.sqrt(mse)
    
    # 计算相对误差
    relative_error = np.mean(np.abs(predictions - actuals) / np.abs(actuals), axis=0)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'relative_error': relative_error
    }

def predict(model, input_data, device, input_mean, input_std, label_mean, label_std):
    """
    使用模型进行预测，包含所有必要的预处理和后处理步骤
    """
    model.eval()
    with torch.no_grad():
        # 确保数据是tensor
        if isinstance(input_data, np.ndarray):
            input_data = torch.FloatTensor(input_data)
        
        # 添加batch维度如果需要
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)
        
        # 预处理：标准化输入数据
        input_normalized = (input_data - input_mean) / (input_std + 1e-6)
        input_normalized = input_normalized.to(device)
        
        # 模型预测
        output_normalized = model(input_normalized)
        
        # 后处理：将输出转换回原始范围
        output = output_normalized * label_std.to(device) + label_mean.to(device)
    
    return output.cpu().numpy()

def validate_model(model_path, data_path, norm_params_path, device):
    """
    验证保存的模型
    """
    # 加载模型
    model = torch.load(model_path)
    model.to(device)
    
    # 加载标准化参数
    norm_params = torch.load(norm_params_path)
    input_mean = norm_params['mean']
    input_std = norm_params['std']
    label_mean = norm_params['label_mean']
    label_std = norm_params['label_std']
    
    # 加载验证数据
    data = np.load(data_path)
    
    # 进行预测
    predictions = predict(model, data, device, input_mean, input_std, label_mean, label_std)
    
    return predictions

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 加载数据
    data_dir = './data'
    train_dataset = BodyParamDataset(
        os.path.join(data_dir, 'train_data.npy'),
        os.path.join(data_dir, 'train_labels.npy')
    )
    val_dataset = BodyParamDataset(
        os.path.join(data_dir, 'val_data.npy'),
        os.path.join(data_dir, 'val_labels.npy')
    )
    
    # 使用原始数据计算标准化参数
    train_data = train_dataset.get_original_data()
    train_errors = train_dataset.get_original_errors()
    
    # 分别计算每种数据的统计信息
    m_mean = train_data[:, :22].mean(dim=0)
    m_std = train_data[:, :22].std(dim=0)
    
    p_mean = train_data[:, 22:-2].mean(dim=0)
    p_std = train_data[:, 22:-2].std(dim=0)
    
    h_mean = train_data[:, -2:].mean(dim=0)
    h_std = train_data[:, -2:].std(dim=0)
    
    error_mean = train_errors.mean(dim=0)
    error_std = train_errors.std(dim=0)
    
    # 标准化数据
    train_dataset.normalize_data(m_mean, m_std, p_mean, p_std, h_mean, h_std, error_mean, error_std)
    val_dataset.normalize_data(m_mean, m_std, p_mean, p_std, h_mean, h_std, error_mean, error_std)
    
    # 保存标准化参数
    torch.save({
        'measurement_mean': m_mean,
        'measurement_std': m_std,
        'point_cloud_mean': p_mean,
        'point_cloud_std': p_std,
        'height_weight_mean': h_mean,
        'height_weight_std': h_std,
        'error_mean': error_mean,
        'error_std': error_std
    }, os.path.join(data_dir, 'normalization_params.pt'))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型参数
    input_size = train_dataset.data.shape[1]
    output_size = train_dataset.measurement_errors.shape[1]
    
    # 创建模型
    model = BodyEstimator(input_size, output_size).to(device)
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005,  # 降低初始学习率
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # 使用带有预热的学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,  # 降低最大学习率
        epochs=1000,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 增加预热期
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0
    )
    
    # 训练模型
    best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=1000,
        device=device,
        error_mean=error_mean,
        error_std=error_std
    )
    
    # 保存最佳模型
    torch.save(best_model_state, os.path.join(data_dir, 'best_model.pt'))
    
if __name__ == '__main__':
    main()
