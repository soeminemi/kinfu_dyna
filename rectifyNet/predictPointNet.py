from turtle import hideturtle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

class MeasurementDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        # Create mask for non-zero values
        self.mask = (self.features != 0).float()
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        mask = self.mask[idx]
        if self.transform:
            features = self.transform(features)
        return features, self.labels[idx], mask

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),  
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features)  
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.leaky_relu(out)

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            batch_size = query.size(1)
            key_padding_mask = torch.zeros((batch_size, 1), dtype=torch.bool)
        else:
            key_padding_mask = None
            
        attn_output, _ = self.attention(query, key, value, key_padding_mask=key_padding_mask)
        return self.norm(attn_output)

class OrderedPointFeatures(nn.Module):
    def __init__(self, num_points, point_dim=4, hidden_size=32):
        super(OrderedPointFeatures, self).__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        #############使用mlp处理点云################
        mlp_size = 16
        self.point_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(point_dim, mlp_size * 2),
                nn.LayerNorm(mlp_size * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(mlp_size * 2, mlp_size),
                nn.LayerNorm(mlp_size),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_points)
        ])
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(num_points * mlp_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.LeakyReLU(0.2)
        )
        ##################使用全连接网络处理点云特征################
        # # 使用全连接网络处理输入
        # self.fc = nn.Linear(num_points * point_dim, hidden_size)
        # # 特征融合
        # self.feature_fusion = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size * 2),
        #     nn.LayerNorm(hidden_size * 2),
        #     nn.LeakyReLU(0.2)
        # )
        ##########################################################
    def forward(self, x):
        batch_size = x.size(0)
        ################使用全连接网络处理点云特征#################
        # # 使用全连接网络处理输入
        # features = self.fc(x)  # (batch_size, hidden_size)
        ###################使用mlp处理点云#######################
        points = x.view(batch_size, self.num_points, self.point_dim)
        # 使用独立MLP处理每个点的特征
        point_features = []
        for i in range(self.num_points):
            # 提取单个点的特征，使用该位置专门的MLP
            point = points[:, i, :]
            feat = self.point_mlps[i](point)  # 使用位置i的专门MLP
            point_features.append(feat)
        
        # 拼接所有点的特征
        features = torch.cat(point_features, dim=1)  # (batch_size, num_points * hidden_size)
        ###################################################################################
        # 特征融合
        output = self.feature_fusion(features)  # (batch_size, hidden_size * 2)
        return output

class PredictionModel(nn.Module):
    def __init__(self, input_size=88, hidden_size=128, output_size=22, num_residual_blocks=3):
        super(PredictionModel, self).__init__()
        
        self.hidden_size = hidden_size
        num_points = (input_size-22) // 4  # 每个点4个维度
        
        # 有序点特征提取
        self.point_feature = OrderedPointFeatures(num_points=num_points, point_dim=4, hidden_size=hidden_size*2)
        # 底层交叉注意力
        self.low_level_attention = CrossAttentionLayer(hidden_size*4, num_heads=8)
        
        # 底层特征处理
        self.low_level_feature = nn.Sequential(
            nn.Linear(hidden_size*4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        # 特征处理
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 64 = hidden_size * 2 from OrderedPointFeatures
            nn.LayerNorm(hidden_size),  
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        # 融合测量值和特征
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size + 22, hidden_size),  # 64 = hidden_size * 2 from OrderedPointFeatures
            nn.LayerNorm(hidden_size),  
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        # 全局注意力
        self.global_attention = CrossAttentionLayer(hidden_size, num_heads=8)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_residual_blocks)
        ])
        
        # 输出层
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            # nn.ReLU(),  # 使用ReLU激活函数确保输出为正值
            nn.Tanh(),  # 限制输出在 -1 到 1 之间
            
        )
        self.scale_factor = nn.Parameter(torch.ones(output_size) * 0.01)  # 可学习的缩放因子
        self.bias_factor = nn.Parameter(torch.ones(output_size) * 0.1) # 可学习的偏移因子

    def forward(self, x):
        # 分离输入的前22维和后面的数据
        measurements = x[:, :22]
        target = x[:, :21]
        x = x[:, 22:]
        
        batch_size = x.size(0)
        
        # 特征提取
        point_features = self.point_feature(x)  # (batch_size, hidden_size * 2)
        
        # 底层特征提取
        low_level_features = self.low_level_attention(point_features, point_features, point_features)  # (batch_size, hidden_size)
        low_level_features = self.low_level_feature(low_level_features)  # (batch_size, hidden_size // 2)
        
        # 特征处理
        features = self.feature_layer(low_level_features)  # (batch_size, hidden_size)
        
        # 全局注意力
        features = features.unsqueeze(0)  # (1, batch_size, hidden_size)
        # 将measurements和features拼接得到新的特征
        combined_features = torch.cat((measurements.unsqueeze(0), features), dim=2)  # (1, batch_size, hidden_size + 21)
        
        # 更新features为拼接后的特征
        features = self.fusion_layer(combined_features)  # (batch_size, hidden_size)
        features = self.global_attention(features, features, features)
        features = features.squeeze(0)  # (batch_size, hidden_size)
        
        # 残差处理
        x = features
        for block in self.residual_blocks:
            x = block(x)
        
        # 输出
        x = self.final_layer(x) * self.scale_factor + target + self.bias_factor
        # 显示measurements
        # print("Measurements:", measurements)
        # print("Predictions:", x)
        
        return x

def load_and_prepare_data():
    # Load datasets
    train_data = np.load('./data/train_data.npy')
    train_labels = np.load('./data/train_labels.npy')
    val_data = np.load('./data/val_data.npy')
    val_labels = np.load('./data/val_labels.npy')
    test_data = np.load('./data/test_data.npy')
    test_labels = np.load('./data/test_labels.npy')
    
    # Print the dimensions of train_data
    print("Dimensions of train_data:", train_data.shape)
    
    # Extract point cloud features (x, y, z, dn)
    train_features = train_data[:, 22:-2] #.reshape(train_data.shape[0], -1, 4)
    val_features = val_data[:, 22:-2] #.reshape(val_data.shape[0], -1, 4)
    test_features = test_data[:, 22:-2] #.reshape(test_data.shape[0], -1, 4)
    
    train_measurements = train_data[:, :22]
    val_measurements = val_data[:, :22]
    test_measurements = test_data[:, :22]
    
    # Flatten for the model input
    train_features = train_features.reshape(train_data.shape[0], -1)
    val_features = val_features.reshape(val_data.shape[0], -1)
    test_features = test_features.reshape(test_data.shape[0], -1)
    
    # Standardize features using training data statistics
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Combine scaled features and measurements
    train_combined = np.hstack((train_measurements , train_features_scaled))
    val_combined = np.hstack((val_measurements, val_features_scaled))
    test_combined = np.hstack((test_measurements, test_features_scaled))
    
    return train_combined, val_combined, test_combined, train_labels, val_labels, test_labels, scaler
    
def train_model(model, train_loader, val_loader, device, num_epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    patience = 50
    no_improve_count = 0
    history = {'train_loss': [], 'val_loss': []}
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 打印当前epoch的结果
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        # 计算和打印验证集的具体指标
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        print_results(val_predictions, val_targets, "Validation")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            # Save best model
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': history
                }, 'best_model1.pth')
                print("Saved new best model")
            except Exception as e:
                print(f"Error saving model: {e}")
            finally:
                print("Model saved")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return history

def evaluate_model(model, data_loader, dataset_name="Test"):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    
    # Calculate errors
    absolute_errors = np.abs(predictions - labels)
    relative_errors = np.abs(predictions - labels) / np.abs(labels) * 100
    
    # Calculate mean errors per dimension
    mae_per_dim = np.mean(absolute_errors, axis=0)
    mre_per_dim = np.mean(relative_errors, axis=0)
    
    print(f"\n{dataset_name} Set Results:")
    print("Mean Absolute Error and Relative Error (per dimension):")
    
    
    # Print detailed results for each sample
    print("\nDetailed Results for Each Sample:")
    print("Sample\tDimension\tPredicted\tActual\t\tAbsolute Error\tRelative Error(%)")
    print("-" * 80)
    
    for sample_idx in range(len(predictions)):
        for dim_idx in range(predictions.shape[1]):
            pred_val = predictions[sample_idx, dim_idx]
            true_val = labels[sample_idx, dim_idx]
            abs_err = absolute_errors[sample_idx, dim_idx]
            rel_err = relative_errors[sample_idx, dim_idx]
            
            try:
                dim_name = measurement_names[dim_idx] if 'measurement_names' in locals() else f"Dim {dim_idx}"
            except:
                dim_name = f"Dim {dim_idx}"
                
            print(f"{sample_idx}\t{dim_name}\t{pred_val:.4f}\t{true_val:.4f}\t{abs_err:.4f}\t{rel_err:.2f}")
        
        # Add a separator line between samples
        if sample_idx < len(predictions) - 1:
            print("-" * 80)
    # Load measurement names (if available)
    try:
        measurement_names = np.load('./data/measurement_names.npy')
        for i, (mae, mre) in enumerate(zip(mae_per_dim, mre_per_dim)):
            name = measurement_names[i] if i < len(measurement_names) else f"Dimension {i}"
            print(f"{name}: MAE = {mae:.4f}, MRE = {mre:.2f}%")
    except:
        for i, (mae, mre) in enumerate(zip(mae_per_dim, mre_per_dim)):
            print(f"Dimension {i}: MAE = {mae:.4f}, MRE = {mre:.2f}%")
    
    print(f"\nOverall Mean Absolute Error: {np.mean(mae_per_dim):.4f}")
    print(f"Overall Mean Relative Error: {np.mean(mre_per_dim):.2f}%")
    
    # Save results to a file
    with open(f'{dataset_name.lower()}_results.txt', 'w') as f:
        f.write("Sample\tDimension\tPredicted\tActual\t\tAbsolute Error\tRelative Error(%)\n")
        f.write("-" * 80 + "\n")
        
        for sample_idx in range(len(predictions)):
            for dim_idx in range(predictions.shape[1]):
                pred_val = predictions[sample_idx, dim_idx]
                true_val = labels[sample_idx, dim_idx]
                abs_err = absolute_errors[sample_idx, dim_idx]
                rel_err = relative_errors[sample_idx, dim_idx]
                
                try:
                    dim_name = measurement_names[dim_idx] if 'measurement_names' in locals() else f"Dim {dim_idx}"
                except:
                    dim_name = f"Dim {dim_idx}"
                    
                f.write(f"{sample_idx}\t{dim_name}\t{pred_val:.4f}\t{true_val:.4f}\t{abs_err:.4f}\t{rel_err:.2f}\n")
            
            if sample_idx < len(predictions) - 1:
                f.write("-" * 80 + "\n")
    
    print(f"\nDetailed results have been saved to '{dataset_name.lower()}_results.txt'")
    
    return predictions, labels, mae_per_dim, mre_per_dim

def print_results(predictions, targets, phase=""):
    """
    计算并打印评估指标
    
    Args:
        predictions: 模型预测值 numpy array
        targets: 真实值 numpy array
        phase: 阶段名称 (e.g., "Training", "Validation", "Test")
    """
    # 计算各种指标
    mae = np.mean(np.abs(predictions - targets))
    me = np.mean(predictions - targets)
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算相对误差
    relative_errors = np.abs(predictions - targets) / (np.abs(targets) + 1e-8)  # 避免除零
    mre = np.mean(relative_errors) * 100  # 转换为百分比
    
    # 计算每个维度的误差
    dim_mae = np.mean(np.abs(predictions - targets), axis=0)
    dim_me = np.mean(predictions - targets, axis=0)
    dim_mse = np.mean((predictions - targets) ** 2, axis=0)
    dim_rmse = np.sqrt(dim_mse)
    
    print(f"\n{phase} Results:")
    print(f"{'='*50}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Error (ME): {me:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Relative Error (MRE): {mre:.2f}%")
    print(f"\nPer-Dimension MAE:")
    for i, error in enumerate(dim_mae):
        print(f"Dimension {i+1}: {error:.4f}, {dim_me[i]:.4f}, {dim_rmse[i]:.4f}")
    print(f"{'='*50}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mre': mre,
        'dim_mae': dim_mae
    }

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('loss_plot.png')
    plt.close()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def test_model(model, test_loader, device, checkpoint_path):
    # Load model parameters from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_predictions = []
    all_targets = []
    all_deviations = []
    
    # Define dimension names
    dimension_names = np.load('data/measurement_names.npy').tolist()
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Perform predictions
            predictions = model(batch_X)
            
            # Store predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate deviations
            deviations = predictions - batch_y
            all_deviations.extend(deviations.cpu().numpy())
    
    # Convert lists to numpy arrays for easier handling
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_deviations = np.array(all_deviations)
    # Load sample names
    sample_names = np.load('./data/test_names.npy')
    
    # Output results
    for i in range(len(all_predictions)):
        print(f"Sample {sample_names[i]}: ")
        print(f"{'Dimension':<15}{'Prediction':<20}{'True Value':<20}{'Deviation':<20}")
        print(f"{'='*70}")
        for dim in range(len(all_predictions[i])):
            print(f"{dimension_names[dim]:<15}{all_predictions[i][dim]:<20}{all_targets[i][dim]:<20}{all_deviations[i][dim]:<20}")
        print("\n")  # New line for separation between samples

def main():
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare data
    train_features_scaled, val_features_scaled, test_features_scaled, train_labels, val_labels, test_labels, scaler = load_and_prepare_data()
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(train_features_scaled)
    y_train = torch.FloatTensor(train_labels)
    X_val = torch.FloatTensor(val_features_scaled)
    y_val = torch.FloatTensor(val_labels)
    X_test = torch.FloatTensor(test_features_scaled)
    y_test = torch.FloatTensor(test_labels)
    
    # Create data loaders with smaller batch size
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = PredictionModel(
        input_size=X_train.shape[1], 
        hidden_size=128,
        output_size=y_train.shape[1], 
        num_residual_blocks=4
    )
    model.to(device)
    
    # 尝试加载已有的最佳模型
    start_epoch = 0
    best_val_loss = float('inf')
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    checkpoint_path = 'model_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming from epoch {start_epoch} with validation loss: {best_val_loss:.4f}")
    else:
        print("No checkpoint found, starting training from scratch")
        model.apply(initialize_weights)
    
    # Choose between training and testing
    mode = input("Enter 'train' to train the model or 'test' to test the model: ").strip().lower()
    
    if mode == 'train':
        # Training loop
        history = train_model(model, train_loader, val_loader, device, num_epochs=600)
        plot_losses(history['train_loss'], history['val_loss'])
        # Load best model for testing
        print("\nLoading best model for testing...")
        checkpoint = torch.load('best_model1.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test the model
        model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_X)
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(batch_y.cpu().numpy())
        
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        
        # Calculate and print test results
        print_results(test_predictions, test_targets, "Test")
        
        # Save the final model and scaler
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_state': scaler,
            'input_size': X_train.shape[1],
            'hidden_size': 128,
            'output_size': y_train.shape[1],
            'num_residual_blocks': 3
        }, 'hw_prediction_model.pth')
        print("\nModel and standardizer saved to 'hw_prediction_model.pth'")
    elif mode == 'test':
        # Testing process
        test_model(model, test_loader, device, 'hw_prediction_model.pth')
    else:
        print("Invalid input. Please enter 'train' or 'test'.")

if __name__ == "__main__":
    main()