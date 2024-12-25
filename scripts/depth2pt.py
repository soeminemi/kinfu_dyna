import numpy as np
import cv2
import open3d as o3d

def depth_to_pointcloud(depth_path, fx, fy, cx, cy):
    """
    将深度图转换为点云
    Args:
depth_path: 深度图路径
fx, fy: 相机焦距
cx, cy: 相机主点坐标
    Returns:
点云对象
    """
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    height, width = depth.shape
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 计算归一化平面坐标
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    
    # 计算射线方向向量
    norm = np.sqrt(x_norm**2 + y_norm**2 + 1)
    
    # 将深度值转换为米
    depth = depth
    
    # 计算实际的3D坐标
    z = depth 
    x = x_norm * z
    y = y_norm * z
    
    # 构建旋转矩阵和平移向量
    R = np.array([
[0.999995, -0.00173292, -0.00259683],
[0.00171829, 0.999983, -0.00562471], 
[0.00260653, 0.00562022, 0.999981]
    ])
    T = np.array([-12.1293, -0.159468, 0.101271])
    
    # 对每个点进行旋转和平移变换
    z = np.dot(R, np.stack([z.flatten(), z.flatten(), z.flatten()], axis=0)) + T.reshape(3,1)
    x = np.dot(R, np.stack([x.flatten(), x.flatten(), x.flatten()], axis=0)) + T.reshape(3,1) 
    y = np.dot(R, np.stack([y.flatten(), y.flatten(), y.flatten()], axis=0)) + T.reshape(3,1)
    
    # 重新整形为原始形状
    z = z[0].reshape(depth.shape)
    x = x[0].reshape(depth.shape)
    y = y[0].reshape(depth.shape)
    # 创建点云
    points = np.stack([x, y, z], axis=-1)
    points = points.reshape(-1, 3)
    valid_points = points[depth.flatten() > 0]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    return pcd

if __name__ == "__main__":
    # 示例用法
    depth_path = "./test/ASC60CE45000218/ASC60CE45000218/Frames/depth/depth_20241120_115425263_0.png"  # 深度图路径
    import sys
    if len(sys.argv) < 2:
print("请提供深度图路径作为命令行参数")
print("用法: python depth2pt.py <深度图路径>")
sys.exit(1)
    depth_path = sys.argv[1]
    # # 相机内参
    # fx = 435.32*1.35
    # fy = 434.86*1.35
    # cx = 314.072
    # cy = 239.634
    # # 更新相机内参
    # fx = 367.04278564453125
    # fy = 367.04278564453125
    # cx = 255.80419921875
    # cy = 203.50630187988281

    # 相机内参
    fx = 252.36473083496094
    fy = 252.37528991699219
    cx = 262.01773071289062
    cy = 253.10002136230469

    
    # 生成点云
    pcd = depth_to_pointcloud(depth_path, fx, fy, cx, cy)
    
    # 保存点云
    o3d.io.write_point_cloud("output.ply", pcd)
    print("点云已保存为output.ply")
