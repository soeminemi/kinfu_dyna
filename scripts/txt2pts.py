import numpy as np
import cv2
import open3d as o3d

def txt_to_pointcloud(txt_path):
    """
    从txt文件读取点云数据并转换为ply格式
    Args:
txt_path: txt文件路径,每行包含 x y z r g b 6个值
    Returns:
点云对象
    """
    # 读取txt文件
    data = np.loadtxt(txt_path)
    
    # 分离点坐标和颜色信息
    points = data[:, 0:3]  # 前3列是xyz坐标
    colors = data[:, 3:6] / 255.0  # 后3列是rgb颜色,归一化到[0,1]
    
    # 创建open3d点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
print("请提供txt文件路径作为命令行参数")
print("用法: python txt2pts.py <txt文件路径>")
sys.exit(1)

    txt_path = sys.argv[1]
    
    # 生成点云
    pcd = txt_to_pointcloud(txt_path)
    
    # 保存为ply文件
    output_path = txt_path.rsplit(".", 1)[0] + ".ply"
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存为{output_path}")
