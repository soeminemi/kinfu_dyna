import numpy as np
import argparse
import os

def load_ply(path):
    """手动加载PLY文件并返回顶点"""
    vertices = []
    with open(path, 'r') as f:
        # 跳过头部信息
        for line in f:
            if line.strip() == 'end_header':
                break
        # 读取顶点数据
        for line in f:
            if len(line.split()) == 3:  # 假设顶点数据为x y z格式
                vertices.append([float(x) for x in line.split()])
    return np.array(vertices)

def find_corresponding_vertices(full_vertices, part_vertices, threshold=1e-5):
    """Find indices of part vertices in full model"""
    indices = []
    for part_vertex in part_vertices:
        # Calculate distances to all vertices in full model
        distances = np.linalg.norm(full_vertices - part_vertex, axis=1)
        # Find closest vertex
        min_idx = np.argmin(distances)
        if distances[min_idx] < threshold:
            indices.append(min_idx)
    return np.array(indices)

def save_indices_txt(filename, indices):
    """将索引保存到文本文件中，每行一个索引"""
    with open(filename, 'w') as f:
        for index in indices:
            f.write(f"{index}\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('part_model', help='部分身体模型PLY文件的路径')
    args = parser.parse_args()

    # Load full body model
    full_body_path = './data/std_body.ply'
    full_vertices = load_ply(full_body_path)
    
    # Load partial body model
    part_vertices = load_ply(args.part_model)
    
    # Find corresponding vertex indices
    indices = find_corresponding_vertices(full_vertices, part_vertices)
    
    # Generate output file path
    output_path = os.path.splitext(args.part_model)[0] + '.mat'
    save_indices_txt(output_path, indices)
    
    print(f"Found {len(indices)} corresponding vertices")
    print(f"Saved indices to {output_path}")

if __name__ == '__main__':
    main()
