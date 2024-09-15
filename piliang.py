import subprocess
import time
import os

def run_testServer(depth_folder, num_runs, testServer_path):
    for i in range(num_runs):
        print(f"运行 testServer.py 第 {i+1} 次")
        # 使用当前路径执行 testServer.py
        subprocess.run(["python", testServer_path, depth_folder])
        if i < num_runs - 1:
            print("等待5秒后开始下一次运行...")
            time.sleep(5)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.join(current_dir, "test")
    testServer_path = os.path.join(current_dir, "testServer.py")  # testServer.py 的路径
    num_runs = 3  # 设置运行次数
    
    if not os.path.exists(test_folder):
        print(f"错误: 找不到test文件夹 {test_folder}")
        exit(1)
    
    if not os.path.exists(testServer_path):
        print(f"错误: 找不到testServer.py文件 {testServer_path}")
        exit(1)
    
    for subfolder in os.listdir(test_folder):
        subfolder_path = os.path.join(test_folder, subfolder)
        if os.path.isdir(subfolder_path):
            depth_folder = os.path.join(subfolder_path, "depths")
            if os.path.exists(depth_folder):
                print(f"处理文件夹: {subfolder}")
                print(f"depths文件夹路径: {depth_folder}")
                run_testServer(depth_folder, num_runs, testServer_path)
            else:
                print(f"警告: {subfolder} 中没有找到depths文件夹")

print("所有运行完成")
