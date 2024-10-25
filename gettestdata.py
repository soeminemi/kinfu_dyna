import paramiko
import os
from scp import SCPClient
from datetime import datetime

def ssh_scp_get(hostname, username, password, remote_path, local_path):
    # 创建SSH客户端
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # 连接到远程主机
        ssh.connect(hostname, username=username, password=password)
        
        # 获取远程文件夹列表及其修改时间
        stdin, stdout, stderr = ssh.exec_command(f'find {remote_path} -maxdepth 1 -type d -printf "%T@ %p\n"')
        folders_with_time = stdout.read().decode().splitlines()
        
        if not folders_with_time:
            print("远程路径下没有文件夹")
            return
        
        # 解析并排序文件夹列表
        folders = []
        for item in folders_with_time:
            timestamp, path = item.split(' ', 1)
            if path != remote_path:  # 排除父目录
                folders.append((float(timestamp), path))
        
        folders.sort()  # 按时间戳排序
        
        # 打印排序后的文件夹列表
        print("可用的文件夹 (按修改时间从早到晚排序):")
        for i, (timestamp, folder) in enumerate(folders):
            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i+1}. {os.path.basename(folder)} (修改时间: {time_str})")
        
        choices = (input("请选择要下载的文件夹编号: "))
        chs = choices.split('-')
        start = 0
        end = 0
        if len(chs) == 1:
            start = int(chs[0])-1
            end = start + 1
        else:
            start = int(chs[0])-1
            end = int(chs[1])
        for choice in range(start, end):
            selected_folder = folders[choice][1]
            # 直接使用选定文件夹的名称作为本地文件夹名
            # local_folder = os.path.join(local_path, os.path.basename(selected_folder))
            local_folder = local_path
            os.makedirs(local_folder, exist_ok=True)
            print(f"将下载{selected_folder}到{local_folder},请稍等...")
            # 使用SCP下载整个文件夹内容，而不是文件夹本身
            with SCPClient(ssh.get_transport()) as scp:
                scp.get(f"{selected_folder}/", local_folder, recursive=True)
            
            print(f"文件夹 '{os.path.basename(selected_folder)}' 的内容已成功下载到 '{local_folder}'")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    finally:
        ssh.close()

# 使用示例
if __name__ == "__main__":
    hostname = "175.6.27.254"
    username = "wjn"
    password = "asdfrewq"
    remote_path = "/home/wjn/.projects/kinfu_dyna/check_results"
    local_path = "./test"
    
    # 确保test文件夹存在
    os.makedirs(local_path, exist_ok=True)
    
    ssh_scp_get(hostname, username, password, remote_path, local_path)
