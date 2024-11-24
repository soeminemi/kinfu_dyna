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
        
        # 获取远程文件列表及其修改时间
        stdin, stdout, stderr = ssh.exec_command(f'find {remote_path} -type f -printf "%T@ %p\n"')
        files_with_time = stdout.read().decode().splitlines()
        
        if not files_with_time:
            print("远程路径下没有文件")
            return
        
        # 解析并排序文件列表
        files = []
        for item in files_with_time:
            timestamp, path = item.split(' ', 1)
            files.append((float(timestamp), path))
        
        files.sort()  # 按时间戳排序
        
        # 打印排序后的文件列表
        print("可用的文件 (按修改时间从早到晚排序):")
        for i, (timestamp, file_path) in enumerate(files):
            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i+1}. {os.path.basename(file_path)} (修改时间: {time_str})")
        
        choices = input("请选择要下载的文件编号 (可用'-'指定范围，如'1-3'): ")
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
            selected_file = files[choice][1]
            local_file = os.path.join(local_path, os.path.basename(selected_file))
            print(f"将下载{selected_file}到{local_file}，请稍等...")
            
            # 使用SCP下载选中的文件
            with SCPClient(ssh.get_transport()) as scp:
                scp.get(selected_file, local_file)
            
            print(f"文件 '{os.path.basename(selected_file)}' 已成功下载到 '{local_file}'")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    finally:
        ssh.close()

# 使用示例
if __name__ == "__main__":
    hostname = "175.6.27.254"
    username = "wjn"
    password = "asdfrewq"
    remote_path = "/home/wjn/.projects/kinfu_dyna/logs"
    local_path = "./logs"
    
    # 确保logs文件夹存在
    os.makedirs(local_path, exist_ok=True)
    
    ssh_scp_get(hostname, username, password, remote_path, local_path)
