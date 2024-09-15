import subprocess
import time
import os
from datetime import datetime

def run_demo():
    # 确保 logs 文件夹存在
    os.makedirs('logs', exist_ok=True)
    log_file = 'logs/faillog.log'

    while True:
        try:
            # 启动 demo 程序
            process = subprocess.Popen(['build/bin/demo'])
            # 等待程序退出
            process.wait()
            
            # 记录退出时间
            exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a') as f:
                f.write(f"程序在 {exit_time} 退出\n")
            
        except KeyboardInterrupt:
            # 如果用户按 Ctrl+C,则退出循环
            break
        print("程序已退出,3秒后重新启动...")
        time.sleep(3)

if __name__ == "__main__":
    run_demo()
