import subprocess
import time

def run_demo():
    while True:
        try:
            # 启动 demo 程序
            process = subprocess.Popen(['build/bin/demo'])
            # 等待程序退出
            process.wait()
        except KeyboardInterrupt:
            # 如果用户按 Ctrl+C,则退出循环
            break
        print("程序已退出,3秒后重新启动...")
        time.sleep(3)

if __name__ == "__main__":
    run_demo()
