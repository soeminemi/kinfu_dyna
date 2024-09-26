import asyncio
import websockets
import time
import os
import random
import socket
import json
import subprocess
from datetime import datetime

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 这里不需要真正连接
        s.connect(('8.8.8.8', 80))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
        print("exception in get local ip")
    finally:
        s.close()
    return IP

async def register_to_websocket(ip, port, verification_code):
    # 替换为实际的 WebSocket 服务器地址
    uri = "ws://175.6.27.254:8765"
    print("try connect to ", uri)
    async with websockets.connect(uri) as websocket:
        print("connected")
        register_data = {
            "ip": ip,
            "port": port,
            "verification_code": verification_code
        }
        message = json.dumps(register_data)
        
        await websocket.send(message)
        result = await websocket.recv()
        return result == "OK"

async def run_demo():
    os.makedirs('logs', exist_ok=True)
    fail_log_file = 'logs/faillog.log'
    local_ip = get_local_ip()
    print(local_ip)
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_log_file = f'logs/log_{current_time}.log'
            
            # 生成随机端口
            port = random.randint(9001, 65535)
            
            # 向 WebSocket 服务器注册
            verification_code = str(time.time());  # 请替换为实际的验证码
            print("尝试注册")
            while not await register_to_websocket(local_ip, port, verification_code):
                print("注册失败，重新注册...")
                port = random.randint(9001, 65535)
            
            print("注册成功，启动程序...")
            # 启动 demo 程序,传入端口参数
            with open(output_log_file, 'w') as f:
                process = subprocess.Popen(['build/bin/demo', str(port)], stdout=f, stderr=subprocess.STDOUT)
                process.wait()
            
            exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(fail_log_file, 'a') as f:
                f.write(f"程序在 {exit_time} 退出，使用端口：{port}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            with open(fail_log_file, 'a') as f:
                f.write(f"发生错误：{str(e)}\n")
        
        print("程序已退出,3秒后重新启动...")
        time.sleep(3)

if __name__ == "__main__":
    asyncio.run(run_demo())
