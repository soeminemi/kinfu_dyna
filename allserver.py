import json
import asyncio
import websockets
import time
import random
import logging
import sys  # 新增导入

class Server:
    def __init__(self):
        self.services = {}  # 存储注册的服务
        self.last_ack = {}  # 存储最后一次 ACK 的时间
        self.service_status = {}  # 存储服务的状态 (忙碌或空闲)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler('server.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )

    async def register(self, websocket, path):
        logging.info("注册请求")
        try:
            message = await websocket.recv()
            logging.info(f"注册消息: {message}")
            data = json.loads(message)
            ip = data['ip']
            port = data['port']
            verification_code = data['verification_code']

            # 检查是否存在重复
            if (ip, port) not in self.services:
                self.services[(ip, port)] = verification_code
                self.last_ack[(ip, port)] = time.time()
                self.service_status[(ip, port)] = "空闲"  # 新注册的服务默认为空闲状态
                await websocket.send("OK")
                logging.info(f"注册成功: IP={ip}, 端口={port}")
            else:
                await websocket.send("IP 和端口已存在")
                logging.warning(f"注册失败: IP={ip}, 端口={port} 已存在")
        except Exception as e:
            await websocket.send(f"注册失败: {str(e)}")
            logging.error(f"注册异常: {str(e)}")

    async def query_services(self, websocket, path):
        logging.info("查询服务")
        available_services = [
            (ip, port)
            for (ip, port) in self.services.keys()
            if self.service_status.get((ip, port)) == "空闲"
        ]
        
        if available_services:
            # 随机选择一个可用的服务
            selected_ip, selected_port = random.choice(available_services)
            
            # 更新选中服务的验证码为当前时间戳
            current_timestamp = str(int(time.time()))
            selected_service = {
                "ip": selected_ip,
                "port": selected_port,
                "verification_code": current_timestamp
            }
            logging.info(f"查询结果: {json.dumps(selected_service)}")
            self.service_status[(selected_ip, selected_port)] = "忙碌"
            # 记录查询时间
            query_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logging.info(f"查询时间: {query_time}")
            logging.info(f"选中的服务: IP={selected_ip}, 端口={selected_port}")
            await websocket.send(json.dumps(selected_service))
        else:
            logging.info("没有可用服务")
            await websocket.send(json.dumps({}))  # 如果没有可用服务,返回空对象

    async def send_ack(self):
        while True:
            for (ip, port) in list(self.services.keys()):
                try:
                    uri = f"ws://{ip}:{port}"
                    logging.info(f"发送 ACK: {uri}")
                    async with websockets.connect(uri) as websocket:
                        ack = {"ack": str(time.time())}
                        await websocket.send(json.dumps(ack))
                        response = await websocket.recv()
                        logging.info(f"ACK 响应: {response}")
                        if response == "OK":
                            logging.info(f"{uri} 空闲")
                            self.last_ack[(ip, port)] = time.time()
                            self.service_status[(ip, port)] = "空闲"
                        else:
                            self.service_status[(ip, port)] = "忙碌"
                except:
                    logging.warning(f"ACK 失败: ws://{ip}:{port}, 时间: {time.time() - self.last_ack.get((ip, port), 0)}")
                    self.service_status[(ip, port)] = "忙碌"
                    # 如果连接失败或超过5分钟没有响应,则移除该服务
                    if time.time() - self.last_ack.get((ip, port), 0) > 20:
                        logging.info(f"删除服务器: {ip}:{port}")
                        del self.services[(ip, port)]
                        del self.last_ack[(ip, port)]
                        if (ip, port) in self.service_status:
                            del self.service_status[(ip, port)]
            await asyncio.sleep(5)  # 每5秒钟检查一次

    async def main(self):
        logging.info("服务器启动")
        register_server = await websockets.serve(self.register, "175.6.27.254", 8765)
        query_server = await websockets.serve(self.query_services, "175.6.27.254", 8766)
        ack_task = asyncio.create_task(self.send_ack())
        
        logging.info("服务器启动")
        await asyncio.gather(register_server.wait_closed(), query_server.wait_closed(), ack_task)

if __name__ == "__main__":
    server = Server()
    logging.info("程序开始运行")
    asyncio.run(server.main())
