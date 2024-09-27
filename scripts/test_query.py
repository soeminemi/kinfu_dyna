import websocket
import json

def query_available_service():
    ws_url = "ws://175.6.27.254:8766"  # 查询服务器的WebSocket地址
    
    try:
        # 创建WebSocket连接
        ws = websocket.create_connection(ws_url)
        
        # 发送查询请求
        ws.send(json.dumps({"cmd": "query"}))
        
        # 接收服务器响应
        response = ws.recv()
        service_info = json.loads(response)
        
        # 关闭WebSocket连接
        ws.close()
        
        if service_info:
            print("可用服务信息:")
            print(f"IP: {service_info['ip']}")
            print(f"端口: {service_info['port']}")
            print(f"验证码: {service_info['verification_code']}")
            return service_info
        else:
            print("当前没有可用的服务")
            return None
    
    except Exception as e:
        print(f"查询服务时发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    query_available_service()
