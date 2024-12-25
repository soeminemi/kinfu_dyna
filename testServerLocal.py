import cv2
import numpy as np
import socket
import websocket
import base64
import json
import cvui
import threading
from imageio import imread
import time
import signal

# 获取深度图, 默认尺寸 424x512
# def get_last_depth():
#     frame = kinect.get_last_depth_frame()
#     frame = frame.astype(np.uint16)
#     # print("frame shape: ", frame.shape)
#     dep_frame = np.reshape(frame, [424, 512])
#     return dep_frame

# 获取rgb图, 1080x1920x4
# def get_last_rbg():
#     frame = kinect.get_last_color_frame()
#     return np.reshape(frame, [1080, 1920, 4])[:, :, 0:3]

# socket client
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_address = ('localhost', 9099)
# client_socket.connect(server_address)

# depths = []
images = []
global lock
lock = threading.Lock()
import time

def pub_msg(ws):
    global msgs
    global flag_exit
    flag_exit = False
    print("start the send thread")
    print("当前thread: [{}]".format(threading.current_thread().name))
    while True:
        lock.acquire()
        if flag_exit:
            lock.release()
            break
        if len(msgs) > 0:
            msg = msgs.pop(0)
            print("send msg to server ", len(msgs))
            lock.release()
            ws.send(msg)
        else:
            lock.release()
        time.sleep(0.01)

def signal_handler(signum, frame):
    global flag_exit
    print("\nCtrl+C detected. Cleaning up...")
    flag_exit = True
    # Give threads time to clean up
    time.sleep(0.5)
    sys.exit(0)

import os
import re
import sys

if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"当前程序执行路径: {os.getcwd()}")
    
    if len(sys.argv) < 2:
        print("使用方法: python testServer.py <深度图像文件夹路径>")
        sys.exit(1)
    
    depth_folder = sys.argv[1]  # 从命令行参数获取深度图像文件夹路径
    
    if not os.path.isdir(depth_folder):
        print(f"错误: 文件夹 '{depth_folder}' 不存在")
        sys.exit(1)
    
    print(f"使用深度图像文件夹: {depth_folder}")
    
    sample_num = 1500
    flag_cache_send = True
    flag_save_disk = False
    flag_start = True
    flag_end = False
    
    # for fid in range(sample_num):
    # 查询可用服务
    def query_available_service():
        query_ws_url = "ws://175.6.27.254:8766"  # 查询服务器的WebSocket地址
        try:
            # 创建WebSocket连接
            query_ws = websocket.create_connection(query_ws_url)
            
            # 发送查询请求
            query_ws.send(json.dumps({"cmd": "query"}))
            
            # 接收服务器响应
            response = query_ws.recv()
            service_info = json.loads(response)
            
            # 关闭WebSocket连接
            query_ws.close()
            
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
    
    # # 获取可用服务信息
    # service_info = query_available_service()
    
    # if not service_info:
    #     print("无法获取可用服务，程序退出")
    #     sys.exit(1)

    # 连接到给定的WebSocket服务
    ws_url = "ws://127.0.0.1:9099"
    vcode = str(int(time.time()))
    fid = 0
    # ws_url = "ws://175.6.27.254:9099"
    ws = websocket.create_connection(ws_url)
    
    show_msg = "Ready"
    orig = (50,50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    color = (255,0,0)
    flag_show_intrinsic = False
    WINDOW_NAME = "Kinect"
    frame = np.zeros((1920, 1080, 3), np.uint8)
    # cvui.init(WINDOW_NAME)
    male_checked = [True]
    female_checked = [False]
    gender = "female"
    button_name = "START"
    result_str = "None"
    
    # 从命令行参数获取重量
    import sys
    weight = 50.0  # 默认值
    flag_weight_set = False
    if len(sys.argv) >= 3:
        try:
            weight = float(sys.argv[2])
            flag_weight_set = True
        except ValueError:
            print("无效的重量参数,使用默认值50.0")
    
    if flag_weight_set:
        print("当前设置的重量为:", weight)
    else:
        print("未设置体重参数")
    
    global msgs
    msgs = []
    
    if flag_cache_send:
        print("try to start the msg sending thread")
        thread_sdmsg = threading.Thread(target=pub_msg, args=(ws,),name="send1")
        thread_sdmsg.start()
        # thread_sdmsg.join()
    
    # depth_folder = "./body_1726305326/depths"  # 指定深度图像文件夹
    # depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])
    
    def sort_key(filename):
        # 从文件名中提取数字
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')], key=sort_key)
    
    depth_index = 0
    intr = {'FocalLengthX': 367.04278564453125, 'FocalLengthY': 367.04278564453125, 'PrincipalPointX': 255.80419921875, 'PrincipalPointY': 203.5063018798828, 'RadialDistortionSecondOrder': 0.09293150156736374, 'RadialDistortionFourthOrder': -0.2737075090408325, 'RadialDistortionSixthOrder': 0.09219703823328018}
    intrinsics_path = os.path.join(os.path.dirname(depth_folder+'../'), "intrinsics.json")
    if os.path.isfile(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            intrinsics = json.load(f)
            intr = intrinsics
            intr["RadialDistortionSecondOrder"] = 0
            intr["RadialDistortionFourthOrder"] = 0
            intr["RadialDistortionSixthOrder"] = 0
    else:
        print("intrinsics.json文件不存在于", os.path.dirname(depth_folder))
        sys.exit(1)
    
    while True:
        depth_file = "./depth.png"
        color_file = "./color.png"
        if depth_index < len(depth_files):
            depth_file = os.path.join(depth_folder, depth_files[depth_index])
            print("sending: ",depth_file)
            last_depth_frame = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            depth_index +=1
        else:
            flag_start = False
            flag_end = True
            show_msg = '所有深度图像已发送,等待接收结果'
        
        # last_color_frame = cv2.imread(color_file)
        last_color_frame = np.ones((300, 300, 3), dtype=np.uint8) * 128  # 128 是中灰色
        showimg = last_color_frame
        # cv2.putText(showimg, show_msg, orig, font, fontscale, color, 2)
        showimg = cv2.rotate(cv2.cvtColor(last_color_frame,cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        
        time.sleep(0.03)
        
        if flag_start:
            depthimg = last_depth_frame
            encoded_image = cv2.imencode('.png', depthimg)[1]
            data = base64.b64encode(np.array(encoded_image).tobytes())
            sd = {}
            # sd["flag_test"]=True
            sd["gender"]=gender
            if flag_weight_set:
                sd["weight"]=50.0
            sd["data"] = data.decode()
            sd["cmd"]="upload"
            sd["img_type"]="depth"
            sd["frame_id"]=str(fid)
            sd["name"]="张三"
            sd["vcode"]=vcode
            # intr = {'FocalLengthX': 435.32, 'FocalLengthY': 434.86, 'PrincipalPointX': 314.072, 'PrincipalPointY': 239.634, 'RadialDistortionSecondOrder': 0., 'RadialDistortionFourthOrder': -0., 'RadialDistortionSixthOrder': 0.}
            sd['intrinsics'] = intr
            sdstr = json.dumps(sd)
            if flag_cache_send:
                lock.acquire()
                msgs.append(sdstr)
                lock.release()
            else:
                ws.send(sdstr)
            
            if fid == 0:
                encoded_image = cv2.imencode('.png', last_color_frame)[1]
                data = base64.b64encode(np.array(encoded_image).tobytes())
                sd = {}
                sd["gender"]=gender
                sd["data"] = data.decode()
                sd["cmd"]="upload"
                sd["img_type"]="color"
                sd["frame_id"]=str(fid)
                sd["vcode"]=vcode
                sd["name"]="test"
                sdstr = json.dumps(sd)
                if flag_cache_send:
                    lock.acquire()
                    msgs.append(sdstr)
                    lock.release()
                else:
                    ws.send(sdstr)
                fid += 1
        
        if flag_end:
            sd = {}
            sd["gender"]=gender
            sd["data"] = ""
            sd["cmd"]="finish"
            sd["measure_type"]="qipao"
            sd["cloth_type"] = "kuansong"
            sd["vcode"]=vcode
            if flag_weight_set:
                sd["weight"]=weight
            sdstr = json.dumps(sd)
            if flag_cache_send:
                lock.acquire()
                msgs.append(sdstr)
                lock.release()
                print("cached msg size: ", len(msgs))
            else:
                ws.send(sdstr)
            
            stime = time.time()
            print("测量中，请稍候")
            fid = 0
            flag_start = False
            flag_end = False
            result = ws.recv()
            print("测量结果如下:",result)
            jr = json.loads(result)
            for key, value in jr.items():
                if key == "body_model":
                    with open(f"body_model.ply", "w") as f:
                        # print(key, value)
                        f.write((value))
                        f.close()
                else:
                    print(key, value)
            print("测量耗时: ", time.time()-stime," 秒")
            print("Press \'s\' to start and \'e\' to stop")
            print("测量完成，程序退出")
            lock.acquire()
            flag_exit = True
            lock.release()
            break