from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import socket
import websocket
import base64
import json
import cvui
import threading
# 获取深度图, 默认尺寸 424x512
def get_last_depth():
    frame = kinect.get_last_depth_frame()
    frame = frame.astype(np.uint16)
    # print("frame shape: ", frame.shape)
    dep_frame = np.reshape(frame, [424, 512])
    return dep_frame

#获取rgb图, 1080x1920x4
def get_last_rbg():
    frame = kinect.get_last_color_frame()
    return np.reshape(frame, [1080, 1920, 4])[:, :, 0:3]
#
#socket client
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_address = ('localhost', 9099)
# client_socket.connect(server_address)
#

depths = []
images=[]
lock = threading.RLock()
import time

def pub_msg(ws):
    global msgs
    while True:
        lock.acquire()
        if len(msgs) > 0:
            msg = msgs.pop(0)
            ws.send(msg)
        else:
            time.sleep(0.01)
        lock.release()

if __name__ == "__main__":
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
    sample_num = 1500
    flag_cache_send = True
    flag_save_disk = False
    flag_start = False
    flag_end = False
    # for fid in range(sample_num):
    fid = 0
    ws_url = "ws://http://zeroonearea.natapp1.cc:80"
    ws = websocket.create_connection(ws_url)
    show_msg = "Ready"
    orig = (50,50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    color = (255,0,0)
    flag_show_intrinsic = False
    WINDOW_NAME = "Kinect"
    frame = np.zeros((1920, 1080, 3), np.uint8)
    cvui.init(WINDOW_NAME)
    male_checked = [True]
    female_checked = [False]
    gender = "female"
    button_name = "START"
    result_str = "None"
    global msgs
    msgs = []
    if flag_cache_send:
        thread_sdmsg = threading.Thread(target=pub_msg, args=(ws))
        thread_sdmsg.start()
    while True:
        last_depth_frame = get_last_depth()
        last_color_frame = get_last_rbg()
        if flag_show_intrinsic:
            intrinsics_matrix = kinect._mapper.GetDepthCameraIntrinsics()
            print(intrinsics_matrix.FocalLengthX,intrinsics_matrix.FocalLengthY,intrinsics_matrix.PrincipalPointX,intrinsics_matrix.PrincipalPointY,intrinsics_matrix.RadialDistortionSecondOrder,intrinsics_matrix.RadialDistortionFourthOrder,intrinsics_matrix.RadialDistortionSixthOrder)
        showimg = last_color_frame
        # cv2.putText(showimg, show_msg, orig, font, fontscale, color, 2)
        showimg = cv2.rotate(cv2.cvtColor(last_color_frame,cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        
        if cvui.checkbox(showimg,50,60,"male",male_checked):
            female_checked=[False]
            gender = "male"
        if cvui.checkbox(showimg,50,80,"female",female_checked):
            male_checked = [False]
            gender = "female"

        if cvui.button(showimg, 50, 100,  button_name):
            if button_name == "START":
                button_name = "STOP"
                flag_start = True
                flag_end = False
                show_msg = "Sending Data to Server......"
                result_str = "None"
            else:
                button_name = "START"
                flag_end = True
                flag_start = False
                show_msg = 'Waiting for Measuring......'
                result_str = "None"
        if cvui.button(showimg, 250, 100,  "EXIT"):
            break
        cvui.text(showimg, 200, 20, show_msg, 1.0, 0x00ff00)
        cvui.text(showimg, 50, 140, result_str, 0.4, 0x00ff00)
        cvui.update()
        cv2.imshow(WINDOW_NAME,showimg)
        key = cv2.waitKey(3)

        if flag_start:
            if flag_save_disk:
                depths.append(last_depth_frame)
                images.append(last_color_frame)
            depthimg = last_depth_frame
            encoded_image = cv2.imencode('.png', depthimg)[1]
            data = base64.b64encode(np.array(encoded_image).tobytes())
            sd = {}
            sd["gender"]=gender
            sd["data"] = data.decode()
            sd["cmd"]="upload"
            sd["img_type"]="depth"
            sd["frame_id"]=str(fid)
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
            sd["cloth_type"] = "jinshen"
            sdstr = json.dumps(sd)
            ws.send(sdstr)
            stime = time.time()
            print("测量中，请稍候")
            fid = 0
            flag_start = False;
            flag_end = False
            result = ws.recv()
            print("测量结果如下")
            jr = json.loads(result)
            for key, value in jr.items():
                print(key, value)
            print("测量耗时: ", time.time()-stime," 秒")
            print("Press \'s\' to start and \'e\' to stop")
            show_msg = "Ready"

    if flag_save_disk:
        print("saving to disk")
        # path = "/home/john/Projects/dynamicfusion/data/desk1"
        path = "./data_kinfu/rotperson_1"
        for fid in range(len(depths)):
            o3d.io.write_image(f"{path}/color/color{fid:05d}.png", images[fid])
            o3d.io.write_image(f"{path}/depth/depth{fid:05d}.png",depths[fid])
            print("saving: ",fid)
    