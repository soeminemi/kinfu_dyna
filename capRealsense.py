import open3d as o3d
import numpy as np
import cv2
import socket
import websocket
import base64
import json
import cvui

volume = o3d.pipelines.integration.ScalableTSDFVolume(
voxel_length=4.0 / 512.0,
sdf_trunc=0.04,
color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
#
#socket client
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_address = ('localhost', 9099)
# client_socket.connect(server_address)
#
depths = []
images=[]
import time
if __name__ == "__main__":
    font_path = "SimHei.ttf"

    sample_num = 1500
    flag_save_disk = False
    o3d.t.io.RealSenseSensor.list_devices()
    rscam = o3d.t.io.RealSenseSensor()
    rscam.start_capture()
    cam_param = rscam.get_metadata()
    print(cam_param)
    flag_start = False
    flag_end = False
    # for fid in range(sample_num):
    fid = 0
    ws_url = "ws://192.168.18.61:9099"
    ws = websocket.create_connection(ws_url)
    show_msg = "ready"
    orig = (50,50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    color = (255,0,0)
    WINDOW_NAME = "realsense"
    frame = np.zeros((1280, 768, 3), np.uint8)
    cvui.init(WINDOW_NAME)
    male_checked = [True]
    female_checked = [False]
    gender = "female"
    button_name = "START"
    result_str = "None"
    while True:
        rgbd_frame = rscam.capture_frame()
        # o3d.io.write_image(f"color/color{fid:05d}.png", rgbd_frame.color.to_legacy())
        # o3d.io.write_image(f"depth/depth{fid:05d}.png", rgbd_frame.depth.to_legacy())

        img_o3d_numpy = np.asarray(rgbd_frame.color.to_legacy())
        print(type(img_o3d_numpy))
        showimg = cv2.rotate(cv2.cvtColor(img_o3d_numpy,cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        
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
        
        key = cv2.waitKey(30)

        if key == 115:
            print("数据正在实时上传服务器......")
            flag_start = True
            flag_end = False
            show_msg = "数据正在实时上传服务器......"
        elif key == 101:
            if flag_start == False:
                break
            flag_end = True
            flag_start = False
            show_msg = "Wait for Server..."
        if flag_start:
            if flag_save_disk:
                depths.append( rgbd_frame.depth.to_legacy())
                images.append(rgbd_frame.color.to_legacy())
            depthimg = np.asarray(rgbd_frame.depth.to_legacy())
            encoded_image = cv2.imencode('.png', depthimg)[1]
            data = base64.b64encode(np.array(encoded_image).tobytes())
            # sd = {}
            # sd["gender"]=gender
            # sd["data"] = data
            ws.send(data)
            fid += 1

        if flag_end:
            ws.send("finished")
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
                if key == "measures":
                    result_str = json.dumps(value)
                    break
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
    rscam.stop_capture()