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
WINDOW_NAME="realsense"
#
#socket client
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_address = ('localhost', 9099)
# client_socket.connect(server_address)
#
def test_cvui():
    global WINDOW_NAME
    frame = np.zeros((1280, 768, 3), np.uint8)
    cvui.init(WINDOW_NAME)
    male_checked = False
    gender = "female"
    button_name = "start"
    while True:
        # Fill the frame with a nice color
        frame[:] = (49, 52, 49)

        # Render UI components to the frame
        cvui.text(frame, 110, 80, 'Hello, world!')
        cvui.text(frame, 110, 120, 'cvui is awesome!')
        cvui.checkbox(frame,110,160,"male",[male_checked])
        if cvui.button(frame, 110, 60, button_name):
            print('Button clicked')
            if button_name == "start":
                button_name = "end"
            else:
                button_name = "start"
        cvui.update()
        # Update cvui stuff and show everything on the screen
        cvui.imshow(WINDOW_NAME, frame)
        if male_checked:
            gender = "male"
            print(gender)
        if cv2.waitKey(20) == 27:
            break

depths = []
images=[]
import time
if __name__ == "__main__":
    test_cvui()
    exit()
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
    show_msg = "Ready"
    orig = (50,50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    color = (255,0,0)
    while True:
        rgbd_frame = rscam.capture_frame()
        # o3d.io.write_image(f"color/color{fid:05d}.png", rgbd_frame.color.to_legacy())
        # o3d.io.write_image(f"depth/depth{fid:05d}.png", rgbd_frame.depth.to_legacy())
        img_o3d_numpy = np.asarray(rgbd_frame.color.to_legacy())
        showimg = cv2.rotate(cv2.cvtColor(img_o3d_numpy,cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        cv2.putText(showimg, show_msg, orig, font, fontscale, color, 2)
        cv2.imshow("color",showimg)
        
        key = cv2.waitKey(30)

        if key == 115:
            print("数据正在实时上传服务器......")
            flag_start = True
            flag_end = False
            show_msg = "Processing..."
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