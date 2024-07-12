import open3d as o3d
import numpy as np
import cv2
import socket


volume = o3d.pipelines.integration.ScalableTSDFVolume(
voxel_length=4.0 / 512.0,
sdf_trunc=0.04,
color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
#
#socket client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 9099)
client_socket.connect(server_address)
#
depths = []
images=[]

if __name__ == "__main__":
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
    while True:
        print("frame:",fid)
        rgbd_frame = rscam.capture_frame()
        
        # o3d.io.write_image(f"color/color{fid:05d}.png", rgbd_frame.color.to_legacy())
        # o3d.io.write_image(f"depth/depth{fid:05d}.png", rgbd_frame.depth.to_legacy())
        img_o3d_numpy = np.asarray(rgbd_frame.color.to_legacy())
        cv2.imshow("color",img_o3d_numpy)
        key = cv2.waitKey(1)
        print(key)
        if key == 115:
            flag_start = True
        elif key == 101:
            flag_end = True
        if flag_start:
            if flag_save_disk:
                depths.append( rgbd_frame.depth.to_legacy())
                images.append(rgbd_frame.color.to_legacy())

            encoded_image = cv2.imencode('.jpg', rgbd_frame.depth.to_legacy())[1]
            data = np.array(encoded_image).tostring()
            client_socket.sendall(data)

        if flag_end:
            break
    if flag_save_disk:
        print("saving to disk")
        # path = "/home/john/Projects/dynamicfusion/data/desk1"
        path = "./data_kinfu/rotperson_1"
        for fid in range(len(depths)):
            o3d.io.write_image(f"{path}/color/color{fid:05d}.png", images[fid])
            o3d.io.write_image(f"{path}/depth/depth{fid:05d}.png",depths[fid])
            print("saving: ",fid)
    rscam.stop_capture()