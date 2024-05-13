import open3d as o3d
import numpy as np
import cv2

volume = o3d.pipelines.integration.ScalableTSDFVolume(
voxel_length=4.0 / 512.0,
sdf_trunc=0.04,
color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

depths = []
images=[]
if __name__ == "__main__":
    o3d.t.io.RealSenseSensor.list_devices()
    rscam = o3d.t.io.RealSenseSensor()
    rscam.start_capture()
    cam_param = rscam.get_metadata()
    print(cam_param)
    for fid in range(150):
        print("frame:",fid)
        rgbd_frame = rscam.capture_frame()
        depths.append( rgbd_frame.depth.to_legacy())
        images.append(rgbd_frame.color.to_legacy())
        # o3d.io.write_image(f"color/color{fid:05d}.png", rgbd_frame.color.to_legacy())
        # o3d.io.write_image(f"depth/depth{fid:05d}.png", rgbd_frame.depth.to_legacy())
        img_o3d_numpy = np.asarray(rgbd_frame.color.to_legacy())
        cv2.imshow("color",img_o3d_numpy)
        cv2.waitKey(1)
    print("saving to disk")
    for fid in range(150):
        o3d.io.write_image(f"color/color{fid:05d}.png", images[fid])
        o3d.io.write_image(f"depth/depth{fid:05d}.png",depths[fid])
    rscam.stop_capture()