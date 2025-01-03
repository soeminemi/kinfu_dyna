#pragma once

#include <kfusion/types.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/cuda/projective_icp.hpp>
#include <vector>
#include <string>
#include <kfusion/warp_field.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace kfusion
{
    namespace cuda
    {
        KF_EXPORTS int getCudaEnabledDeviceCount();
        KF_EXPORTS void setDevice(int device);
        KF_EXPORTS std::string getDeviceName(int device);
        KF_EXPORTS bool checkIfPreFermiGPU(int device);
        KF_EXPORTS void printCudaDeviceInfo(int device);
        KF_EXPORTS void printShortCudaDeviceInfo(int device);
    }

    struct KF_EXPORTS KinFuParams
    {
        static KinFuParams default_params();

        int cols;  //pixels
        int rows;  //pixels

        Intr intr;  //Camera parameters

        Vec3i volume_dims; //number of voxels
        Vec3f volume_size; //meters
        Affine3f volume_pose; //meters, inital pose

        float bilateral_sigma_depth;   //meters
        float bilateral_sigma_spatial;   //pixels
        int   bilateral_kernel_size;   //pixels

        float icp_truncate_depth_dist; //meters
        float icp_dist_thres;          //meters
        float icp_angle_thres;         //radians
        std::vector<int> icp_iter_num; //iterations for level index 0,1,..,3

        float tsdf_min_camera_movement; //meters, integrate only if exceedes
        float tsdf_trunc_dist;             //meters;
        int tsdf_max_weight;               //frames

        float raycast_step_factor;   // in voxel sizes
        float gradient_delta_factor; // in voxel sizes

        float depth_scale; //transform to mm from the depth image

        Vec3f light_pose; //meters

    };

    class KF_EXPORTS KinFu
    {
    public:        
        typedef cv::Ptr<KinFu> Ptr;

        KinFu(const KinFuParams& params);
        void set_params(kfusion::KinFuParams params);
        const KinFuParams& params() const;
        KinFuParams& params();

        const cuda::TsdfVolume& tsdf() const;
        cuda::TsdfVolume& tsdf();

        const cuda::ProjectiveICP& icp() const;
        cuda::ProjectiveICP& icp();

        const WarpField& getWarp() const;
        WarpField &getWarp();

        void reset();

        bool operator()(const cuda::Depth& dpeth, const cuda::Image& image = cuda::Image());

        void renderImage(cuda::Image& image, int flags = 0);
        void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);

        void getPoints(cv::Mat& points);
        void toPly(cv::Mat& points, cv::Mat &normals, std::string spath);
        void toPlyColor(cv::Mat& points, cv::Mat &normals, std::string spath, uint8_t r, uint8_t g, uint8_t b);
        void toPlyColorFilter(cv::Mat& points, cv::Mat &normals, std::string spath, uint8_t r, uint8_t g, uint8_t b);
        void toPlyVec3(cv::Mat& points, cv::Mat &normals, std::string spath);
        void toPlyVec3Color(cv::Mat& points, cv::Mat &normals, std::string spath, uint8_t r, uint8_t g, uint8_t b);
        void dynamicfusion(cuda::Depth& depth, cuda::Cloud live_frame, cuda::Normals current_normals);
        Affine3f getCameraPose (int time = -1) const;
        void loopClosureOptimize(std::vector<Affine3f>& poses,
                                std::vector<int> loop_frame_idx,
                                std::vector<Affine3f> loop_poses);
        void loopClosureOptimize();
        void append_depth_image(cv::Mat depth)
        {
            depth_imgs_.push_back(depth);
        }
        bool isLoopClosed() const { return flag_closed_; }
        bool isFinished() const { return flag_finish_; }

        // Convert depth image to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr depthToPCL(const cv::Mat& depth, const Intr& intr);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr depthToPCLWithNormals(const cv::Mat& depth, const Intr& intr);

    private:
        void allocate_buffers();

        int frame_counter_;
        KinFuParams params_;

        std::vector<Affine3f> poses_;
        std::vector<Affine3f> poses_frame_;

        cuda::Dists dists_;
        cuda::Frame curr_, prev_,first_, prev_frame_;

        cuda::Cloud points_;
        cuda::Normals normals_;
        cuda::Depth depths_;

        cv::Ptr<cuda::TsdfVolume> volume_;
        cv::Ptr<cuda::TsdfVolume> volume_loop_;
        cv::Ptr<cuda::ProjectiveICP> icp_;
        cv::Ptr<WarpField> warp_;
        bool flag_closed_;
        bool flag_finish_;
        std::vector<int> loop_frame_idx_;
        std::vector<Affine3f> loop_poses_;
        std::vector<cv::Mat> depth_imgs_;
        int first_frame_idx_ = 20;
        Affine3f affine_prev_; 
        Affine3f min_affine;
        int min_frame_idx = 0;
        std::vector<Affine3f> anchor_poses_;
        std::vector<cv::Mat> anchor_depths_;
    };
}
