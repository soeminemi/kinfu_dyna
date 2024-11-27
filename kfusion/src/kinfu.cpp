#include "precomp.hpp"
#include "internal.hpp"
#include "io/ply.hpp"
// #include <ceres/ceres.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
using namespace std;
using namespace kfusion;
using namespace kfusion::cuda;

static inline float deg2rad (float alpha) { return alpha * 0.017453293f; }

// class PoseGraphError {
// public:
//     PoseGraphError(const Eigen::Matrix4f& relative_pose) : relative_pose_(relative_pose) {}

//     template <typename T>
//     bool operator()(const T* const pose1, const T* const pose2, T* residuals) const {
//         // 将输入的位姿转换为 Eigen 矩阵
//         Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation1(pose1 + 4);
//         Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation2(pose2 + 4);
//         Eigen::Map<const Eigen::Matrix<T, 4, 1>> quaternion1(pose1);
//         Eigen::Map<const Eigen::Matrix<T, 4, 1>> quaternion2(pose2);

//         // 将四元数转换为旋转矩阵
//         Eigen::Matrix<T, 3, 3> R1 = Eigen::Quaternion<T>(quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]).toRotationMatrix();
//         Eigen::Matrix<T, 3, 3> R2 = Eigen::Quaternion<T>(quaternion2[0], quaternion2[1], quaternion2[2], quaternion2[3]).toRotationMatrix();

//         // 计算相对位姿
//         Eigen::Matrix<T, 3, 1> t1 = translation1;
//         Eigen::Matrix<T, 3, 1> t2 = translation2;
//         Eigen::Matrix<T, 3, 1> t_relative = t2 - t1;

//         // 计算误差
//         Eigen::Matrix<T, 3, 1> t_error = t_relative - relative_pose_.block<3, 1>(0, 3);
//         Eigen::Matrix<T, 3, 3> R_error = R2 * R1.transpose() - relative_pose_.block<3, 3>(0, 0);

//         // 将误差存储到 residuals 中
//         residuals[0] = t_error.norm();
//         residuals[1] = R_error.norm();
//         residuals[2] = R_error.norm();
//         return true;
//     }

//     static ceres::CostFunction* Create(const Eigen::Matrix4f& relative_pose) {
//         return new ceres::AutoDiffCostFunction<PoseGraphError, 6, 7, 7>(
//             new PoseGraphError(relative_pose));
//     }

// private:
//     Eigen::Matrix4f relative_pose_;
// };

kfusion::KinFuParams kfusion::KinFuParams::default_params()
{
    const int iters[] = {10, 5, 4, 0};
    const int levels = sizeof(iters)/sizeof(iters[0]);

    KinFuParams p;

    // p.cols = 640;  //pixels
    // p.rows = 480;  //pixels
    // p.intr = Intr(598.68896484375f, 599.1634521484375f, 328.7f, 235.72f);
    // if (device_type == "realsense")
    // {
    //     p.cols = 1280;  //pixels
    //     p.rows = 780;  //pixels
    //     p.intr = Intr(898.033f, 898.745f, 653.17f, 353.58f);
    // }
    // else if(device_type == "kinect")
    // {
        p.cols = 512;  //pixels
        p.rows = 414;  //pixels
        p.intr = Intr(365.3566f, 365.3566f, 261.4155f, 206.6168f);
    // }
    // else
    // {
    //     p.cols = 1280;  //pixels
    //     p.rows = 780;  //pixels
    //     p.intr = Intr(898.033f, 898.745f, 653.17f, 353.58f);
    // }
    p.volume_dims = Vec3i::all(256);  //number of voxels
    p.volume_size = Vec3f::all(2.0f);  //meters
    p.volume_pose = Affine3f().translate(Vec3f(-p.volume_size[0]/2, -p.volume_size[1]/2, 0.9f)); //设置初始

    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icp_truncate_depth_dist = 0.f;        //meters, disabled
    p.icp_dist_thres = 0.1f;                //meters
    p.icp_angle_thres = deg2rad(30.f); //radians
    p.icp_iter_num.assign(iters, iters + levels);

    p.tsdf_min_camera_movement = 0.f; //meters, disabled
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.75f;  //in voxel sizes
    p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.light_pose = p.volume_pose.translation()/4; //meters
    p.light_pose = Vec3f::all(0.f); //meters
    p.depth_scale = 0.25;

    return p;
}

kfusion::KinFu::KinFu(const KinFuParams& params) : frame_counter_(0), params_(params)
{
    CV_Assert(params.volume_dims[0] % 32 == 0);

    volume_ = cv::Ptr<cuda::TsdfVolume>(new cuda::TsdfVolume(params_.volume_dims));
    warp_ = cv::Ptr<WarpField>(new WarpField());
    volume_->setTruncDist(params_.tsdf_trunc_dist);
    volume_->setMaxWeight(params_.tsdf_max_weight);
    volume_->setSize(params_.volume_size);
    volume_->setPose(params_.volume_pose);
    volume_->setRaycastStepFactor(params_.raycast_step_factor);
    volume_->setGradientDeltaFactor(params_.gradient_delta_factor);
    volume_->setDepthScale(params_.depth_scale);
    icp_ = cv::Ptr<cuda::ProjectiveICP>(new cuda::ProjectiveICP());
    icp_->setDistThreshold(params_.icp_dist_thres);
    icp_->setAngleThreshold(params_.icp_angle_thres);
    icp_->setIterationsNum(params_.icp_iter_num);

    volume_loop_ = cv::Ptr<cuda::TsdfVolume>(new cuda::TsdfVolume(params_.volume_dims));
    volume_loop_->setTruncDist(params_.tsdf_trunc_dist);
    volume_loop_->setMaxWeight(params_.tsdf_max_weight);
    volume_loop_->setSize(params_.volume_size);
    volume_loop_->setPose(params_.volume_pose);
    volume_loop_->setRaycastStepFactor(params_.raycast_step_factor);
    volume_loop_->setGradientDeltaFactor(params_.gradient_delta_factor);
    volume_loop_->setDepthScale(params_.depth_scale);

    allocate_buffers();
    reset();
}

void kfusion::KinFu::set_params(const kfusion::KinFuParams params)
{  params_ = params; }

const kfusion::KinFuParams& kfusion::KinFu::params() const
{ return params_; }

kfusion::KinFuParams& kfusion::KinFu::params()
{ return params_; }

const kfusion::cuda::TsdfVolume& kfusion::KinFu::tsdf() const
{ return *volume_; }

kfusion::cuda::TsdfVolume& kfusion::KinFu::tsdf()
{ return *volume_; }

const kfusion::cuda::ProjectiveICP& kfusion::KinFu::icp() const
{ return *icp_; }

kfusion::cuda::ProjectiveICP& kfusion::KinFu::icp()
{ return *icp_; }

void kfusion::KinFu::allocate_buffers()
{
    const int LEVELS = cuda::ProjectiveICP::MAX_PYRAMID_LEVELS;

    int cols = params_.cols;
    int rows = params_.rows;

    dists_.create(rows, cols);

    curr_.depth_pyr.resize(LEVELS);
    curr_.normals_pyr.resize(LEVELS);
    prev_.depth_pyr.resize(LEVELS);

    prev_.normals_pyr.resize(LEVELS);
    first_.depth_pyr.resize(LEVELS);
    first_.normals_pyr.resize(LEVELS);
    prev_frame_.depth_pyr.resize(LEVELS);
    prev_frame_.normals_pyr.resize(LEVELS);

    first_.points_pyr.resize(LEVELS);
    curr_.points_pyr.resize(LEVELS);
    prev_.points_pyr.resize(LEVELS);
    prev_frame_.points_pyr.resize(LEVELS);

    for(int i = 0; i < LEVELS; ++i)
    {
        curr_.depth_pyr[i].create(rows, cols);
        curr_.normals_pyr[i].create(rows, cols);
        first_.depth_pyr[i].create(rows, cols);
        first_.normals_pyr[i].create(rows, cols);
        prev_.depth_pyr[i].create(rows, cols);
        prev_.normals_pyr[i].create(rows, cols);
        prev_frame_.depth_pyr[i].create(rows, cols);
        prev_frame_.normals_pyr[i].create(rows, cols);

        curr_.points_pyr[i].create(rows, cols);
        prev_.points_pyr[i].create(rows, cols);
        first_.points_pyr[i].create(rows, cols);
        prev_frame_.points_pyr[i].create(rows, cols);

        cols /= 2;
        rows /= 2;
    }

    depths_.create(params_.rows, params_.cols);
    normals_.create(params_.rows, params_.cols);
    points_.create(params_.rows, params_.cols);
}

void kfusion::KinFu::reset()
{
    if (frame_counter_)
        cout << "Reset" << endl;
    //reset the frame counter
    frame_counter_ = 0;
    poses_.clear();
    poses_.reserve(3000);
    poses_.push_back(Affine3f::Identity());
    poses_frame_.clear();
    poses_frame_.reserve(3000);
    poses_frame_.push_back(Affine3f::Identity());
    loop_frame_idx_.clear();
    loop_frame_idx_.reserve(3000);
    loop_poses_.clear();
    loop_poses_.reserve(3000);
    flag_closed_ = false;
    
    volume_->clear();
    warp_->clear();
    depth_imgs_.clear();
    depth_imgs_.reserve(3000);

}

kfusion::Affine3f kfusion::KinFu::getCameraPose (int time) const
{
    if (time > (int)poses_.size () || time < 0)
        time = (int)poses_.size () - 1;
    return poses_[time];
}
const kfusion::WarpField& kfusion::KinFu::getWarp() const
{
    return *warp_; 
}

kfusion::WarpField &kfusion::KinFu::getWarp()
{
    return *warp_;
}

// main procedure of dynamic fusion
bool kfusion::KinFu::operator()(const kfusion::cuda::Depth& depth, const kfusion::cuda::Image& /*image*/)
{
    cout<<"frame_counter_ is: "<<frame_counter_<<endl;
    const KinFuParams& p = params_;
    const int LEVELS = icp_->getUsedLevelsNum();
    cuda::computeDists(depth, dists_, p.intr);
    cuda::depthBilateralFilter(depth, curr_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);

    if (p.icp_truncate_depth_dist > 0)
        kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);

    for (int i = 1; i < LEVELS; ++i)
        cuda::depthBuildPyramid(curr_.depth_pyr[i-1], curr_.depth_pyr[i], p.bilateral_sigma_depth);

    for (int i = 0; i < LEVELS; ++i)
#if defined USE_DEPTH
        cuda::computeNormalsAndMaskDepth(p.intr(i), curr_.depth_pyr[i], curr_.normals_pyr[i]);
#else
        cuda::computePointNormals(p.intr(i), curr_.depth_pyr[i], curr_.points_pyr[i], curr_.normals_pyr[i]);
#endif

    cuda::waitAllDefaultStream();
    float last_angle = 0;
    Affine3f last_affine;
    bool flag_integrate = false;
    // 对affine进行卡尔曼滤波处理
    static cv::Mat Pk = cv::Mat::eye(6, 6, CV_32F) * 1000;  // 初始协方差矩阵
    static cv::Mat Qk = cv::Mat::eye(6, 6, CV_32F) * 0.1;   // 过程噪声
    static cv::Mat Rk = cv::Mat::eye(6, 6, CV_32F) * 0.1;   // 测量噪声
    static cv::Mat xk = cv::Mat::zeros(6, 1, CV_32F);       // 状态向量 [x,y,z,rx,ry,rz]
    //can't perform more on first frame
    if (frame_counter_ == 0)
    {
        // 对affine进行卡尔曼滤波处理
        // 使用OpenCV初始化卡尔曼滤波器的状态
        cv::Mat Pk = cv::Mat::eye(6, 6, CV_32F) * 1000;  // 初始协方差矩阵
        cv::Mat Qk = cv::Mat::eye(6, 6, CV_32F) * 0.1;   // 过程噪声
        cv::Mat Rk = cv::Mat::eye(6, 6, CV_32F) * 0.1;   // 测量噪声
        cv::Mat xk = cv::Mat::zeros(6, 1, CV_32F);       // 状态向量 [x,y,z,rx,ry,rz]
        cout<<"first frame, try to integrate"<<endl;
        cuda::depthBilateralFilter(depth, first_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);
        if (p.icp_truncate_depth_dist > 0)
            kfusion::cuda::depthTruncation(first_.depth_pyr[0], p.icp_truncate_depth_dist);

        for (int i = 1; i < LEVELS; ++i)
            cuda::depthBuildPyramid(first_.depth_pyr[i-1], first_.depth_pyr[i], p.bilateral_sigma_depth);

        for (int i = 0; i < LEVELS; ++i){
        #if defined USE_DEPTH
            cuda::computeNormalsAndMaskDepth(p.intr(i), first_.depth_pyr[i], first_.normals_pyr[i]);
        #else
            cuda::computePointNormals(p.intr(i), first_.depth_pyr[i], first_.points_pyr[i], first_.normals_pyr[i]);
        #endif
        }
        cuda::waitAllDefaultStream();
        flag_closed_ = false;
        volume_->integrate(dists_, poses_.back(), p.intr);
#if defined USE_DEPTH
        curr_.depth_pyr.swap(prev_.depth_pyr);
#else
        curr_.points_pyr.swap(prev_.points_pyr);
#endif
        curr_.normals_pyr.swap(prev_.normals_pyr);

        // //
        // {
        //     #if defined USE_DEPTH
        //     curr_.depth_pyr.swap(prev_frame_.depth_pyr);
        //     #else
        //         curr_.points_pyr.swap(prev_frame_.points_pyr);
        //     #endif
        //     curr_.normals_pyr.swap(prev_frame_.normals_pyr);
        // }

        //initialize the warp field 
        // cv::Mat frame_init;
        // volume_->computePoints(frame_init);
        // auto aff = volume_->getPose();
        // aff = aff.inv();
        // std::cout<<"init affine: "<<aff.rotation()<<", "<<aff.translation()<<std::endl;
        // warp_->init(frame_init, params_.volume_dims, aff);
        // auto init_nodes = warp_->getNodesAsMat();
        // toPlyVec3(init_nodes, init_nodes, "init_nodes.ply");
        return ++frame_counter_, true;
    }
    if(flag_closed_)
    {
        return ++frame_counter_, true;
    }
    // ICP
    Affine3f affine; // cuur -> prev
    {
        //ScopeTime time("icp");
#if defined USE_DEPTH
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.depth_pyr, curr_.normals_pyr, prev_.depth_pyr, prev_.normals_pyr);
#else
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr, prev_.normals_pyr);
#endif
        if (!ok)
        {
            return reset(), false;
        }
    }
    
    std::cout << "Before Kalman Filtering, Affine变换矩阵:" << affine.rotation() << std::endl;
    std::cout << "Before Kalman Filtering, 平移向量:" << affine.translation() << std::endl;
    cv::Vec3f t = cv::Vec3f(affine.translation()[0], affine.translation()[1], affine.translation()[2]);
    cv::Mat rot = cv::Mat(affine.rotation());
    cv::Vec3f euler;
    rot.convertTo(rot, CV_32F);
    
    // 定义rotationMatrixToEulerAngles函数
    // 辅助函数
    auto rotationMatrixToEulerAngles = [](cv::Mat &R) -> cv::Vec3f {
        float sy = sqrt(R.at<float>(0,0) * R.at<float>(0,0) +  R.at<float>(1,0) * R.at<float>(1,0) );
        bool singular = sy < 1e-6;
        float x, y, z;
        if (!singular)
        {
            x = atan2(R.at<float>(2,1) , R.at<float>(2,2));
            y = atan2(-R.at<float>(2,0), sy);
            z = atan2(R.at<float>(1,0), R.at<float>(0,0));
        }
        else
        {
            x = atan2(-R.at<float>(1,2), R.at<float>(1,1));
            y = atan2(-R.at<float>(2,0), sy);
            z = 0;
        }
        return cv::Vec3f(x, y, z);
    };
    
    auto eulerAnglesToRotationMatrix = [](cv::Vec3f &theta) -> cv::Mat {
        cv::Mat R_x = (cv::Mat_<float>(3,3) <<
                       1,       0,              0,
                       0,       cos(theta[0]),   -sin(theta[0]),
                       0,       sin(theta[0]),   cos(theta[0])
                       );
        cv::Mat R_y = (cv::Mat_<float>(3,3) <<
                       cos(theta[1]),    0,      sin(theta[1]),
                       0,               1,      0,
                       -sin(theta[1]),   0,      cos(theta[1])
                       );
        cv::Mat R_z = (cv::Mat_<float>(3,3) <<
                       cos(theta[2]),    -sin(theta[2]),      0,
                       sin(theta[2]),    cos(theta[2]),       0,
                       0,               0,                  1);
        cv::Mat R = R_z * R_y * R_x;
        return R;
    };
    
    euler = rotationMatrixToEulerAngles(rot);
    
    // 测量向量
    cv::Mat z = (cv::Mat_<float>(6,1) << t[0], t[1], t[2], euler[0], euler[1], euler[2]);
    
    // 预测步骤
    cv::Mat F = cv::Mat::eye(6, 6, CV_32F); // 状态转移矩阵
    xk = F * xk;  // 预测状态
    Pk = F * Pk * F.t() + Qk;  // 预测协方差
    
    // 更新步骤
    cv::Mat H = cv::Mat::eye(6, 6, CV_32F); // 观测矩阵
    cv::Mat S = H * Pk * H.t() + Rk;
    cv::Mat K = Pk * H.t() * S.inv(); // 卡尔曼增益
    
    xk = xk + K * (z - H * xk);  // 更新状态
    Pk = (cv::Mat::eye(6, 6, CV_32F) - K * H) * Pk;  // 更新协方差
    
    // 将滤波后的结果转换回Affine3f
    cv::Vec3f filtered_t(xk.at<float>(0), xk.at<float>(1), xk.at<float>(2));
    cv::Vec3f filtered_euler_angles(xk.at<float>(3), xk.at<float>(4), xk.at<float>(5));
    cv::Mat filtered_rot = eulerAnglesToRotationMatrix(filtered_euler_angles);
    
    cv::Matx33f rotMat;
    filtered_rot.copyTo(rotMat);
    affine = Affine3f(rotMat, filtered_t);
    
    std::cout << "After Kalman Filtering, Affine变换矩阵:" << affine.rotation() << std::endl;
    std::cout << "After Kalman Filtering, 平移向量:" << affine.translation() << std::endl;
    
    cout<<"try to push back pose"<<endl;
    poses_.push_back(poses_.back() * affine); // curr -> global， affine pre->curr
    // 将旋转矩阵转换为欧拉角(弧度)
    cv::Mat R = cv::Mat(poses_.back().rotation());
    cv::Vec3f euler_angles;
    euler_angles[0] = atan2(R.at<float>(2,1), R.at<float>(2,2));
    euler_angles[1] = atan2(-R.at<float>(2,0), sqrt(R.at<float>(2,1)*R.at<float>(2,1) + R.at<float>(2,2)*R.at<float>(2,2)));
    euler_angles[2] = atan2(R.at<float>(1,0), R.at<float>(0,0));
    
    // 转换为角度并输出
    std::cout<<"当前帧号: "<<frame_counter_<<std::endl;
    std::cout << "旋转角度(度): roll=" << euler_angles[0]*180/M_PI 
              << ", pitch=" << euler_angles[1]*180/M_PI
              << ", yaw=" << euler_angles[2]*180/M_PI << std::endl;
    float roll_angle = euler_angles[0]*180/M_PI;
   
    if(fabs(roll_angle)<15 && frame_counter_>100)
    {
        Affine3f taffine;
        bool ok = icp_->estimateTransform(taffine, p.intr,curr_.points_pyr, curr_.normals_pyr, first_.points_pyr, first_.normals_pyr);
        if(ok)
        {
            // 将taffine的旋转矩阵转换为欧拉角
            cv::Mat tR = cv::Mat(taffine.rotation());
            cv::Vec3f t_euler;
            t_euler[0] = atan2(tR.at<float>(2,1), tR.at<float>(2,2));
            t_euler[1] = atan2(-tR.at<float>(2,0), sqrt(tR.at<float>(2,1)*tR.at<float>(2,1) + tR.at<float>(2,2)*tR.at<float>(2,2)));
            t_euler[2] = atan2(tR.at<float>(1,0), tR.at<float>(0,0));
            
            // 转换为角度并输出
            std::cout << "taffine旋转角度(度): roll=" << t_euler[0]*180/M_PI 
                    << ", pitch=" << t_euler[1]*180/M_PI 
                    << ", yaw=" << t_euler[2]*180/M_PI << std::endl;
            std::cout << "taffine: " << taffine.translation()<<endl<<", "<<taffine.rotation() << std::endl;
            
            loop_frame_idx_.push_back(frame_counter_);
            loop_poses_.push_back(taffine);
        }
        if(loop_frame_idx_.size()>10)
        {
            flag_closed_ = true;
        }
    }
    if(flag_closed_ == false )
    {
        ///////////////////////////////////////////////////////////////////////////////////////////
        // Volume integration
        auto d = curr_.depth_pyr[0];
        auto pts = curr_.points_pyr[0];
        auto n = curr_.normals_pyr[0];
        // std::cout<<"dynamic fusion to canonical space"<<std::endl;
        // dynamicfusion(d, pts, n);
        // We do not integrate volume if camera does not move.
        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        bool integrate = (rnorm + tnorm)/2 >= p.tsdf_min_camera_movement;
        if (true)
        {
            //ScopeTime time("tsdf");
            volume_->integrate(dists_, poses_.back(), p.intr);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Ray casting
        {
            //ScopeTime time("ray-cast-all");
    #if defined USE_DEPTH
            volume_->raycast(poses_.back(), p.intr, prev_.depth_pyr[0], prev_.normals_pyr[0]);
            for (int i = 1; i < LEVELS; ++i)
                resizeDepthNormals(prev_.depth_pyr[i-1], prev_.normals_pyr[i-1], prev_.depth_pyr[i], prev_.normals_pyr[i]);
    #else
            volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]); // tsdf volume to points pyramid
            //warp 到当前的形状, 先刚性变换到当前视角再warp还是先warp再刚性变换？这个应该是有区别的
            // warp_->warp(prev_.points_pyr[0],prev_.normals_pyr[0]);
            //
            for (int i = 1; i < LEVELS; ++i)
                resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
    #endif
            cuda::waitAllDefaultStream();
        } 
        if(frame_counter_ == first_frame_idx_)
        {
            #if defined USE_DEPTH
            volume_->raycast(poses_.back(), p.intr, first_.depth_pyr[0], first_.normals_pyr[0]);
            for (int i = 1; i < LEVELS; ++i)
                resizeDepthNormals(first_.depth_pyr[i-1], first_.normals_pyr[i-1], first_.depth_pyr[i], first_.normals_pyr[i]);
    #else
            volume_->raycast(poses_[0], p.intr, first_.points_pyr[0], first_.normals_pyr[0]); // tsdf volume to points pyramid
            for (int i = 1; i < LEVELS; ++i)
                resizePointsNormals(first_.points_pyr[i-1], first_.normals_pyr[i-1], first_.points_pyr[i], first_.normals_pyr[i]);
    #endif
            cuda::waitAllDefaultStream();
        }
        // {  
        //     //测试代码段，改为前后帧匹配 
        //     #if defined USE_DEPTH
        //     curr_.depth_pyr.swap(prev_frame_.depth_pyr);
        //     #else
        //         curr_.points_pyr.swap(prev_frame_.points_pyr);
        //     #endif
        //     curr_.normals_pyr.swap(prev_frame_.normals_pyr);
        // }
    }
    else
    {
        ;
    }
    return ++frame_counter_, true;
}
void kfusion::KinFu::getPoints(cv::Mat& points)
{
    prev_.points_pyr[0].download(points.ptr<void>(), points.step);
}
void kfusion::KinFu::loopClosureOptimize()
{
    if(flag_closed_)
    {
        loopClosureOptimize(poses_, loop_frame_idx_, loop_poses_);
    }
    else{
        cout<<"no loop closure"<<endl;
    }
}
void kfusion::KinFu::loopClosureOptimize(
                                        std::vector<Affine3f>& poses,
                                        std::vector<int> loop_frame_idx,
                                        std::vector<Affine3f> loop_poses) {
    std::cout << "开始闭环优化..." << std::endl;
    std::cout << "poses大小: " << poses.size() << std::endl;
    std::cout << "闭环帧序号: " << loop_frame_idx[0] << std::endl; 
    std::cout << "depth images大小: " << depth_imgs_.size() << std::endl;
    // 帧总数
    int frame_count = poses.size();
    for(int i=0; i<loop_frame_idx.size(); i++)
    {
        if(loop_frame_idx[i] >= frame_count) {
            std::cout << "错误:闭环帧序号超出范围" << std::endl;
        return;
        }
    }
    // 构建图优化问题
    auto cv2eigen = [](const cv::Mat& cvMat, Eigen::Matrix3d& eigenMat) {
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                eigenMat(i,j) = cvMat.at<double>(i,j);
            }
        }
    };
    auto eigen2cv = [](const Eigen::Matrix3d& eigenMat, cv::Mat& cvMat) {
        cvMat = cv::Mat(3, 3, CV_64F);
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                cvMat.at<double>(i,j) = eigenMat(i,j);
            }
        }
    };
    // 创建优化器
    g2o::SparseOptimizer optimizer;
    
    // 配置求解器
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver = 
        std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = 
        new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);

    // 添加顶点
    for(int i = 0; i < frame_count; i++) {
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(i);
        
        // 设置顶点初始估计值
        Eigen::Isometry3d pose;
        cv::Mat R;
        cv::Mat(poses[i].rotation()).convertTo(R, CV_64F);
        Eigen::Matrix3d rotation;
        
        cv2eigen(R, rotation);
        pose.linear() = rotation;
        pose.translation() = Eigen::Vector3d(
            poses[i].translation()[0],
            poses[i].translation()[1], 
            poses[i].translation()[2]);
        
        v->setEstimate(pose);
        
        // 第一帧固定
        if(i == 0)
            v->setFixed(true);
            
        optimizer.addVertex(v);
    }

    // 添加帧间约束边
    for(int i = 1; i < frame_count; i++) {
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->setId(i);
        edge->setVertex(0, optimizer.vertex(i-1));
        edge->setVertex(1, optimizer.vertex(i));
        
        // 计算相对位姿约束
        Eigen::Isometry3d relative_pose = Eigen::Isometry3d::Identity();
        cv::Mat R;
        cv::Mat((poses[i-1].inv() * poses[i]).rotation()).convertTo(R, CV_64F);
        Eigen::Matrix3d rotation;
        cv2eigen(R, rotation);
        relative_pose.linear() = rotation;
        auto t = (poses[i-1].inv() * poses[i]).translation();
        relative_pose.translation() = Eigen::Vector3d(t[0], t[1], t[2]);
        
        edge->setMeasurement(relative_pose);
        
        // 设置信息矩阵
        // 计算边的长度(相对位移的模长)作为权重
        double edge_length = relative_pose.translation().norm();
        Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Identity() ;//* edge_length;
        edge->setInformation(information);
        
        optimizer.addEdge(edge);
    }

    // 添加闭环约束边
    for(int i=0; i<loop_frame_idx.size(); i++)
    {
        g2o::EdgeSE3* loop_edge = new g2o::EdgeSE3();
        loop_edge->setVertex(0, optimizer.vertex(0));
        loop_edge->setVertex(1, optimizer.vertex(loop_frame_idx[i])); //添加从初始帧到闭环帧的边
        
        // 计算闭环约束
        Eigen::Isometry3d loop_constraint = Eigen::Isometry3d::Identity();
        cv::Mat R;
        cv::Mat(( loop_poses[i]).rotation()).convertTo(R, CV_64F);
        Eigen::Matrix3d rotation;
        cv2eigen(R, rotation);
        loop_constraint.linear() = rotation;
        auto t = ( loop_poses[i]).translation();
        loop_constraint.translation() = Eigen::Vector3d(t[0], t[1], t[2]);
    
        loop_edge->setMeasurement(loop_constraint);
    
        // 设置闭环约束的信息矩阵(权重更大)
        Eigen::Matrix<double,6,6> loop_information = Eigen::Matrix<double,6,6>::Identity() * 0.1;
        loop_edge->setInformation(loop_information);
        optimizer.addEdge(loop_edge);
    }

    // 执行优化
    cout<<"start optimization g2o"<<endl;
    optimizer.initializeOptimization();
    int iterations = optimizer.optimize(30);
    cout<<"end optimization g2o"<<endl;
    // 输出优化相关信息
    double chi2 = optimizer.chi2();
    std::cout << "优化信息:" << std::endl;
    std::cout << "- 优化迭代次数: " << iterations << std::endl;
    std::cout << "- 最终误差值: " << chi2 << std::endl;
    std::cout << "- 边的数量: " << optimizer.edges().size() << std::endl;
    std::cout << "- 顶点的数量: " << optimizer.vertices().size() << std::endl;
    // 获取优化结果更新poses
    for(int i = 0; i < frame_count; i++) {
        g2o::VertexSE3* v = static_cast<g2o::VertexSE3*>(optimizer.vertex(i));
        Eigen::Isometry3d pose = v->estimate();
        
        cv::Mat R;
        Eigen::Matrix3d rotation = pose.rotation();
        eigen2cv(rotation, R);
        R.convertTo(R, CV_32F);
        if(i==frame_count-1)
            cout<<"origin pose: "<<poses[i].translation()<<", "<<poses[i].rotation()<<endl;
        poses[i] = Affine3f(R, Vec3f(
            pose.translation().x(),
            pose.translation().y(),
            pose.translation().z()
        ));
        if(i==frame_count-1)
            cout<<"optimized pose: "<<poses[i].translation()<<", "<<poses[i].rotation()<<endl;
        // cout<<"--------------------------------"<<endl;
    }
    std::cout << "最终相机位姿:" << std::endl;
    for(int i =  poses_.size()-1; i < poses_.size(); i++) {
        std::cout << "位姿 " << i << ": " 
                 << "平移=" << poses_[i].translation() <<endl
                 << ", 旋转=" << poses_[i].rotation() << std::endl;
    }
    // 使用新的poses重新计算TSDF体素
    cv::Mat view_host_;
    cuda::Image view_device_;
    volume_->clear(); 
    volume_loop_->clear();
    cuda::Depth depth_device_tmp_;
    // 重新积分每一帧到TSDF
    cout<<"start reintegrate"<<endl;
    auto &p = params_;
    const int LEVELS = icp_->getUsedLevelsNum();
    //先合成一个闭环的TSDF，用于后续位姿估计的基准
    for(int i=0; i<frame_count; i++)
    {
        // 稀疏帧合成作为锚点，再次基础上进行闭环优化
        int j=i%40;
        if(j>3)
            continue;
        auto &depth = depth_imgs_[i];
        depth_device_tmp_.upload(depth.data, depth.step, depth.rows, depth.cols);
        cuda::computeDists(depth_device_tmp_, dists_, p.intr);
        volume_->integrate(dists_, poses[i], p.intr);
        if(false)
        { 
            volume_->raycast(poses[i], p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]); 
            for (int i = 1; i < LEVELS; ++i)
                resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
            cuda::waitAllDefaultStream();
            // 在当前相机视角下进行raycast
            renderImage(view_device_, 0);
            view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
            view_device_.download(view_host_.ptr<void>(), view_host_.step);
            cv::imshow("loopSceneLoop", view_host_);
            cv::waitKey(1);
        }
    }
    cout<<"reintegrate again"<<endl;
    for(int i = 0; i < frame_count; i++) {
        auto &depth = depth_imgs_[i];
        depth_device_tmp_.upload(depth.data, depth.step, depth.rows, depth.cols);
        cuda::computeDists(depth_device_tmp_, dists_, p.intr);
        cuda::depthBilateralFilter(depth_device_tmp_, curr_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);
        if (p.icp_truncate_depth_dist > 0)
            kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);
        for (int i = 1; i < LEVELS; ++i)
            cuda::depthBuildPyramid(curr_.depth_pyr[i-1], curr_.depth_pyr[i], p.bilateral_sigma_depth);
        for (int i = 0; i < LEVELS; ++i)
            cuda::computePointNormals(p.intr(i), curr_.depth_pyr[i], curr_.points_pyr[i], curr_.normals_pyr[i]);
        cuda::waitAllDefaultStream();
        
        if(i==0)
        {
            cuda::computeDists(depth_device_tmp_, dists_, p.intr);
            cuda::depthBilateralFilter(depth_device_tmp_, first_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);
            if (p.icp_truncate_depth_dist > 0)
                kfusion::cuda::depthTruncation(first_.depth_pyr[0], p.icp_truncate_depth_dist);
            for (int i = 1; i < LEVELS; ++i)
                cuda::depthBuildPyramid(first_.depth_pyr[i-1], first_.depth_pyr[i], p.bilateral_sigma_depth);
            for (int i = 0; i < LEVELS; ++i)
                cuda::computePointNormals(p.intr(i), first_.depth_pyr[i], first_.points_pyr[i], first_.normals_pyr[i]);
            cuda::waitAllDefaultStream();
            volume_->integrate(dists_, poses[i], p.intr);
            continue;
        }
        for(int j=0; j<1; j++)
        {
            volume_->raycast(poses[i], p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]); 
            for (int i = 1; i < LEVELS; ++i)
            resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
            cuda::waitAllDefaultStream();
            //重新估算pose
            Affine3f affine;
            bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr, prev_.normals_pyr);
            poses[i] = poses[i] * affine;
        }
        
        // if(i<frame_count-2)
        // {
        //     poses[i+1] = poses[i+1] * affine;//更新下一帧的pose
        // }
        if(i==loop_frame_idx[0])
        {
            //输出闭环前后的偏差
            Affine3f taffine;
             icp_->estimateTransform(taffine, p.intr, curr_.points_pyr, curr_.normals_pyr, first_.points_pyr, first_.normals_pyr);
            // 计算poses[i]的欧拉角
            cv::Mat R_pose = cv::Mat(poses[i].rotation());
            cv::Vec3f euler_pose;
            euler_pose[0] = atan2(R_pose.at<float>(2,1), R_pose.at<float>(2,2));
            euler_pose[1] = atan2(-R_pose.at<float>(2,0), sqrt(R_pose.at<float>(2,1)*R_pose.at<float>(2,1) + R_pose.at<float>(2,2)*R_pose.at<float>(2,2)));
            euler_pose[2] = atan2(R_pose.at<float>(1,0), R_pose.at<float>(0,0));
            
            // 计算taffine的欧拉角
            cv::Mat R_taffine = cv::Mat(taffine.rotation());
            cv::Vec3f euler_taffine;
            euler_taffine[0] = atan2(R_taffine.at<float>(2,1), R_taffine.at<float>(2,2));
            euler_taffine[1] = atan2(-R_taffine.at<float>(2,0), sqrt(R_taffine.at<float>(2,1)*R_taffine.at<float>(2,1) + R_taffine.at<float>(2,2)*R_taffine.at<float>(2,2)));
            euler_taffine[2] = atan2(R_taffine.at<float>(1,0), R_taffine.at<float>(0,0));
            
            // 输出角度(转换为度)
            std::cout << "poses[" << i << "]旋转角度(度): roll=" << euler_pose[0]*180/M_PI 
                      << ", pitch=" << euler_pose[1]*180/M_PI
                      << ", yaw=" << euler_pose[2]*180/M_PI << std::endl;
                      
            std::cout << "taffine旋转角度(度): roll=" << euler_taffine[0]*180/M_PI 
                      << ", pitch=" << euler_taffine[1]*180/M_PI
                      << ", yaw=" << euler_taffine[2]*180/M_PI << std::endl;
        }
        volume_->integrate(dists_, poses[i], p.intr);
        if(false)
        { 
            volume_->raycast(poses[i], p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]); 
            for (int i = 1; i < LEVELS; ++i)
                resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
            cuda::waitAllDefaultStream();
            // 在当前相机视角下进行raycast
            renderImage(view_device_, 0);
            view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
            view_device_.download(view_host_.ptr<void>(), view_host_.step);
            cv::Mat rotated_view;
            cv::rotate(view_host_, rotated_view, cv::ROTATE_90_CLOCKWISE);
            cv::imshow("loopScene", rotated_view);
            cv::waitKey(10);
        }
    }
    // // 每30帧获取一次点云并转换到世界坐标系
    // std::vector<pcl::PointXYZRGB> all_points;
    // for(int i = 0; i < frame_count; i += 30) {
    //     // 生成随机颜色
    //     uint8_t r = rand() % 256;
    //     uint8_t g = rand() % 256; 
    //     uint8_t b = rand() % 256;

    //     cv::Mat depth = depth_imgs_[i];
        
    //     // 遍历深度图的每个像素
    //     for(int y = 0; y < depth.rows; y++) {
    //         for(int x = 0; x < depth.cols; x++) {
    //             float z = depth.at<ushort>(y,x) * 0.001f; // 转换为米
    //             if(z > 0) {
    //                 // 反投影到相机坐标系
    //                 float x_cam = (x - p.intr.cx) * z / p.intr.fx;
    //                 float y_cam = (y - p.intr.cy) * z / p.intr.fy;
                    
    //                 // 转换到世界坐标系
    //                 cv::Mat pt_cam = (cv::Mat_<float>(4,1) << x_cam, y_cam, z, 1);
    //                 cv::Mat pt_world = cv::Mat::zeros(4, 4, CV_32F);
    //                 // 将pt_world的左上3x3赋值为旋转矩阵
    //                 auto rot = poses[i].rotation();
    //                 for(int r = 0; r < 3; r++) {
    //                     for(int c = 0; c < 3; c++) {
    //                         pt_world.at<float>(r,c) = rot(r,c);
    //                     }
    //                 }
                    
    //                 // 将pt_world的右上3x1赋值为平移向量
    //                 pt_world(cv::Rect(3, 0, 1, 3)) = poses[i].translation();
    //                 cv::Mat pt_wd = pt_world * pt_cam;
    //                 // 添加到点云
    //                 pcl::PointXYZRGB point;
    //                 point.x = pt_wd.at<float>(0);
    //                 point.y = pt_wd.at<float>(1); 
    //                 point.z = pt_wd.at<float>(2);
    //                 point.r = r;
    //                 point.g = g;
    //                 point.b = b;
    //                 all_points.push_back(point);
    //             }
    //         }
    //     }
    // }

    // // 保存为PLY文件
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB>> cloud_points;
    // cloud_points = std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB>>(all_points.begin(), all_points.end());
    // cloud->points = cloud_points;
    // cloud->width = all_points.size();
    // cloud->height = 1;
    // pcl::io::savePLYFile("world_points.ply", *cloud);
    std::cout << "闭环优化完成,共处理 " << frame_count << " 帧" << std::endl;
    std::cout<<"图像总数量为: "<<depth_imgs_.size()<<"闭环重建帧数:"<<frame_count<<std::endl;
}

void kfusion::KinFu::toPly(cv::Mat& points, cv::Mat &normals, std::string spath)
{
    int ptnum = 0;
    std::vector<cv::Vec4f> pts;
    std::vector<cv::Vec4f> nls;
    pts.reserve(points.rows*points.cols);
    nls.reserve(points.rows*points.cols);
    for (size_t i = 0; i < points.rows; i++)
    {
        for (size_t j = 0; j < points.cols; j++)
        {
            cv::Vec4f pt = points.at<cv::Vec4f>(i,j);
            cv::Vec4f nl = points.at<cv::Vec4f>(i,j);
            if(!isnan(pt[0]))
            {
                // cv::Vec3f pt = cv::Vec3f(pts[0],pts[1],pts[2]);
                pts.push_back(pt);
                nls.push_back(nl);
                ptnum ++;
            }
        }
    }
    saveToPly(pts, nls, spath);
    std::cout<<"point number of final cloud: "<<ptnum<<std::endl;
}
void kfusion::KinFu::toPlyColor(cv::Mat& points, cv::Mat &normals, std::string spath, uint8_t r, uint8_t g, uint8_t b)
{
    int ptnum = 0;
    std::vector<cv::Vec4f> pts;
    std::vector<cv::Vec4f> nls;
    pts.reserve(points.rows*points.cols);
    nls.reserve(points.rows*points.cols);
    for (size_t i = 0; i < points.rows; i++)
    {
        for (size_t j = 0; j < points.cols; j++)
        {
            cv::Vec4f pt = points.at<cv::Vec4f>(i,j);
            cv::Vec4f nl = normals.at<cv::Vec4f>(i,j);
            if(!isnan(pt[0]) || isnan(nl[0]))
            {
                // cv::Vec3f pt = cv::Vec3f(pts[0],pts[1],pts[2]);
                pts.push_back(pt);
                nls.push_back(nl);
                ptnum ++;
            }
        }
    }
    saveToPlyColor(pts, nls, spath, r, g, b);
    std::cout<<"point number of final cloud: "<<ptnum<<std::endl;
}

void kfusion::KinFu::toPlyColorFilter(cv::Mat& points, cv::Mat &normals, std::string spath, uint8_t r, uint8_t g, uint8_t b)
{
    int ptnum = 0;
    std::vector<cv::Vec4f> pts;
    std::vector<cv::Vec4f> nls;
    pts.reserve(points.rows*points.cols);
    nls.reserve(points.rows*points.cols);
    double thres_groud = 0;
    double min_x = 10000;
    //groud 为x方向上的最大值
    for (size_t i = 0; i < points.rows; i++)
    {
        for (size_t j = 0; j < points.cols; j++)
        {
            cv::Vec4f pt = points.at<cv::Vec4f>(i,j);
            cv::Vec4f nl = normals.at<cv::Vec4f>(i,j);
            if(!isnan(pt[0]) || isnan(nl[0]))
            {
                // cv::Vec3f pt = cv::Vec3f(pts[0],pts[1],pts[2]);
                pts.push_back(pt);
                nls.push_back(nl);
                ptnum ++;
                if (pt[0]>thres_groud)
                {
                    thres_groud = pt[0];
                }
                if (pt[0]<min_x)
                {
                    min_x = pt[0];
                }
            }
        }
    }
    // 计算pt[0]小于thres_groud-0.3条件下所有点的pt[1]和pt[2]的均值
    double sum_y = 0, sum_z = 0;
    int count = 0;
    for (const auto& pt : pts) {
        if (pt[0] < thres_groud - 0.3) {
            sum_y += pt[1];
            sum_z += pt[2];
            count++;
        }
    }
    //判断count是否为0
    double avg_y = sum_y / count;
    double avg_z = sum_z / count;

    // 滤除不符合条件的点
    std::vector<cv::Vec4f> filtered_pts;
    std::vector<cv::Vec4f> filtered_nls;
    for (size_t i = 0; i < pts.size(); i++) {
        const auto& pt = pts[i];
        if (pt[0] < thres_groud - 0.3 || 
            (pt[0] >= thres_groud - 0.3 && pt[0] <= thres_groud && 
             std::sqrt(std::pow(pt[1] - avg_y, 2) + std::pow(pt[2] - avg_z, 2)) <= 0.4)) {
            filtered_pts.push_back(pt);
            filtered_nls.push_back(nls[i]);
        }
    }

    // 用过滤后的点替换原来的点
    pts = filtered_pts;
    nls = filtered_nls;
    // 重新计算thres_ground
    // 重新计算thres_ground
    thres_groud = -std::numeric_limits<double>::infinity();
    for (const auto& pt : pts) {
        if (pt[0] > thres_groud) {
            thres_groud = pt[0];
        }
    }
    
    ifstream sfile("./data/thick.txt");
    double thick;
    if(sfile.is_open())
    {
        sfile>>thick;
        sfile.close();
    }
    cout<<"max and min x: "<<thres_groud<<", "<<min_x<<","<<thick<<endl;
    thres_groud = thres_groud - thick;

    std::vector<cv::Vec4f> ptsf;
    std::vector<cv::Vec4f> nlsf;
    for (size_t i = 0; i < pts.size(); i++)
    {
        if(pts[i][0]>thres_groud)
        {
            continue;
        }
        ptsf.push_back(pts[i]);
        nlsf.push_back(nls[i]);
    }
    
    saveToPlyColor(ptsf, nlsf, spath, r, g, b);
    std::cout<<"point number of final cloud: "<<ptnum<<std::endl;
}
void kfusion::KinFu::toPlyVec3(cv::Mat& points, cv::Mat &normals, std::string spath)
{
    int ptnum = 0;
    std::vector<cv::Vec3f> pts;
    std::vector<cv::Vec3f> nls;
    pts.reserve(points.rows*points.cols);
    nls.reserve(points.rows*points.cols);
    for (size_t i = 0; i < points.rows; i++)
    {
        for (size_t j = 0; j < points.cols; j++)
        {
            cv::Vec3f pt = points.at<cv::Vec3f>(i,j);
            cv::Vec3f nl = points.at<cv::Vec3f>(i,j);
            if(!isnan(pt[0]))
            {
                // cv::Vec3f pt = cv::Vec3f(pts[0],pts[1],pts[2]);
                pts.push_back(pt);
                nls.push_back(nl);
                ptnum ++;
            }
        }
    }
    saveToPly(pts, nls, spath);
}

void kfusion::KinFu::toPlyVec3Color(cv::Mat& points, cv::Mat &normals, std::string spath, uint8_t r, uint8_t g, uint8_t b)
{
    int ptnum = 0;
    std::vector<cv::Vec3f> pts;
    std::vector<cv::Vec3f> nls;
    pts.reserve(points.rows*points.cols);
    nls.reserve(points.rows*points.cols);
    for (size_t i = 0; i < points.rows; i++)
    {
        for (size_t j = 0; j < points.cols; j++)
        {
            cv::Vec3f pt = points.at<cv::Vec3f>(i,j);
            cv::Vec3f nl = points.at<cv::Vec3f>(i,j);
            if(!isnan(pt[0]))
            {
                // cv::Vec3f pt = cv::Vec3f(pts[0],pts[1],pts[2]);
                pts.push_back(pt);
                nls.push_back(nl);
                ptnum ++;
            }
        }
    }
    saveToPlyColor(pts, nls, spath,r,g,b);
}
/**
 * \brief 将当前深度图和当前fusion的结果进行融合，计算动态warp
 * \param image
 * \param flag
 */
void kfusion::KinFu::dynamicfusion(cuda::Depth& depth, cuda::Cloud live_frame, cuda::Normals current_normals)
{
    warp_->setProject(params_.intr.fx,params_.intr.fy,params_.intr.cx,params_.intr.cy);
    warp_->image_width = params_.cols;
    warp_->image_height = params_.rows;
    //1. prepare some vars
    cuda::Cloud cloud;
    cuda::Normals normals;
    cloud.create(depth.rows(), depth.cols());
    normals.create(depth.rows(), depth.cols());
    cv::Mat cloud_host(depth.rows(), depth.cols(), CV_32FC4); //内存上当前fusion结果的点云
    auto camera_pose = poses_.back(); // 初始帧相机位姿在当前帧坐标系下的位姿，camera_pose * canonical = current
    // Aff_p = pose_
    // Aff_c = camera_pose_
    // xc = Aff_p * xv
    // xc = Aff_c * xl
    // camera_pose = camera_pose.inv(cv::DECOMP_SVD);
    auto inverse_pose = camera_pose.inv(cv::DECOMP_SVD); //transform to initial camera pose x_cano = inverse_pose * x_live
    // warp_->aff_inv = inverse_pose;
    // warp_->setWarpToLive(camera_pose);
    warp_->aff_inv = camera_pose;
    warp_->setWarpToLive(inverse_pose);
    // 投影到当前相机视角下的canonical空间点云,通过光线追踪得到当前相机pose下的cloud和normals
    tsdf().raycast(camera_pose, params_.intr, cloud, normals);
    cloud.download(cloud_host.ptr<Point>(), cloud_host.step);
    
    std::vector<Vec3f> canonical_cur(cloud_host.rows * cloud_host.cols); //canonical under current cam pose
    std::vector<Vec3f> canonical(cloud_host.rows * cloud_host.cols);     //canonical under initial cam pose

    //dynamicfusion的主要过程
    // 1. 基于raycast得到当前相机视角下的点云 canonical_cur
    // 2. 当前视角下的深度相机获取的点云 live
    // 3. 以上两者存在点对点的对应关系吗？canonical经过warp后的点云才和live存在点到点对应关系，但需要重新raycast才能得到
    // 4. 将对应点转换回到初始相机坐标系下，在同一个坐标系下估计warp的值
    // 5. 基于warp和pose，将live fuse到volume中去

    // 1. 
    for (int i = 0; i < cloud_host.rows; i++)
    {
        for (int j = 0; j < cloud_host.cols; j++) {
            auto point = cloud_host.at<Point>(i, j);
            canonical_cur[i * cloud_host.cols + j] = cv::Vec3f(point.x, point.y, point.z);
            //获取初始帧坐标系下的canonical点云坐标
            canonical[i * cloud_host.cols + j] = camera_pose * canonical_cur[i * cloud_host.cols + j];
        }
    }

    // 2 当前帧的cloud，当前camera pose下的当前点云
    live_frame.download(cloud_host.ptr<Point>(), cloud_host.step);
    std::vector<Vec3f> live(cloud_host.rows * cloud_host.cols);
    for (int i = 0; i < cloud_host.rows; i++)
    {
        for (int j = 0; j < cloud_host.cols; j++) {
            auto point = cloud_host.at<Point>(i, j);
            live[i * cloud_host.cols + j] = cv::Vec3f(point.x, point.y, point.z);
        }
    }

    //canonical normals, under current cam pose
    cv::Mat normal_host(cloud_host.rows, cloud_host.cols, CV_32FC4);
    normals.download(normal_host.ptr<Normal>(), normal_host.step);

    // canonical normals, 都在当前相机视角下进行优化
    std::vector<Vec3f> canonical_normals_cur(normal_host.rows * normal_host.cols); // canonical normal under current cam pose
    std::vector<Vec3f> canonical_normals(normal_host.rows * normal_host.cols);     // canonical normal under initial cam pose
    for (int i = 0; i < normal_host.rows; i++)
    {
        for (int j = 0; j < normal_host.cols; j++) {
            auto point = normal_host.at<Normal>(i, j);
            canonical_normals_cur[i * normal_host.cols + j] = cv::Vec3f(point.x, point.y, point.z);
            canonical_normals[i * normal_host.cols + j] = camera_pose.rotation() * cv::Vec3f(point.x, point.y, point.z);// TODO no translation include 
        }
    }

    std::vector<Vec3f> canonical_visible(canonical);
    //
    // saveToPly(canonical_cur, canonical_normals_cur, "canonical_beforwarp_cur.ply");
    // saveToPlyColor(canonical, canonical_normals, "canonical_beforwarp.ply",255,0,0);
    // warp_->warp(canonical, canonical_normals, false); // warp the vertices and affine to live frame
    // expand the nodes
    if(false) //当warp点云的时候出现距离node过远的点时扩展当前点云
    {
        cv::Mat frame_init;
        volume_->computePoints(frame_init);
        toPly(frame_init,frame_init, "expnode_pcl.ply");
        auto aff = volume_->getPose();
        aff = aff.inv();
        warp_->update_deform_node(frame_init, aff);
        //存扩展后的nodes
        auto nd = warp_->getNodesAsMat();
        toPlyVec3Color(nd,nd,"cur_nodes.ply",255,0,0);
    }
    saveToPlyColor(live, canonical_normals, "live.ply",0,255,0);
    // 3 get the correspondence between warped canonical and live frame
    //优化warpfield
    warp_->energy_data(canonical, canonical_normals, live, canonical_normals);
    // optimiser_->optimiseWarpData(canonical, canonical_normals, live, canonical_normals); // Normals are not used yet so just send in same data
    std::vector<Vec3f> warp_nodes;
    warp_->getWarpedNode(warp_nodes);
    saveToPlyColor(warp_nodes, warp_nodes, "warp_nodes_live.ply",0,255,0);
    std::cout<<"try to warp"<<std::endl;
    warp_->setWarpToLive(Affine3f::Identity());
    warp_->warp(canonical, canonical_normals);

    saveToPlyColor(canonical, canonical_normals, "aft_opt.ply",0,0,255);
//    //ScopeTime time("fusion");
    std::cout<<"dynamic surface fusion"<<std::endl;
    //!!!!!!!
   
    tsdf().surface_fusion(getWarp(), canonical, canonical_visible, depth, camera_pose, params_.intr);
    std::cout<<"download depth cloud"<<std::endl;
    cv::Mat depth_cloud(depth.rows(),depth.cols(), CV_16U);
    depth.download(depth_cloud.ptr<void>(), depth_cloud.step);
    cv::Mat display;
    depth_cloud.convertTo(display, CV_8U, 255.0/4000);
    std::cout<<"show depth diff"<<std::endl;
    cv::imshow("Depth diff", display);
    // volume_->compute_points();
    // cv::Mat points, normals_t;
    // std::cout<<"get points"<<std::endl;
    // volume_->get_points(points);
    // std::cout<<"compute normals"<<std::endl;
    // volume_->compute_normals();
    std::cout<<"END of dynamic fusion"<<std::endl;
}
void kfusion::KinFu::renderImage(cuda::Image& image, int flag)
{
    const KinFuParams& p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);

#if defined USE_DEPTH
    #define PASS1 prev_.depth_pyr
#else
    #define PASS1 prev_.points_pyr
#endif
    if (flag < 1 || flag > 3)
        cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cuda::renderTangentColors(prev_.normals_pyr[0], image);
    else /* if (flag == 3) */
    {
        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());
        cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, i1);
        cuda::renderTangentColors(prev_.normals_pyr[0], i2);
    }
#undef PASS1
}


void kfusion::KinFu::renderImage(cuda::Image& image, const Affine3f& pose, int flag)
{
    const KinFuParams& p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
    depths_.create(p.rows, p.cols);
    normals_.create(p.rows, p.cols);
    points_.create(p.rows, p.cols);

#if defined USE_DEPTH
    #define PASS1 depths_
#else
    #define PASS1 points_
#endif

    volume_->raycast(pose, p.intr, PASS1, normals_);

    if (flag < 1 || flag > 3)
        cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cuda::renderTangentColors(normals_, image);
    else /* if (flag == 3) */
    {
        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, i1);
        cuda::renderTangentColors(normals_, i2);
    }
#undef PASS1
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//namespace pcl
//{
//    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
//    {
//        Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
//        Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

//        double rx = R(2, 1) - R(1, 2);
//        double ry = R(0, 2) - R(2, 0);
//        double rz = R(1, 0) - R(0, 1);

//        double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
//        double c = (R.trace() - 1) * 0.5;
//        c = c > 1. ? 1. : c < -1. ? -1. : c;

//        double theta = acos(c);

//        if( s < 1e-5 )
//        {
//            double t;

//            if( c > 0 )
//                rx = ry = rz = 0;
//            else
//            {
//                t = (R(0, 0) + 1)*0.5;
//                rx = sqrt( std::max(t, 0.0) );
//                t = (R(1, 1) + 1)*0.5;
//                ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
//                t = (R(2, 2) + 1)*0.5;
//                rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

//                if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
//                    rz = -rz;
//                theta /= sqrt(rx*rx + ry*ry + rz*rz);
//                rx *= theta;
//                ry *= theta;
//                rz *= theta;
//            }
//        }
//        else
//        {
//            double vth = 1/(2*s);
//            vth *= theta;
//            rx *= vth; ry *= vth; rz *= vth;
//        }
//        return Eigen::Vector3d(rx, ry, rz).cast<float>();
//    }
//}


