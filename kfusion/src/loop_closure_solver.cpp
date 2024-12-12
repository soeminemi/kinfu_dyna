#include "kfusion/loop_closure_solver.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <iostream>
#include <random>
using namespace std;

namespace kfusion {
LoopClosureSolver::LoopClosureSolver()
    : max_correspondence_distance_(0.05f)
    , transformation_epsilon_(1e-8f)
    , euclidean_fitness_epsilon_(1e-6f)
    , max_iterations_(50)
{
}

void LoopClosureSolver::setupICP(pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>& icp) const {
    icp.setMaxCorrespondenceDistance(max_correspondence_distance_);
    icp.setTransformationEpsilon(transformation_epsilon_);
    icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon_);
    icp.setMaximumIterations(max_iterations_);
    icp.setUseSymmetricObjective(true);
    
    // 设置点到平面距离度量
    pcl::registration::TransformationEstimationPointToPlane<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr trans_est(
        new pcl::registration::TransformationEstimationPointToPlane<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
    icp.setTransformationEstimation(trans_est);
}

cv::Affine3f LoopClosureSolver::estimateRelativePose(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& source_cloud,
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& target_cloud,
    const cv::Affine3f& initial_guess,
    std::vector<int>& source_indices,
    std::vector<int>& target_indices
) {
    pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
    setupICP(icp);
    
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);

    // 准备PCL格式的初始猜测矩阵
    Eigen::Matrix4f init_guess_mat = Eigen::Matrix4f::Identity();
    const cv::Matx44f& cv_mat = initial_guess.matrix;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            init_guess_mat(i, j) = cv_mat(i, j);

    // 执行配准
    cout<<"try to align with normal icp"<<endl;
    pcl::PointCloud<pcl::PointXYZRGBNormal> aligned;
    icp.align(aligned, init_guess_mat);
    cout<<"aligned"<<endl;
    // // 按照init_guess_mat转换source点云
    // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed_source(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    // pcl::transformPointCloud(*source_cloud, *transformed_source, init_guess_mat);

    // // 保存target和转换后的source为ply格式
    // pcl::io::savePLYFile("target_cloud.ply", *target_cloud);
    pcl::io::savePLYFile("aligned_target_cloud_"+to_string(idx_)+".ply", *target_cloud);
    pcl::io::savePLYFile("aligned_cloud_"+to_string(idx_)+".ply", aligned);

    if (!icp.hasConverged()) {
        std::cout << "ICP did not converge!" << std::endl;
        return initial_guess;
    }
    else {
        std::cout << "ICP converged successfully!" << std::endl;
        // Get correspondences for converged result
        pcl::Correspondences correspondences = *icp.correspondences_;
        source_indices.clear();
        target_indices.clear();
        source_indices.reserve(correspondences.size());
        target_indices.reserve(correspondences.size());
        for (const auto& corr : correspondences) {
            source_indices.push_back(corr.index_query);
            target_indices.push_back(corr.index_match);
        }
    }

    // 将PCL变换矩阵转换回OpenCV格式
    Eigen::Matrix4f final_transform = icp.getFinalTransformation();
    cv::Mat final_cv_mat(4, 4, CV_32F);
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            final_cv_mat.at<float>(i,j) = final_transform(i,j);

    return cv::Affine3f(final_cv_mat);
}

std::vector<cv::Affine3f> LoopClosureSolver::optimizePoses(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& point_clouds,
    const std::vector<cv::Affine3f>& initial_poses,
    const std::vector<std::pair<int, int>>& loop_pairs
) {
    // 计算闭环帧之间的相对位姿,并获取对应点对
    std::vector<cv::Affine3f> relative_poses;
    std::vector<int> source_indices, target_indices;
    std::vector<std::vector<std::pair<int, int>>> loop_corres;
    cout<<"基于icp进行相对位姿估计及对应点获取"<<endl;
    int idx = 1;
    for (const auto& pair : loop_pairs) {
        cout<<"估计ICP 从: "<<pair.first<<" 到 "<<pair.second<<endl;
        cv::Affine3f initial_guess = initial_poses[pair.second].inv() * initial_poses[pair.first];
        setIndex(idx);
        idx++;
        cv::Affine3f relative_pose = estimateRelativePose(
            point_clouds[pair.first],
            point_clouds[pair.second],
            initial_guess,
            source_indices,
            target_indices
        );
        std::vector<std::pair<int, int>> pcorres;
        pcorres.clear();
        pcorres.reserve(source_indices.size());
        for (size_t i = 0; i < source_indices.size(); i++)
        {
            std::pair<int, int> corr = std::make_pair(source_indices[i], target_indices[i]);
            pcorres.push_back(corr);
        }
        loop_corres.push_back(pcorres);
        relative_poses.push_back(relative_pose);  
    }

    // 使用BA优化位姿
    return optimizePosesBA(point_clouds, initial_poses, loop_pairs, loop_corres);
}

// 定义BA优化的代价函数
struct PoseOptimizationError {
    PoseOptimizationError(const Eigen::Vector3d& source_point,
                         const Eigen::Vector3d& target_point,
                         const Eigen::Vector3d& source_normal,
                         const Eigen::Vector3d& target_normal,
                         const double weight = 1.0)
        : source_point_(source_point), target_point_(target_point),
          source_normal_(source_normal), target_normal_(target_normal),
          weight_(weight) {}

    template <typename T>
    bool operator()(const T* const pose_source,    // source pose [qw,qx,qy,qz,tx,ty,tz]
                   const T* const pose_target,     // target pose [qw,qx,qy,qz,tx,ty,tz]
                   T* residuals) const {
        
        // 将source点转换为T类型
        T pt_source[3] = {T(source_point_[0]), 
                  T(source_point_[1]), 
                  T(source_point_[2])};

        // source pose的旋转和平移
        T source_q[4] = {pose_source[0], pose_source[1], 
                        pose_source[2], pose_source[3]};
        T source_t[3] = {pose_source[4], pose_source[5], pose_source[6]};

        // 将target点转换为T类型
        T pt_target[3] = {T(target_point_[0]), 
                  T(target_point_[1]), 
                  T(target_point_[2])};
        
        // target pose的旋转和平移
        T target_q[4] = {pose_target[0], pose_target[1], 
                        pose_target[2], pose_target[3]};
        T target_t[3] = {pose_target[4], pose_target[5], pose_target[6]};

        // 将点从世界坐标系转换到相机坐标系
        T p_source[3], p_target[3];
        ceres::QuaternionRotatePoint(source_q, pt_source, p_source);
        p_source[0] += source_t[0];
        p_source[1] += source_t[1];
        p_source[2] += source_t[2];

        ceres::QuaternionRotatePoint(target_q, pt_target, p_target);
        p_target[0] += target_t[0];
        p_target[1] += target_t[1];
        p_target[2] += target_t[2];

        // 转换法向量
        T n_source[3] = {T(source_normal_[0]), 
                        T(source_normal_[1]), 
                        T(source_normal_[2])};
        T n_target[3] = {T(target_normal_[0]), 
                        T(target_normal_[1]), 
                        T(target_normal_[2])};
        
        T rotated_n_source[3], rotated_n_target[3];
        ceres::QuaternionRotatePoint(source_q, n_source, rotated_n_source);
        ceres::QuaternionRotatePoint(target_q, n_target, rotated_n_target);

        // 计算法向量的点积，用于判断法向量是否接近
        T normal_similarity = rotated_n_source[0] * rotated_n_target[0] +
                            rotated_n_source[1] * rotated_n_target[1] +
                            rotated_n_source[2] * rotated_n_target[2];
        
        // 计算点到面的距离
        T point_diff[3] = {p_source[0] - p_target[0],
                          p_source[1] - p_target[1],
                          p_source[2] - p_target[2]};

        // 点到面的距离 = (p1 - p2) · n2
        T point_to_plane_dist = point_diff[0] * rotated_n_target[0] +
                               point_diff[1] * rotated_n_target[1] +
                               point_diff[2] * rotated_n_target[2];

        // 最终残差 = 权重 * 点到面距离
        residuals[0] = T(weight_) * point_to_plane_dist;

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& source_point,
                                     const Eigen::Vector3d& target_point,
                                     const Eigen::Vector3d& source_normal,
                                     const Eigen::Vector3d& target_normal,
                                     const double weight) {
        return new ceres::AutoDiffCostFunction<PoseOptimizationError, 1, 7, 7>(
            new PoseOptimizationError(source_point, target_point, 
                                    source_normal, target_normal, weight));
    }

private:
    const Eigen::Vector3d source_point_;
    const Eigen::Vector3d target_point_;
    const Eigen::Vector3d source_normal_;
    const Eigen::Vector3d target_normal_;
    const double weight_;
};

std::vector<cv::Affine3f> LoopClosureSolver::optimizePosesBA(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& point_clouds,
    const std::vector<cv::Affine3f>& initial_poses,
    const std::vector<std::pair<int, int>>& loop_pairs,
    const std::vector<std::vector<std::pair<int, int> > > &loop_corres
) {
    // // 计算所有点云按照给定的initial_poses变换后的点云的所有点的中心点
    // Eigen::Vector3d center(0, 0, 0);
    // int total_points = 0;

    // for (size_t i = 0; i < point_clouds.size(); ++i) {
    //     const auto& cloud = point_clouds[i];
    //     const auto& pose = initial_poses[i];

    //     for (const auto& point : *cloud) {
    //         cv::Point3f pt(point.x, point.y, point.z);
    //         cv::Point3f transformed_pt = pose * pt;
    //         center += Eigen::Vector3d(transformed_pt.x, transformed_pt.y, transformed_pt.z);
    //         ++total_points;
    //     }
    // }

    // if (total_points > 0) {
    //     center /= total_points;
    // }

    // 创建Ceres问题
    ceres::Problem problem;
    
    // 为每个位姿创建参数块
    std::vector<double*> pose_params;
    for (const auto& pose : initial_poses) {
        // 为每个位姿分配内存：4个参数用于四元数，3个用于平移
        double* params = new double[7];
        
        // 从OpenCV的Affine3f转换到四元数和平移向量
        cv::Matx33f R = pose.rotation();
        cv::Vec3f t = pose.translation();
        
        // 转换旋转矩阵为四元数
        Eigen::Matrix3d eigen_R;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                eigen_R(i,j) = R(i,j);
        
        Eigen::Quaterniond q(eigen_R);
        params[0] = q.w();
        params[1] = q.x();
        params[2] = q.y();
        params[3] = q.z();
        
        // 设置平移向量
        params[4] = t[0];
        params[5] = t[1];
        params[6] = t[2];
        
        pose_params.push_back(params);

        // 添加参数块到问题中
        problem.AddParameterBlock(params, 7);     // pose 参数块
        // problem.AddParameterBlock(params + 4, 3); // 平移向量参数块
        
        // 第一帧位姿固定
        if (pose_params.size() == 1) {
            problem.SetParameterBlockConstant(params);     // 固定四元数
            // problem.SetParameterBlockConstant(params + 4); // 固定平移向量
        }
    }
    float max_x = -std::numeric_limits<float>::max();
    for (const auto& point_cloud : point_clouds) {
        for (const auto& point : point_cloud->points) {
            max_x = std::max(max_x, point.x);
        }
    }
    std::cout << "max x: " << max_x << std::endl;
    
    // 添加闭环约束
    double weight = 1.0;
    const size_t MAX_CORRESPONDENCES = 10000;
    float ct_idx = loop_pairs.size()/2;
    for (size_t i = 0; i < loop_pairs.size(); ++i) {
        // 设置权重，第一个和最后一个回环约束权重更大
        // weight = 1.0;
        // if(i == 0 || i == loop_pairs.size() - 1) {
        //     weight = 10.f;
        // }
        weight = fabs(i-ct_idx) * 10/ct_idx;
        if(weight < 2)
            weight = 2;
        auto source_idx = loop_pairs[i].first;
        auto target_idx = loop_pairs[i].second;
        
        std::vector<size_t> indices(loop_corres[i].size());
        std::iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, ..., n-1
        
        if (loop_corres[i].size() > MAX_CORRESPONDENCES) {
            // Random shuffle and take first MAX_CORRESPONDENCES elements
            std::random_shuffle(indices.begin(), indices.end());
            indices.resize(MAX_CORRESPONDENCES);
            std::cout << "Using " << MAX_CORRESPONDENCES << " points out of " << loop_corres[i].size() 
                      << " for loop closure " << i << " (weight: " << weight << ")" << std::endl;
        } else {
            std::cout << "Using all " << loop_corres[i].size() << " points for loop closure " << i 
                      << " (weight: " << weight << ")" << std::endl;
        }
        
        for (size_t idx : indices) {
            auto pt_cpidx = loop_corres[i][idx];
            auto source_indice = pt_cpidx.first;
            auto target_indice = pt_cpidx.second;
            const auto& source_point = point_clouds[source_idx]->points[source_indice];
            const auto& target_point = point_clouds[target_idx]->points[target_indice];
            if (source_point.x > max_x - 0.2) continue;
            if (target_point.x > max_x - 0.2) continue;
            Eigen::Vector3d source_eigen(source_point.x, source_point.y, source_point.z);
            Eigen::Vector3d target_eigen(target_point.x, target_point.y, target_point.z);
            Eigen::Vector3d source_normal_eigen(source_point.normal_x, source_point.normal_y, source_point.normal_z);
            Eigen::Vector3d target_normal_eigen(target_point.normal_x, target_point.normal_y, target_point.normal_z);
            
            // 添加代价函数
            ceres::CostFunction* cost_function = 
                PoseOptimizationError::Create(source_eigen, target_eigen, source_normal_eigen, target_normal_eigen, weight);
                
            problem.AddResidualBlock(cost_function,
                                   new ceres::HuberLoss(0.01),   // 使用Huber核函数处理异常值
                                   pose_params[source_idx],     // 旋转平移参数
                                   pose_params[target_idx]      // 目标旋转平移参数
                                   );
        }
    }
    
    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    cout<<"开始优化"<<endl;
    // 求解优化问题
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    
    // 将优化后的结果转换回OpenCV格式
    std::vector<cv::Affine3f> optimized_poses;
    for (const auto& params : pose_params) {
        // 从四元数和平移向量构建变换矩阵
        Eigen::Quaterniond q(params[0], params[1], params[2], params[3]);
        Eigen::Vector3d t(params[4], params[5], params[6]);
        
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3,3>(0,0) = q.normalized().toRotationMatrix();
        transform.block<3,1>(0,3) = t;
        
        // 转换为OpenCV格式
        cv::Mat transform_cv(4, 4, CV_32F);
        for(int i = 0; i < 4; i++)
            for(int j = 0; j < 4; j++)
                transform_cv.at<float>(i,j) = static_cast<float>(transform(i,j));
        
        optimized_poses.push_back(cv::Affine3f(transform_cv));
        
        // 释放内存
        delete[] params;
    }
    
    return optimized_poses;
}

} // namespace kfusion
