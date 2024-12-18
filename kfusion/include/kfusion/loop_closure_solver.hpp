#pragma once

#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/core/affine.hpp>
#include <pcl/point_types.h>
#include "internal.hpp"
#include <string>

namespace kfusion {
class LoopClosureSolver {
public:
    LoopClosureSolver();
    ~LoopClosureSolver() = default;

    // 主要接口函数：优化位姿
    std::vector<cv::Affine3f> optimizePoses(
        const std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& point_clouds,
        const std::vector<cv::Affine3f>& initial_poses,
        const std::vector<std::pair<int, int>>& loop_pairs
    );
    
private:
    // ICP参数设置
    void setupICP(pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>& icp) const;
    
    // 计算两个点云之间的相对位姿，并获取对应点序号
    cv::Affine3f estimateRelativePose(
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& source_cloud,
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& target_cloud,
        const cv::Affine3f& initial_guess,
        std::vector<int>& source_indices,
        std::vector<int>& target_indices
    );

    // BA优化位姿
    std::vector<cv::Affine3f> optimizePosesBA(
        const std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& point_clouds,
        const std::vector<cv::Affine3f>& initial_poses,
        const std::vector<std::pair<int, int>>& loop_pairs,
        const std::vector<std::vector<std::pair<int, int> > >& loop_corres
    );

    // ICP相关参数
    float max_correspondence_distance_;
    float transformation_epsilon_;
    float euclidean_fitness_epsilon_;
    int max_iterations_;
    std::string icp_name_;
};

} // namespace kfusion
