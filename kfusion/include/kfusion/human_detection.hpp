#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>

namespace kfusion {
    class HumanDetector {
    public:
        HumanDetector();
        
        // 主要的人体检测函数
        void detectHumanBody(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
        
        // 参数设置函数
        void setVoxelSize(float size);
        void setClusterTolerance(float tolerance);
        void setMinClusterSize(int size);
        void setMaxClusterSize(int size);
        
    private:
        // 参数
        float voxel_size_;
        float cluster_tolerance_;
        int min_cluster_size_;
        int max_cluster_size_;
        
        // 内部处理函数
        void downsampleCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, 
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered);
        void removeOutliers(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
        void removePlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
        bool isHumanCluster(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster);
    };
}
