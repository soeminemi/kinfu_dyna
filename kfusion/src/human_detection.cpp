#include "kfusion/human_detection.hpp"

namespace kfusion {

HumanDetector::HumanDetector()
    : voxel_size_(0.01f)  // 1cm体素
    , cluster_tolerance_(0.02f)  // 2cm
    , min_cluster_size_(100)
    , max_cluster_size_(25000)
{
}

void HumanDetector::setVoxelSize(float size) { voxel_size_ = size; }
void HumanDetector::setClusterTolerance(float tolerance) { cluster_tolerance_ = tolerance; }
void HumanDetector::setMinClusterSize(int size) { min_cluster_size_ = size; }
void HumanDetector::setMaxClusterSize(int size) { max_cluster_size_ = size; }

void HumanDetector::detectHumanBody(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    // 1. 降采样
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    downsampleCloud(cloud, cloud_filtered);

    // 2. 去除离群点
    removeOutliers(cloud_filtered);

    // 3. 去除平面（地面）
    removePlane(cloud_filtered);

    // 4. 欧式聚类
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    // 5. 分析每个聚类
    for (const auto& cluster : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (const auto& idx : cluster.indices) {
            cluster_cloud->points.push_back(cloud_filtered->points[idx]);
        }

        if (isHumanCluster(cluster_cloud)) {
            // 标记为人体（红色）
            for (auto& point : cluster_cloud->points) {
                point.r = 255;
                point.g = 0;
                point.b = 0;
            }
            // 更新原始点云中对应的点
            for (const auto& idx : cluster.indices) {
                cloud->points[idx].r = 255;
                cloud->points[idx].g = 0;
                cloud->points[idx].b = 0;
            }
        }
    }
}

void HumanDetector::downsampleCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered) {
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    vg.filter(*cloud_filtered);
}

void HumanDetector::removeOutliers(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud);
}

void HumanDetector::removePlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() > 0) {
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud);
    }
}

bool HumanDetector::isHumanCluster(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster) {
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    
    float height = max_pt.y - min_pt.y;
    float width = max_pt.x - min_pt.x;
    float depth = max_pt.z - min_pt.z;
    
    // 人体尺寸检查（单位：米）
    bool size_check = (height > 1.5 && height < 2.0 &&  // 身高范围
                      width > 0.4 && width < 0.8 &&     // 宽度范围
                      depth > 0.2 && depth < 0.5);      // 深度范围
    
    // 点云密度检查
    float volume = height * width * depth;
    float point_density = cluster->size() / volume;
    bool density_check = (point_density > 1000);  // 每立方米至少1000个点
    
    return size_check && density_check;
}

}
