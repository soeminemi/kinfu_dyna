#pragma once

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace kfusion {

class CircularMotionConstraint {
public:
    CircularMotionConstraint(const cv::Vec3f& center = cv::Vec3f(0,0,0), float radius = 1.0f)
        : center_(center), radius_(radius) {}

    // 设置圆心和半径
    void setCenter(const cv::Vec3f& center) { center_ = center; }
    void setRadius(float radius) { radius_ = radius; }

    // 获取圆心和半径
    cv::Vec3f getCenter() const { return center_; }
    float getRadius() const { return radius_; }

    // 计算位姿到圆周运动的约束误差
    float computeError(const cv::Affine3f& pose) const {
        cv::Vec3f position(pose.translation());
        // 计算当前位置到圆心的距离与半径之差
        float dist = cv::norm(position - center_);
        return (dist - radius_) * (dist - radius_);
    }

    // 计算位姿对约束的雅可比矩阵
    cv::Mat computeJacobian(const cv::Affine3f& pose) const {
        cv::Vec3f position(pose.translation());
        cv::Vec3f diff = position - center_;
        float dist = cv::norm(diff);
        
        // 计算对位置的偏导数
        cv::Mat J = cv::Mat::zeros(1, 6, CV_32F);
        if(dist > 1e-6) {
            float factor = 2.0f * (dist - radius_) / dist;
            J.at<float>(0,3) = factor * diff[0];
            J.at<float>(0,4) = factor * diff[1];
            J.at<float>(0,5) = factor * diff[2];
        }
        return J;
    }

    // 从轨迹估计圆心和半径
    void estimateFromTrajectory(const std::vector<cv::Affine3f>& poses) {
        if (poses.empty()) return;

        // 提取所有相机位置
        std::vector<cv::Point3f> points;
        for (const auto& pose : poses) {
            cv::Vec3f pos = pose.translation();
            points.push_back(cv::Point3f(pos[0], pos[1], pos[2]));
        }

        // 找到主平面 (使用PCA)
        cv::Mat points_mat(points.size(), 3, CV_32F);
        for (size_t i = 0; i < points.size(); i++) {
            points_mat.at<float>(i, 0) = points[i].x;
            points_mat.at<float>(i, 1) = points[i].y;
            points_mat.at<float>(i, 2) = points[i].z;
        }

        cv::PCA pca(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);
        cv::Mat eigenvalues = pca.eigenvalues;
        cv::Mat eigenvectors = pca.eigenvectors;

        // 将点投影到主平面上
        cv::Mat mean = pca.mean;
        std::vector<cv::Point2f> projected_points;
        cv::Vec3f normal(eigenvectors.at<float>(2, 0),
                        eigenvectors.at<float>(2, 1),
                        eigenvectors.at<float>(2, 2));

        // 定义平面的基向量
        cv::Vec3f basis1(eigenvectors.at<float>(0, 0),
                       eigenvectors.at<float>(0, 1),
                       eigenvectors.at<float>(0, 2));
        cv::Vec3f basis2(eigenvectors.at<float>(1, 0),
                       eigenvectors.at<float>(1, 1),
                       eigenvectors.at<float>(1, 2));

        for (const auto& point : points) {
            cv::Vec3f p(point.x - mean.at<float>(0),
                       point.y - mean.at<float>(1),
                       point.z - mean.at<float>(2));
            
            float x = p.dot(basis1);
            float y = p.dot(basis2);
            projected_points.push_back(cv::Point2f(x, y));
        }

        // 拟合圆
        float sum_x = 0, sum_y = 0;
        float sum_xx = 0, sum_yy = 0, sum_xy = 0;
        float sum_xxx = 0, sum_yyy = 0, sum_xxy = 0, sum_xyy = 0;
        
        for (const auto& p : projected_points) {
            float x = p.x, y = p.y;
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
            sum_xxx += x * x * x;
            sum_yyy += y * y * y;
            sum_xxy += x * x * y;
            sum_xyy += x * y * y;
        }

        float n = static_cast<float>(projected_points.size());
        cv::Mat A = (cv::Mat_<float>(3, 3) <<
            sum_xx, sum_xy, sum_x,
            sum_xy, sum_yy, sum_y,
            sum_x,  sum_y,  n);
        
        cv::Mat b = (cv::Mat_<float>(3, 1) <<
            -sum_xxx - sum_xyy,
            -sum_xxy - sum_yyy,
            -sum_xx - sum_yy);

        cv::Mat solution;
        cv::solve(A, b, solution);

        float a = solution.at<float>(0);
        float b_val = solution.at<float>(1);
        float c = solution.at<float>(2);

        // 计算圆心和半径
        float center_x = -a / 2;
        float center_y = -b_val / 2;
        radius_ = std::sqrt(center_x * center_x + center_y * center_y - c);
        // radius_ = radius_; // 降低圆周半径的误差
        // 将圆心从2D投影空间转回3D空间
        cv::Vec3f mean_vec(mean.at<float>(0), mean.at<float>(1), mean.at<float>(2));
        cv::Vec3f center_3d = mean_vec + 
            basis1 * center_x +
            basis2 * center_y;

        center_ = center_3d;
    }

private:
    cv::Vec3f center_;  // 圆心位置
    float radius_;      // 圆周半径
};

} // namespace kfusion
