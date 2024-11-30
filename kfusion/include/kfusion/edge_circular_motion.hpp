#pragma once

#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include "circular_motion_constraint.hpp"

namespace g2o {

class EdgeCircularMotion : public BaseBinaryEdge<3, kfusion::CircularMotionConstraint, VertexSE3, VertexSE3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeCircularMotion() : BaseBinaryEdge<3, kfusion::CircularMotionConstraint, VertexSE3, VertexSE3>() {}

    void computeError() override {
        const VertexSE3* v1 = static_cast<const VertexSE3*>(_vertices[0]);
        const VertexSE3* v2 = static_cast<const VertexSE3*>(_vertices[1]);

        // 获取两个位姿的位置
        Eigen::Vector3d p1 = v1->estimate().translation();
        Eigen::Vector3d p2 = v2->estimate().translation();

        // 获取圆心和半径
        cv::Vec3f center = _measurement.getCenter();
        float radius = _measurement.getRadius();

        // 计算两个点到圆心的距离与半径的差
        Eigen::Vector3d center_eigen(center[0], center[1], center[2]);
        double dist1 = (p1 - center_eigen).norm() - radius;
        double dist2 = (p2 - center_eigen).norm() - radius;

        // 计算两个点的向量是否垂直于圆心向量
        Eigen::Vector3d v1_to_center = center_eigen - p1;
        Eigen::Vector3d v2_to_center = center_eigen - p2;
        Eigen::Vector3d v1_to_v2 = p2 - p1;
        
        double dot1 = v1_to_center.normalized().dot(v1_to_v2.normalized());
        double dot2 = v2_to_center.normalized().dot(v1_to_v2.normalized());

        _error[0] = dist1;  // 第一个点到圆的距离误差
        _error[1] = dist2;  // 第二个点到圆的距离误差
        _error[2] = (std::abs(dot1) + std::abs(dot2)) / 2.0;  // 切向误差
    }

    virtual bool read(std::istream& is) override {
        return true;
    }

    virtual bool write(std::ostream& os) const override {
        return true;
    }
};

} // namespace g2o
