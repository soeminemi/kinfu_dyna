#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "kfusion/types.hpp"

namespace kfusion {

// Cost function for pose graph optimization
class PoseGraphError {
public:
    PoseGraphError(const Eigen::Matrix4d& relative_pose)
        : relative_pose_(relative_pose) {}

    template <typename T>
    bool operator()(const T* const p1, const T* const q1,
                   const T* const p2, const T* const q2,
                   T* residuals) const {
        // Convert quaternion to rotation matrix
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t1(p1);
        Eigen::Map<const Eigen::Quaternion<T>> q1_map(q1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t2(p2);
        Eigen::Map<const Eigen::Quaternion<T>> q2_map(q2);

        // Compute the relative transformation
        Eigen::Quaternion<T> q_rel = q1_map.conjugate() * q2_map;
        Eigen::Matrix<T, 3, 1> t_rel = q1_map.conjugate() * (t2 - t1);

        // Convert relative pose to matrix form
        Eigen::Matrix<T, 4, 4> pred_rel = Eigen::Matrix<T, 4, 4>::Identity();
        pred_rel.block(0, 0, 3, 3) = q_rel.toRotationMatrix();
        pred_rel.block(0, 3, 3, 1) = t_rel;

        // Compute error
        Eigen::Matrix<T, 4, 4> error = pred_rel.inverse() * relative_pose_.cast<T>();
        
        // Extract rotation and translation parts of the error
        Eigen::Matrix<T, 3, 3> R_error = error.block(0, 0, 3, 3);
        Eigen::Matrix<T, 3, 1> t_error = error.block(0, 3, 3, 1);

        // Convert rotation error to angle-axis
        T angle_axis[3];
        ceres::RotationMatrixToAngleAxis(R_error.data(), angle_axis);

        // Fill residuals
        residuals[0] = angle_axis[0];
        residuals[1] = angle_axis[1];
        residuals[2] = angle_axis[2];
        residuals[3] = t_error[0];
        residuals[4] = t_error[1];
        residuals[5] = t_error[2];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix4d& relative_pose) {
        return new ceres::AutoDiffCostFunction<PoseGraphError, 6, 3, 4, 3, 4>(
            new PoseGraphError(relative_pose));
    }

private:
    const Eigen::Matrix4d relative_pose_;
};

class CeresGraph {
public:
    static void optimizePoseGraph(std::vector<Affine3f>& poses,
                                const std::vector<int>& loop_frame_idx,
                                const std::vector<Affine3f>& loop_poses);
};

} // namespace kfusion
