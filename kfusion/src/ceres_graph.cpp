#include "kfusion/ceres_graph.hpp"

namespace kfusion {

// Cost function for circular motion constraint
class CircularMotionError {
public:
    CircularMotionError(const Eigen::Vector3d& center, double radius)
        : center_(center), radius_(radius) {}

    template <typename T>
    bool operator()(const T* const translation,
                   const T* const rotation,
                   T* residuals) const {
        // Convert center to template type
        const T center_t[3] = {
            T(center_[0]), T(center_[1]), T(center_[2])
        };
        
        // Calculate distance error (should be equal to radius)
        T dx = translation[0] - center_t[0];
        T dy = translation[1] - center_t[1];
        T dz = translation[2] - center_t[2];
        T distance = ceres::sqrt(dx * dx + dy * dy + dz * dz);
        
        // Residual is the difference from desired radius
        residuals[0] = (distance - T(radius_)) * T(10.0); // Scale factor for better convergence
        
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& center, double radius) {
        return new ceres::AutoDiffCostFunction<CircularMotionError, 1, 3, 4>(
            new CircularMotionError(center, radius));
    }

private:
    Eigen::Vector3d center_;
    double radius_;
};

void CeresGraph::optimizePoseGraph(std::vector<Affine3f>& poses,
                                 const std::vector<int>& loop_frame_idx,
                                 const std::vector<std::pair<int, int>>& loop_pairs,
                                 const std::vector<Affine3f>& loop_poses,
                                 const cv::Vec3f& center,
                                 const float radius) {
    if (poses.empty() || loop_frame_idx.empty() || loop_poses.empty())
        return;

    // Problem setup
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization* quaternion_parameterization = new ceres::EigenQuaternionParameterization();

    // Parameters for each pose (translation and rotation as quaternion)
    std::vector<Eigen::Vector3d> translations(poses.size());
    std::vector<Eigen::Quaterniond> rotations(poses.size());

    // Convert poses to Eigen format
    for (size_t i = 0; i < poses.size(); ++i) {
        cv::Mat R;
        cv::Mat(poses[i].rotation()).convertTo(R, CV_64F);
        Eigen::Matrix3d rotation;
        cv2eigen(R, rotation);
        
        rotations[i] = Eigen::Quaterniond(rotation);
        translations[i] = Eigen::Vector3d(
            poses[i].translation()[0],
            poses[i].translation()[1],
            poses[i].translation()[2]);
    }

    // Add sequential constraints
    for (size_t i = 1; i < poses.size(); ++i) {
        Eigen::Matrix4d relative_pose = Eigen::Matrix4d::Identity();
        
        // Compute relative transformation between consecutive frames
        Eigen::Matrix3d R_rel = rotations[i].conjugate() * rotations[i-1].toRotationMatrix();
        Eigen::Vector3d t_rel = rotations[i].conjugate() * (translations[i-1] - translations[i]);
        
        relative_pose.block(0, 0, 3, 3) = R_rel;
        relative_pose.block(0, 3, 3, 1) = t_rel;

        ceres::CostFunction* cost_function = 
            PoseGraphError::Create(relative_pose);

        problem.AddResidualBlock(cost_function, loss_function,
                               translations[i-1].data(), rotations[i-1].coeffs().data(),
                               translations[i].data(), rotations[i].coeffs().data());

        // Keep quaternions normalized using shared quaternion parameterization
        problem.SetParameterization(rotations[i-1].coeffs().data(), quaternion_parameterization);
        problem.SetParameterization(rotations[i].coeffs().data(), quaternion_parameterization);
    }
    // 添加闭环相对位姿约束
    std::vector<Eigen::Vector3d> anchor_translations(loop_poses.size());
    std::vector<Eigen::Quaterniond> anchor_rotations(loop_poses.size());

    // Convert poses to Eigen format
    for (size_t i = 0; i < loop_poses.size(); ++i) {
        cv::Mat R;
        cv::Mat(loop_poses[i].rotation()).convertTo(R, CV_64F);
        Eigen::Matrix3d rotation;
        cv2eigen(R, rotation);
        
        anchor_rotations[i] = Eigen::Quaterniond(rotation);
        anchor_translations[i] = Eigen::Vector3d(
            loop_poses[i].translation()[0],
            loop_poses[i].translation()[1],
            loop_poses[i].translation()[2]);
    }
    // add loop closure constraints
    for (size_t i = 1; i <loop_pairs.size(); ++i) {
        Eigen::Matrix4d relative_pose = Eigen::Matrix4d::Identity();
        int idx1 = loop_pairs[i].first;//source idx
        int idx2 = loop_pairs[i].second;//target idx

        // Compute relative transformation between consecutive frames
        Eigen::Matrix3d R_rel = anchor_rotations[idx1].conjugate() * anchor_rotations[idx2].toRotationMatrix();
        Eigen::Vector3d t_rel = anchor_rotations[idx1].conjugate() * (anchor_translations[idx2] - anchor_translations[idx1]);
        
        relative_pose.block(0, 0, 3, 3) = R_rel;
        relative_pose.block(0, 3, 3, 1) = t_rel;

        ceres::CostFunction* cost_function = 
            PoseGraphError::Create(relative_pose);
        int pidx1 = loop_frame_idx[idx1];
        int pidx2 = loop_frame_idx[idx2];
        problem.AddResidualBlock(cost_function, loss_function,
                               translations[pidx2].data(), rotations[pidx2].coeffs().data(),
                               translations[pidx1].data(), rotations[pidx1].coeffs().data());

        // Keep quaternions normalized using shared quaternion parameterization
        // problem.SetParameterization(rotations[i-1].coeffs().data(), quaternion_parameterization);
        // problem.SetParameterization(rotations[i].coeffs().data(), quaternion_parameterization);
    }

    // Add loop closure constraints
    // 给定的loop约束是全局坐标，与第一帧绑定
    for (size_t i = 0; i < loop_frame_idx.size(); ++i) {
        int idx1 = loop_frame_idx[i];
        if (idx1 !=1 || idx1 != poses.size()-1)
        {
            continue;
        }
        
        // Convert loop pose to Eigen format
        cv::Mat R_loop;
        cv::Mat(loop_poses[i].rotation()).convertTo(R_loop, CV_64F);
        Eigen::Matrix3d rotation_loop;
        cv2eigen(R_loop, rotation_loop);
        
        Eigen::Matrix4d relative_pose = Eigen::Matrix4d::Identity();
        relative_pose.block(0, 0, 3, 3) = rotation_loop;
        relative_pose.block(0, 3, 3, 1) = Eigen::Vector3d(
            loop_poses[i].translation()[0],
            loop_poses[i].translation()[1],
            loop_poses[i].translation()[2]);

        ceres::CostFunction* loop_cost_function = 
            PoseGraphError::Create(relative_pose);

        problem.AddResidualBlock(loop_cost_function, loss_function,
                               translations[idx1].data(), rotations[idx1].coeffs().data(),
                               translations[0].data(), rotations[0].coeffs().data());

        // Keep quaternions normalized for loop closure constraints // 上面已经加了归一化约束
        // problem.SetParameterization(rotations[idx1].coeffs().data(), quaternion_parameterization);
        // problem.SetParameterization(rotations.back().coeffs().data(), quaternion_parameterization); 
    }

    // // Convert center to Eigen format
    // Eigen::Vector3d center_eigen(center[0], center[1], center[2]);

    // // Add circular motion constraints for each pose
    // for (size_t i = 0; i < poses.size(); ++i) {
    //     ceres::CostFunction* circular_cost = 
    //         CircularMotionError::Create(center_eigen, radius*0.995);
        
    //     problem.AddResidualBlock(circular_cost, loss_function,
    //                            translations[i].data(),
    //                            rotations[i].coeffs().data());
    // }

    // Fix the first pose
    problem.SetParameterBlockConstant(translations[0].data());
    problem.SetParameterBlockConstant(rotations[0].coeffs().data());

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    
    // Update poses with optimized values
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Matrix3d R = rotations[i].toRotationMatrix();
        cv::Mat R_cv;
        eigen2cv(R, R_cv);
        R_cv.convertTo(R_cv, CV_32F);
        
        poses[i] = Affine3f(
            R_cv,
            Vec3f(translations[i].x(), translations[i].y(), translations[i].z())
        );
    }
}

} // namespace kfusion
