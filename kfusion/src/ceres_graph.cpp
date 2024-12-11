#include "kfusion/ceres_graph.hpp"

namespace kfusion {

void CeresGraph::optimizePoseGraph(std::vector<Affine3f>& poses,
                                 const std::vector<int>& loop_frame_idx,
                                 const std::vector<Affine3f>& loop_poses) {
    if (poses.empty() || loop_frame_idx.empty() || loop_poses.empty())
        return;

    // Problem setup
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

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
        Eigen::Matrix3d R_rel = rotations[i-1].conjugate() * rotations[i].toRotationMatrix();
        Eigen::Vector3d t_rel = rotations[i-1].conjugate() * (translations[i] - translations[i-1]);
        
        relative_pose.block(0, 0, 3, 3) = R_rel;
        relative_pose.block(0, 3, 3, 1) = t_rel;

        ceres::CostFunction* cost_function = 
            PoseGraphError::Create(relative_pose);

        problem.AddResidualBlock(cost_function, loss_function,
                               translations[i-1].data(), rotations[i-1].coeffs().data(),
                               translations[i].data(), rotations[i].coeffs().data());

        // Keep quaternions normalized
        problem.SetManifold(rotations[i-1].coeffs().data(), 
                          new ceres::EigenQuaternionManifold);
        problem.SetManifold(rotations[i].coeffs().data(), 
                          new ceres::EigenQuaternionManifold);
    }

    // Add loop closure constraints
    for (size_t i = 0; i < loop_frame_idx.size(); ++i) {
        int idx1 = loop_frame_idx[i];
        
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
                               translations.back().data(), rotations.back().coeffs().data());
    }

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
    
    // Update poses with optimized values
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Matrix3f rotation = rotations[i].toRotationMatrix().cast<float>();
        cv::Mat R_optimized;
        eigen2cv(rotation, R_optimized);
        
        poses[i] = Affine3f(
            R_optimized,
            Vec3f(translations[i].x(), translations[i].y(), translations[i].z())
        );
    }
}

} // namespace kfusion
