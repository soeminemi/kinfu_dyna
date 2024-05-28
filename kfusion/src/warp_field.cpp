#include <kfusion/utils/dual_quaternion.hpp>
#include <kfusion/utils/knn_point_cloud.hpp>
#include <kfusion/nanoflann.hpp>
#include "kfusion/warp_field.hpp"
#include "internal.hpp"
#include "precomp.hpp"
#include <kfusion/optimisation.hpp>

using namespace kfusion;
std::vector<utils::DualQuaternion<float>> neighbours; //THIS SHOULD BE SOMEWHERE ELSE BUT TOO SLOW TO REINITIALISE
utils::PointCloud cloud;
nanoflann::KNNResultSet<float> *resultSet_;
std::vector<float> out_dist_sqr_;
std::vector<size_t> ret_index_;

WarpField::WarpField()
{
    nodes_ = new std::vector<deformation_node>();
    index_ = new kd_tree_t(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    ret_index_ = std::vector<size_t>(KNN_NEIGHBOURS);
    out_dist_sqr_ = std::vector<float>(KNN_NEIGHBOURS);
    resultSet_ = new nanoflann::KNNResultSet<float>(KNN_NEIGHBOURS);
    resultSet_->init(&ret_index_[0], &out_dist_sqr_[0]);
    neighbours = std::vector<utils::DualQuaternion<float>>(KNN_NEIGHBOURS);
    warp_to_live_ = cv::Affine3f();
}

WarpField::~WarpField()
{
    delete nodes_;
    delete resultSet_;
    delete index_;
}

/**
 *
 * @param first_frame
 * @param normals
 */
void WarpField::init(const cv::Mat& first_frame)
{
    nodes_->resize(first_frame.cols * first_frame.rows);
    auto voxel_size = kfusion::KinFuParams::default_params().volume_size[0] /
                      kfusion::KinFuParams::default_params().volume_dims[0];

//    FIXME:: this is a test, remove later
    voxel_size = 1;
    int step = 50;
    std::cout << "voxel size init mat: " << voxel_size << ","<<first_frame.cols<<" * "<< first_frame.rows<<std::endl;
    for(size_t i = 0; i < first_frame.rows; i+=step)
        for(size_t j = 0; j < first_frame.cols; j+=step)
        {
            auto point = first_frame.at<Point>(i,j);
            if(!std::isnan(point.x))
            {
                nodes_->at(i*first_frame.cols+j).transform = utils::DualQuaternion<float>();
                nodes_->at(i*first_frame.cols+j).vertex = Vec3f(point.x,point.y,point.z);
                nodes_->at(i*first_frame.cols+j).weight = 3 * voxel_size;
            }
        }
    buildKDTree();
}

/**
 *
 * @param first_frame
 * @param normals
 */
void WarpField::init(const std::vector<Vec3f>& first_frame)
{
    nodes_->resize(first_frame.size());
    auto voxel_size = kfusion::KinFuParams::default_params().volume_size[0] /
                      kfusion::KinFuParams::default_params().volume_dims[0];
    std::cout<<"node size init vec: "<<nodes_->size()<<std::endl;
//    FIXME: this is a test, remove
    voxel_size = 1;
    for (size_t i = 0; i < first_frame.size(); i++)
    {
        auto point = first_frame[i];
        if (!std::isnan(point[0]))
        {
            nodes_->at(i).transform = utils::DualQuaternion<float>();
            nodes_->at(i).vertex = point;
            nodes_->at(i).weight = 3 * voxel_size;
        }
    }
    buildKDTree();
}

/**
 * \brief
 * \param frame
 * \param normals
 * \param pose
 * \param tsdfVolume
 * \param edges
 */
void WarpField::energy(const cuda::Cloud &frame,
                       const cuda::Normals &normals,
                       const Affine3f &pose,
                       const cuda::TsdfVolume &tsdfVolume,
                       const std::vector<std::pair<utils::DualQuaternion<float>, utils::DualQuaternion<float>>> &edges
)
{
    assert(normals.cols()==frame.cols());
    assert(normals.rows()==frame.rows());
}

/**
 *
 * @param canonical_vertices
 * @param canonical_normals
 * @param live_vertices
 * @param live_normals
 * @return
 */
void WarpField::energy_data(const std::vector<Vec3f> &canonical_vertices,
                            const std::vector<Vec3f> &canonical_normals,
                            const std::vector<Vec3f> &live_vertices,
                            const std::vector<Vec3f> &live_normals
)
{
    std::cout<<"node size: "<<nodes_->size()<<std::endl;
    ceres::Problem problem;
    float weights[KNN_NEIGHBOURS];
    unsigned long indices[KNN_NEIGHBOURS];

    WarpProblem warpProblem(this);
    for(int i = 0; i < live_vertices.size(); i++)
    {
        if(std::isnan(canonical_vertices[i][0]) ||
           std::isnan(canonical_vertices[i][1]) ||
           std::isnan(canonical_vertices[i][2]) ||
           std::isnan(live_vertices[i][0]) ||
           std::isnan(live_vertices[i][1]) ||
           std::isnan(live_vertices[i][2]))
            continue;
        getWeightsAndUpdateKNN(canonical_vertices[i], weights);

//        FIXME: could just pass ret_index
        for(int j = 0; j < KNN_NEIGHBOURS; j++)
            indices[j] = ret_index_[j];

        ceres::CostFunction* cost_function = DynamicFusionDataEnergy::Create(live_vertices[i],
                                                                             live_normals[i],
                                                                             canonical_vertices[i],
                                                                             canonical_normals[i],
                                                                             this,
                                                                             weights,
                                                                             indices);
        problem.AddResidualBlock(cost_function,  NULL /* squared loss */, warpProblem.mutable_epsilon(indices));

    }
    //基于ceres求解warpField
    ceres::Solver::Options options;
//    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    // options.num_linear_solver_threads = 8;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    warpProblem.updateWarp();
}
/**
 * \brief
 * \param edges
 */
void WarpField::energy_reg(const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
        kfusion::utils::DualQuaternion<float>>> &edges)
{

}


/**
 *
 * @param points
 * @param normals
 */
void WarpField::warp(std::vector<Vec3f>& points, std::vector<Vec3f>& normals) const
{
    int i = 0;
    for (auto& point : points)
    {
        if(std::isnan(point[0]) || std::isnan(normals[i][0]))
            continue;
        utils::DualQuaternion<float> dqb = DQB(point);
        dqb.transform(point);
        point = warp_to_live_ * point;

        dqb.transform(normals[i]);
        normals[i] = warp_to_live_ * normals[i];
        i++;
    }
}

/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
utils::DualQuaternion<float> WarpField::DQB(const Vec3f& vertex) const
{
    float weights[KNN_NEIGHBOURS];
    getWeightsAndUpdateKNN(vertex, weights);
    utils::Quaternion<float> translation_sum(0,0,0,0);
    utils::Quaternion<float> rotation_sum(0,0,0,0);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
    {
        translation_sum += weights[i] * nodes_->at(ret_index_[i]).transform.getTranslation();
        rotation_sum += weights[i] * nodes_->at(ret_index_[i]).transform.getRotation();
    }
    rotation_sum.normalize();
    auto res = utils::DualQuaternion<float>(translation_sum, rotation_sum);
    return res;
}

/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
void WarpField::getWeightsAndUpdateKNN(const Vec3f& vertex, float weights[KNN_NEIGHBOURS]) const
{
    KNN(vertex);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
        weights[i] = weighting(out_dist_sqr_[i], nodes_->at(ret_index_[i]).weight);
}

/**
 * \brief
 * \param squared_dist
 * \param weight
 * \return
 */
float WarpField::weighting(float squared_dist, float weight) const
{
    return (float) exp(-squared_dist / (2 * weight * weight));
}

/**
 * \brief
 * \return
 */
void WarpField::KNN(Vec3f point) const
{
    resultSet_->init(&ret_index_[0], &out_dist_sqr_[0]);
    index_->findNeighbors(*resultSet_, point.val, nanoflann::SearchParams(10));
}

/**
 * \brief
 * \return
 */
const std::vector<deformation_node>* WarpField::getNodes() const
{
    return nodes_;
}

/**
 * \brief
 * \return
 */
std::vector<deformation_node>* WarpField::getNodes()
{
    return nodes_;
}

/**
 * \brief
 * \return
 */
void WarpField::buildKDTree()
{
    //    Build kd-tree with current warp nodes.
    cloud.pts.resize(nodes_->size());
    for(size_t i = 0; i < nodes_->size(); i++)
        cloud.pts[i] = nodes_->at(i).vertex;
    index_->buildIndex();
}`

const cv::Mat WarpField::getNodesAsMat() const
{
    cv::Mat matrix(1, nodes_->size(), CV_32FC3);
    for(int i = 0; i < nodes_->size(); i++)
    {
        nodes_->at(i).transform.getTranslation(matrix.at<cv::Vec3f>(i));
        matrix.at<cv::Vec3f>(i) += nodes_->at(i).vertex;
    }
    return matrix;
}

/**
 * \brief
 */
void WarpField::clear()
{

}
void WarpField::setWarpToLive(const Affine3f &pose)
{
    warp_to_live_ = pose;
}

std::vector<float>* WarpField::getDistSquared() const
{
    return &out_dist_sqr_;
}
std::vector<size_t>* WarpField::getRetIndex() const
{
    return &ret_index_;
}