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

bool WarpField::get_volume_flag(const int &x, const int &y, const int &z)
{
    return volume_flag[x + y * vdim_x + z * vdim_x*vdim_y] > 0;
}

void WarpField::expand_nodesflag(const int x, const int y, const int z, const int exp_len)
{
    int start_x, end_x, start_y, end_y, start_z, end_z;
    start_x = std::max(x-exp_len,0);
    end_x = std::min(x+exp_len, vdim_x - 1);
    start_y = std::max(y-exp_len, 0);
    end_y = std::min(y+exp_len, vdim_y - 1);
    start_z = std::max(z-exp_len, 0);
    end_z = std::min(z+exp_len, vdim_z - 1);
    // std::cout<<"expand: "<<exp_len<<", "<<x<<", "<<y<<", "<<z<<", vdim:"<<vdim_x<<", "<<vdim_y<<", "<<vdim_z<<std::endl;
    for (size_t i = start_x; i <= end_x; i++)
    {
        for (size_t j = start_y; j <= end_y; j++)
        {
            for (size_t k = start_z; k <= end_z; k++)
            {
                volume_flag[i + j * vdim_x + k * vdim_x*vdim_y] = 1;
            }
        }
    }
}

/**
 *
 * @param first_frame
 * @param normals
 */
void WarpField::init(const cv::Mat& first_frame, const kfusion::Vec3i &vdims, cv::Affine3f &aff_inv)
{
    std::cout<<"start to init volume flag: "<<first_frame.cols<<", "<<first_frame.rows<<std::endl;
    int vsize = vdims[0]*vdims[1]*vdims[2];
    vdim_x = vdims[0];
    vdim_y = vdims[1];
    vdim_z = vdims[2];
    volume_flag = new int[vsize];
    for(int i = 0;i<vsize;i++)
    {
        volume_flag[i] = 0;
    }
    
    exp_len_ = 20;
    
    // note that nodes_ should be initialized first!!
    // nodes_->resize(first_frame.cols * first_frame.rows);
    nodes_->reserve(first_frame.cols * first_frame.rows);
    auto voxel_size = kfusion::KinFuParams::default_params().volume_size[0] /
                      kfusion::KinFuParams::default_params().volume_dims[0];
    exp_len_ = 0.015/voxel_size;
// //    FIXME:: this is a test, remove later
//     voxel_size = 1;
    std::cout<<"start to init nodes, expand length: "<<exp_len_<<std::endl;
    int step = 1;
    int node_num = 0;
    int not_node_num = 0;
    for(size_t i = 0; i < first_frame.rows; i+=step)
    {
        for(size_t j = 0; j < first_frame.cols; j+=step)
        {
            auto point = first_frame.at<Point>(i,j);
            if(!std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z))
            {
                auto pt_vol = aff_inv * point;
                pt_vol.x = int(pt_vol.x/voxel_size-0.5f);
                pt_vol.y = int(pt_vol.y/voxel_size-0.5f);
                pt_vol.z = int(pt_vol.z/voxel_size-0.5f);
                if(get_volume_flag(pt_vol.x, pt_vol.y, pt_vol.z) ==  false)
                {
                    deformation_node tnode;
                    tnode.transform = utils::DualQuaternion<float>();
                    tnode.vertex = Vec3f(point.x,point.y,point.z);
                    tnode.weight = 3 * voxel_size; //???, only weights to set the area affected
                    nodes_->push_back(tnode);
                    // nodes_->at(i*first_frame.cols+j).transform = utils::DualQuaternion<float>();
                    // nodes_->at(i*first_frame.cols+j).vertex = Vec3f(point.x,point.y,point.z); 
                    // nodes_->at(i*first_frame.cols+j).weight = 3 * voxel_size;
                    // !!!!!
                    // need to transform point to volume coordinates
                    expand_nodesflag(pt_vol.x, pt_vol.y, pt_vol.z, exp_len_);
                    node_num ++;
                }
                else
                {
                    not_node_num++;
                }
            }
        }
    }
    buildKDTree();
}
/**
 * canonical_frame
*/
void WarpField::update_deform_node(const cv::Mat& canonical_frame, cv::Affine3f &aff_inv) // expand the nodes if necessary
{
    std::cout<<"start to expand the nodes, expand length: "<<exp_len_<<std::endl;
    int step = 1;
    int node_num = 0;
    int not_node_num = 0;
    auto voxel_size = kfusion::KinFuParams::default_params().volume_size[0] /
                      kfusion::KinFuParams::default_params().volume_dims[0];
    for(size_t i = 0; i < canonical_frame.rows; i+=step)
    {
        for(size_t j = 0; j < canonical_frame.cols; j+=step)
        {
            auto point = canonical_frame.at<Point>(i,j);
            if(!std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z))
            {
                auto pt_vol = aff_inv * point;
                pt_vol.x = int(pt_vol.x/voxel_size-0.5f);
                pt_vol.y = int(pt_vol.y/voxel_size-0.5f);
                pt_vol.z = int(pt_vol.z/voxel_size-0.5f);
                if(get_volume_flag(pt_vol.x, pt_vol.y, pt_vol.z) ==  false)
                {
                    deformation_node tnode;
                    tnode.transform = utils::DualQuaternion<float>();
                    tnode.vertex = Vec3f(point.x,point.y,point.z);
                    tnode.weight = 3 * voxel_size; //设置node的影响范围
                    nodes_->push_back(tnode);
                    // nodes_->at(i*canonical_frame.cols+j).transform = utils::DualQuaternion<float>();
                    // nodes_->at(i*canonical_frame.cols+j).vertex = Vec3f(point.x,point.y,point.z); 
                    // nodes_->at(i*canonical_frame.cols+j).weight = 3 * voxel_size;
                    // !!!!!
                    // need to transform point to volume coordinates
                    expand_nodesflag(pt_vol.x, pt_vol.y, pt_vol.z, exp_len_);
                    // std::cout<<"add node: "<<pt_vol.x<<","<< pt_vol.y<<", "<<pt_vol.z<<": "<<point.x<<","<<point.y<<","<<point.z<<", expflag: "<<get_volume_flag(pt_vol.x, pt_vol.y, pt_vol.z)<<std::endl;
                    // std::cout<<"add node: "<<point.x<<","<<point.y<<","<<point.z<<std::endl;
                    node_num ++;
                }
                else
                {
                    not_node_num++;
                }
            }
        }
    }
    buildKDTree();
    std::cout<<"add new node number: "<<node_num<<std::endl;
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
                            const std::vector<Vec3f> &live_normals //live normals are not used in optimization
)
{
    // std::cout<<"node size: "<<nodes_->size()<<std::endl;
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
        // filter out error correspondence
       
        if(fabs(canonical_vertices[i][2]-live_vertices[i][2])>0.1)
            continue;
        std::cout<<canonical_vertices[i][0]<<","<<live_vertices[i][0]<<".."<<canonical_vertices[i][1]<<","<<live_vertices[i][1]<<std::endl;
        getWeightsAndUpdateKNN(canonical_vertices[i], weights);
        // 当前点距离node过远，不考虑用于node的位置优化
        if(weights[0]==0)
        {
            continue;
        }
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
    options.max_num_iterations = 3;
    // options.num_linear_solver_threads = 8;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    warpProblem.updateWarp();
    // for (size_t i = 0; i < nodes_->size(); i++)
    // {
    //     auto node = nodes_->at(i);
    //     std::cout<<node.transform<<std::endl;
    // }
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
void WarpField::warp(std::vector<Vec3f>& points, std::vector<Vec3f>& normals, bool flag_debug) 
{
    flag_exp = false;
    int i = -1;
    for (auto& point : points)
    {
        i++;
        if(std::isnan(point[0]) /*|| std::isnan(normals[i][0])*/)
            continue;
        if(flag_debug)
            std::cout<<"dqb pt: "<<i<<","<<point[0]<<","<<point[1]<<", "<<point[2]<<std::endl;
        utils::DualQuaternion<float> dqb = DQB(point); 
        if(flag_debug)
            std::cout<<"++dqb point: "<<point<<","<<dqb<<std::endl;
        dqb.transform(point);
        if(flag_debug)
            std::cout<<"--dqb point: "<<point<<","<<dqb<<std::endl;

        point = warp_to_live_ * point;
        if(std::isnan(normals[i][0]))
            continue;
        dqb.transform(normals[i]);
        normals[i] = warp_to_live_ * normals[i];
        
    }
}

/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
utils::DualQuaternion<float> WarpField::DQB(const Vec3f& vertex) 
{
    float weights[KNN_NEIGHBOURS];
    getWeightsAndUpdateKNN(vertex, weights);
    utils::Quaternion<float> translation_sum(0,0,0,0);
    utils::Quaternion<float> rotation_sum(0,0,0,0);
    if(weights[0]==0)
    {
        auto &vt = nodes_->at(ret_index_[0]).vertex;
        std::cout<<"err knn: "<<vertex[0]<<","<<vertex[1]<<", "<<vertex[2]<<"--"<<vt[0]<<","<<vt[1]<<", "<<vt[2]<<std::endl;
    }
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
    {
        if(KNN_NEIGHBOURS == 1)
        {
            weights[0] = 1;
            flag_exp = true;
        }
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
    cloud.pts.clear();
    cloud.pts.resize(nodes_->size());
    for(size_t i = 0; i < nodes_->size(); i++)
        cloud.pts[i] = nodes_->at(i).vertex;
    index_->buildIndex();
}

const cv::Mat WarpField::getNodesAsMat() const
{
    cv::Mat matrix(1, nodes_->size(), CV_32FC3);
    for(int i = 0; i < nodes_->size(); i++)
    {
        // nodes_->at(i).transform.getTranslation(matrix.at<cv::Vec3f>(i));
        // matrix.at<cv::Vec3f>(i) += nodes_->at(i).vertex;
        matrix.at<cv::Vec3f>(i) = nodes_->at(i).vertex;
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