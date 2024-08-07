#ifndef KFUSION_WARP_FIELD_HPP
#define KFUSION_WARP_FIELD_HPP

#include <kfusion/utils/dual_quaternion.hpp>
#include <kfusion/types.hpp>
#include <kfusion/nanoflann.hpp>
#include <kfusion/utils/knn_point_cloud.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>

#define KNN_NEIGHBOURS 8 //when optimize the parameters ， we only need KNN_NEIGHBOURS to be 1
namespace kfusion
{
    typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, utils::PointCloud>,
            utils::PointCloud,
            3 /* dim */
    > kd_tree_t;


    /*!
     * \struct node
     * \brief A node of the warp field
     * \details The state of the warp field Wt at time t is defined by the values of a set of n
     * deformation nodes Nt_warp = {dg_v, dg_w, dg_se3}_t. Here, this is represented as follows
     *
     * \var node::vertex
     * Position of the vertex in space. This will be used when computing KNN for warping points.
     *
     * \var node::transform
     * Transformation for each vertex to warp it into the live frame, equivalent to dg_se in the paper.
     *
     * \var node::weight
     * Equivalent to dg_w
     */

    struct deformation_node
    {
        Vec3f vertex;
        kfusion::utils::DualQuaternion<float> transform;
        float weight = 0;
        std::vector<int> nb_idxes;// neighbour indexs
    }; 

//投影到图像
    struct Project
    {
        cv::Vec2f f, c;
        Project(){}
        Project(float fx, float fy, float cx, float cy);
        cv::Vec2f operator()(const cv::Vec3f& p) ;
    };
//从图像反投影
    struct Reproject
    {
        Reproject() {}
        Reproject(float fx, float fy, float cx, float cy);
        cv::Vec2f finv, c;
        cv::Vec3f operator()(int x, int y, float z) ;
    };

    class WarpField
    {
    public:
        WarpField();
        ~WarpField();
        //初始化warp filed, 如何扩展？
        void init(const cv::Mat& first_frame, const kfusion::Vec3i &vdims, Affine3f &aff_inv);
        void update_deform_node(const cv::Mat& canonical_frame,cv::Affine3f &aff_inv); // expand the nodes if necessary
        void construct_edge(std::vector<int> &appended_nodes_idxs);
        void init(const std::vector<Vec3f>& first_frame);
        //calculate the energy of the warp field
        void energy(const cuda::Cloud &frame,
                    const cuda::Normals &normals,
                    const Affine3f &pose,
                    const cuda::TsdfVolume &tsdfVolume,
                    const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                            kfusion::utils::DualQuaternion<float>>> &edges
        );

        void energy_data(const std::vector<Vec3f> &canonical_vertices,
                          const std::vector<Vec3f> &canonical_normals,
                          const std::vector<Vec3f> &live_vertices,
                          const std::vector<Vec3f> &live_normals);
                          
        void energy_reg(const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                kfusion::utils::DualQuaternion<float>>> &edges);

        void warp(std::vector<Vec3f>& points, std::vector<Vec3f>& normals, bool flag_debug = false) ;
        void getWarpedNode(std::vector<Vec3f> &warp_nodes);
        utils::DualQuaternion<float> DQB(const Vec3f& vertex);
        void transform_to_live(Vec3f &point);
        void rotate_to_live(Vec3f &normal);

        void getWeightsAndUpdateKNN(const Vec3f& vertex, float weights[KNN_NEIGHBOURS]) const;

        float weighting(float squared_dist, float weight) const;
        void KNN(Vec3f point) const;

        void clear();

        const std::vector<deformation_node>* getNodes() const;
        std::vector<deformation_node>* getNodes();
        const cv::Mat getNodesAsMat() const;
        void setWarpToLive(const Affine3f &pose);
        std::vector<float>* getDistSquared() const;
        std::vector<size_t>* getRetIndex() const;
        void buildKDTree();
        void expand_nodesflag(const int x, const int y, const int z, const int exp_len);
        bool get_volume_flag(const int &x, const int &y, const int &z);
        void setProject(float fx, float fy, float cx, float cy);
        deformation_node& getDeformationNode(int idx);

        // test correspondence
        bool testCorrrespondence(const std::vector<cv::Vec3f>* live_vertex_,
                            const std::vector<cv::Vec3f>* live_normal_,
                            const cv::Vec3f& canonical_vertex_,
                            const cv::Vec3f& canonical_normal_,
                            kfusion::WarpField *warpField_,
                            const float weights_[KNN_NEIGHBOURS],
                            const unsigned long knn_indices_[KNN_NEIGHBOURS],
                            std::vector<Vec3f> & pts_live,
                            std::vector<Vec3f> & pts_cano,
                            std::vector<cv::Scalar> &cls_cano);
            
    

    private:
        std::vector<deformation_node>* nodes_;
        kd_tree_t* index_;
        Affine3f warp_to_live_;
        int *volume_flag;
        int vdim_x;
        int vdim_y;
        int vdim_z;
        int exp_len_;
        double total_err;
        double total_num;
    public:
        int fail_num;
        bool flag_exp;
        Project projector_;
        int image_width;
        int image_height;
        Affine3f aff_inv;
    };
}
#endif //KFUSION_WARP_FIELD_HPP
