#ifndef KFUSION_OPTIMISATION_H
#define KFUSION_OPTIMISATION_H
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <kfusion/warp_field.hpp>
#include <kfusion/utils/dual_quaternion.hpp>
#include <opencv2/surface_matching/icp.hpp>
#define THRES_CL 0.05
struct DynamicFusionDataEnergy
{
    DynamicFusionDataEnergy(const std::vector<cv::Vec3f>* live_vertex,
                            const std::vector<cv::Vec3f>* live_normal,
                            const cv::Vec3f& canonical_vertex,
                            const cv::Vec3f& canonical_normal,
                            kfusion::WarpField *warpField,
                            kfusion::utils::DualQuaternion<float> dqb,
                            const float weights[KNN_NEIGHBOURS],
                            const unsigned long knn_indices[KNN_NEIGHBOURS]
                            )
            : live_vertex_(live_vertex),
              live_normal_(live_normal),
              canonical_vertex_(canonical_vertex),
              canonical_normal_(canonical_normal),
              warpField_(warpField),
              dqb_(dqb)
    {
        weights_ = new float[KNN_NEIGHBOURS];
        knn_indices_ = new unsigned long[KNN_NEIGHBOURS];
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
        {
            weights_[i] = weights[i];
            knn_indices_[i] = knn_indices[i];
        }
    }
    ~DynamicFusionDataEnergy()
    {
        delete[] weights_;
        delete[] knn_indices_;
    } 
    // 计算warp对应的loss original, for comparing
//     template <typename T>
//     bool operator()(T const * const * epsilon_, T* residuals) const
//     {
//         auto nodes = warpField_->getNodes();

//         T total_translation[3] = {T(0), T(0), T(0)};
//         float total_translation_float[3] = {0, 0, 0};

//         for(int i = 0; i < KNN_NEIGHBOURS; i++)
//         {
//             // auto quat = nodes->at(knn_indices_[i]).transform;
//             // //---not used-----
//             // cv::Vec3f vert;
//             // quat.getTranslation(vert); //当前node的平移

//             // T eps_t[3] = {epsilon_[i][3], epsilon_[i][4], epsilon_[i][5]}; //参数对应的平移，和node对应的平移不是一样的吗？

//             // float temp[3];
//             // quat.getTranslation(temp[0], temp[1], temp[2]);

// //            total_translation[0] += (T(temp[0]) +  eps_t[0]);
// //            total_translation[1] += (T(temp[1]) +  eps_t[1]);
// //            total_translation[2] += (T(temp[2]) +  eps_t[2]);
// //
//             // total_translation[0] += (T(temp[0]) +  eps_t[0]) * T(weights_[i]);
//             // total_translation[1] += (T(temp[1]) +  eps_t[1]) * T(weights_[i]);
//             // total_translation[2] += (T(temp[2]) +  eps_t[2]) * T(weights_[i]);

//             total_translation[0] += (epsilon_[i][3]) * T(weights_[i]);
//             total_translation[1] += (epsilon_[i][4]) * T(weights_[i]);
//             total_translation[2] += (epsilon_[i][5]) * T(weights_[i]);
//             //损失函数和论文并不一致，没有使用点到面的距离
//         }
        
//         residuals[0] = T(live_vertex_[0] - canonical_vertex_[0]) - total_translation[0];
//         residuals[1] = T(live_vertex_[1] - canonical_vertex_[1]) - total_translation[1];
//         residuals[2] = T(live_vertex_[2] - canonical_vertex_[2]) - total_translation[2];
//         return true;
//     }

    template <typename T>
    bool operator()(T const * const * epsilon_, T* residuals) const
    {
        //1. got DOB of the canonical vertex
        //2. transform the canonical vertex
        //3. find the correspondent live vertex
        //4. calc the loss
        // auto nodes = warpField_->getNodes();

        // //1. calc DQB of canonical vertex// do not calc dqb in loss funciton, calc outside--Y
        // kfusion::utils::Quaternion<float> translation_sum(0,0,0,0);
        // kfusion::utils::Quaternion<float> rotation_sum(0,0,0,0);
        // if(weights_[0] <= 0.01)
        // {
        //     translation_sum += 1 * nodes->at(knn_indices_[0]).transform.getTranslation();
        //     rotation_sum += 1 * nodes->at(knn_indices_[0]).transform.getRotation();
        // }
        // else
        // {
        //     for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
        //     {
                
        //         translation_sum += weights_[i] * nodes->at(knn_indices_[i]).transform.getTranslation();
        //         rotation_sum += weights_[i] * nodes->at(knn_indices_[i]).transform.getRotation();
        //     }
        // }
        // rotation_sum.normalize();
        // auto dqb = kfusion::utils::DualQuaternion<float>(translation_sum, rotation_sum); //Got DQB of canonical vertex
        //2. transform canonical vertex
        // T weight_t = T(weights_[0]);
        T weight_t = T(1.0);
        auto tv = dqb_.transform(canonical_vertex_);
        auto warp_cv = tv;
        auto warp_cn = dqb_.rotate(canonical_normal_);
        // std::cout<<warp_cn<<std::endl;
        warpField_->transform_to_live(tv);
        //3. find the corresponding live vertex
        auto live_coo = warpField_->projector_(tv);
        auto idx = round(live_coo[0]) + round(live_coo[1]) * warpField_->image_width;
        if(idx < 0 || idx >= warpField_->image_width * warpField_->image_height)
        {
            residuals[0] = weight_t*T(THRES_CL);
            residuals[1] = weight_t*T(THRES_CL);
            residuals[2] = weight_t*T(THRES_CL);
            return true;
        }
        auto live_vt = (*live_vertex_)[idx];
        if(std::isnan(live_vt[0]) ||std::isnan(live_vt[1]) ||std::isnan(live_vt[2]))
        {
            residuals[0] = weight_t*T(THRES_CL);
            residuals[1] = weight_t*T(THRES_CL);
            residuals[2] = weight_t*T(THRES_CL);
            return true;
        }
        //4. 获得DQB，考虑到只优化一个，不用加权了
        T temp_rotation[9];
//         for(int i = 0; i < /*KNN_NEIGHBOURS*/1; i++)
//         {
//             // T eular[3] = {epsilon_[i][3], epsilon_[i][4], epsilon_[i][5]};
//             // T eps_t[3] = {epsilon_[i][3], epsilon_[i][4], epsilon_[i][5]}; //参数对应的平移
//             // EulerAnglesToRotationMatrix(eular, temp_rotation);
//         }
        

        T cano_v[3] = {T(warp_cv[0]),T(warp_cv[1]),T(warp_cv[2])};

        auto ipose_live_v = warpField_->aff_inv * live_vt; //transform to the initial pose 

        if(fabs(ipose_live_v[2]-canonical_vertex_[2])>THRES_CL)
        {
            residuals[0] = weight_t*(THRES_CL);
            residuals[1] = weight_t*(THRES_CL);
            residuals[2] = weight_t*(THRES_CL);
        }
        else
        {
            T eular[3] = {epsilon_[0][0], epsilon_[0][1], epsilon_[0][2]}; //参数对应的Eular angle
            T eps_t[3] = {epsilon_[0][3], epsilon_[0][4], epsilon_[0][5]}; //参数对应的平移
            ceres::EulerAnglesToRotationMatrix(eular, 3, temp_rotation);

            T live_v[3] = {T(ipose_live_v[0]),T(ipose_live_v[1]),T(ipose_live_v[2])};
            T cano_n[3] = {T(warp_cn[0]),T(warp_cn[1]),T(warp_cn[2])};
            // calc the rotation
            T cx = temp_rotation[0] * cano_v[0] + temp_rotation[1] * cano_v[1] + temp_rotation[2] * cano_v[2];
            T cy = temp_rotation[3] * cano_v[0] + temp_rotation[4] * cano_v[1] + temp_rotation[5] * cano_v[2];
            T cz = temp_rotation[6] * cano_v[0] + temp_rotation[7] * cano_v[1] + temp_rotation[8] * cano_v[2];

            T nx = temp_rotation[0] * cano_n[0] + temp_rotation[1] * cano_n[1] + temp_rotation[2] * cano_n[2];
            T ny = temp_rotation[3] * cano_n[0] + temp_rotation[4] * cano_n[1] + temp_rotation[5] * cano_n[2];
            T nz = temp_rotation[6] * cano_n[0] + temp_rotation[7] * cano_n[1] + temp_rotation[8] * cano_n[2];

            T delta_v[3] = {(cx + eps_t[0] -live_v[0]) ,(cy+ eps_t[1] -live_v[1]) ,(cz+ eps_t[2]-live_v[2])};
            // residuals[0] = weight_t*(nx * delta_v[0] + ny * delta_v[1] + nz * delta_v[2]);
            residuals[0] = weight_t * (delta_v[0]);
            residuals[1] = weight_t * (delta_v[1]);
            residuals[2] = weight_t * (delta_v[2]);
        }

        return true;
    }

/**
 * Tukey loss function as described in http://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/12-1.pdf
 * \param x
 * \param c
 * \return
 *
 * \note
 * The value c = 4.685 is usually used for this loss function, and
 * it provides an asymptotic efficiency 95% that of linear
 * regression for the normal distribution
 *
 * In the paper, a value of 0.01 is suggested for c
 */
    template <typename T>
    T tukeyPenalty(T x, T c = T(0.01)) const
    {
        return ceres::abs(x) <= c ? x * ceres::pow((T(1.0) - (x * x) / (c * c)), 2) : T(0.0);
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    // TODO: this will only have one residual at the end, remember to change
    static ceres::CostFunction* Create(const std::vector<cv::Vec3f>* live_vertex,
                                       const std::vector<cv::Vec3f>* live_normal,
                                       const cv::Vec3f& canonical_vertex,
                                       const cv::Vec3f& canonical_normal,
                                       kfusion::WarpField* warpField,
                                       const float weights[KNN_NEIGHBOURS],
                                       const unsigned long ret_index[KNN_NEIGHBOURS],
                                       const kfusion::utils::DualQuaternion<float> dqb)
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<DynamicFusionDataEnergy, 4>(
                new DynamicFusionDataEnergy(live_vertex,
                                            live_normal,
                                            canonical_vertex,
                                            canonical_normal,
                                            warpField,
                                            dqb,
                                            weights,
                                            ret_index
                                            ));
        for(int i=0; i < /*KNN_NEIGHBOURS*/1; i++)
            cost_function->AddParameterBlock(6);
        cost_function->SetNumResiduals(3);
        return cost_function;
    }

    const std::vector<cv::Vec3f> *live_vertex_;
    const std::vector<cv::Vec3f> *live_normal_;
    const cv::Vec3f canonical_vertex_;
    const cv::Vec3f canonical_normal_;

    float *weights_;

    unsigned long *knn_indices_;

    kfusion::WarpField *warpField_;

    kfusion::utils::DualQuaternion<float> dqb_;
};

struct DynamicFusionRegEnergy
{
    DynamicFusionRegEnergy(){};
    ~DynamicFusionRegEnergy(){};
    template <typename T>
    bool operator()(T const * const * epsilon_, T* residuals) const
    {
        residuals[0] = epsilon_[0][0];
        residuals[1] = epsilon_[0][1];
        residuals[2] = epsilon_[0][2];
        residuals[3] = epsilon_[0][3];
        residuals[4] = epsilon_[0][4];
        residuals[5] = epsilon_[0][5];
        return true;
    }

/**
 * Huber penalty function, implemented as described in https://en.wikipedia.org/wiki/Huber_loss
 * In the paper, a value of 0.0001 is suggested for delta.
 * \param a
 * \param delta
 * \return
 */
    template <typename T>
    T huberPenalty(T a, T delta = 0.0001) const
    {
        return ceres::abs(a) <= delta ? a * a / 2 : delta * ceres::abs(a) - delta * delta / 2;
    }

    static ceres::CostFunction* Create()
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<DynamicFusionRegEnergy, 4>(
                new DynamicFusionRegEnergy());
        for(int i=0; i < /*KNN_NEIGHBOURS*/1; i++)
            cost_function->AddParameterBlock(6);
        cost_function->SetNumResiduals(6);
        return cost_function;
    }
};

struct DynamicFusionEdgeEnergy
{
    DynamicFusionEdgeEnergy(int i, int j, kfusion::WarpField* warpField):i_(i),j_(j), warpField_(warpField)
    {}
    ~DynamicFusionEdgeEnergy(){}
    template <typename T>
    bool operator()(T const * const * epsilon_, T* residuals) const
    {
        // std::cout<<"opt: "<<i_<<", "<<j_<<std::endl;
        auto &nd_i = warpField_->getDeformationNode(i_);
        auto &nd_j = warpField_->getDeformationNode(j_);
  
        auto vi = nd_j.vertex;
        auto vj = nd_j.vertex;

        nd_i.transform.transform(vi);
        nd_j.transform.transform(vj);

        T R_i[9];
        T R_j[9];
        T eular_i[3] = {epsilon_[0][0], epsilon_[0][1], epsilon_[0][2]}; //参数对应的Eular angle
        T t_i[3] = {epsilon_[0][3], epsilon_[0][4], epsilon_[0][5]}; //参数对应的平移

        T eular_j[3] = {epsilon_[1][0], epsilon_[1][1], epsilon_[1][2]}; //参数对应的Eular angle
        T t_j[3] = {epsilon_[1][3], epsilon_[1][4], epsilon_[1][5]}; //参数对应的平移

        ceres::EulerAnglesToRotationMatrix(eular_i, 3, R_i);
        ceres::EulerAnglesToRotationMatrix(eular_j, 3, R_j);

        T v_i[3] = {T(vi[0]),T(vi[1]),T(vi[2])};
        T v_j[3] = {T(vj[0]),T(vj[1]),T(vj[2])};
        // calc the rotation
        T cx_i = R_i[0] * v_i[0] + R_i[1] * v_i[1] + R_i[2] * v_i[2] + t_i[0];
        T cy_i = R_i[3] * v_i[0] + R_i[4] * v_i[1] + R_i[5] * v_i[2] + t_i[1];
        T cz_i = R_i[6] * v_i[0] + R_i[7] * v_i[1] + R_i[8] * v_i[2] + t_i[2];

        T cx_j = R_j[0] * v_j[0] + R_j[1] * v_j[1] + R_j[2] * v_j[2] + t_j[0];
        T cy_j = R_j[3] * v_j[0] + R_j[4] * v_j[1] + R_j[5] * v_j[2] + t_j[1];
        T cz_j = R_j[6] * v_j[0] + R_j[7] * v_j[1] + R_j[8] * v_j[2] + t_j[2];
        
        residuals[0] = 0.5*(cx_i - cx_j);
        residuals[1] = 0.5*(cy_i - cy_j);
        residuals[2] = 0.5*(cz_i - cz_j);

        return true;
    }

    static ceres::CostFunction* Create(int i, int j, kfusion::WarpField* warpField)
    {
        // std::cout<<"add "<<i<<"and "<<j<<std::endl;
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<DynamicFusionEdgeEnergy, 4>(
                new DynamicFusionEdgeEnergy(i,j,warpField));
        for(int i=0; i < 2; i++)
            cost_function->AddParameterBlock(6);
        cost_function->SetNumResiduals(3);
        return cost_function;
    }
    int i_;
    int j_;
    kfusion::WarpField* warpField_;
};

class WarpProblem {
public:
    explicit WarpProblem(kfusion::WarpField *warp) : warpField_(warp)
    {
        //初始化warp对应的parameters, 0,1,2表示旋转，x,y,z表示旋转轴的方向，其模表示旋转的角度，3，4，5表示平移
        parameters_ = new double[warpField_->getNodes()->size() * 6];
        for(int i = 0; i < warp->getNodes()->size() * 6; i+=6)
        {
            auto transform = warp->getNodes()->at(i / 6).transform;
            float x,y,z;
            // //平移
            // transform.getTranslation(x,y,z);
            // parameters_[i] = x;
            // parameters_[i+1] = y;
            // parameters_[i+2] = z;
            // //旋转
            // transform.getRotation().getRodrigues(x,y,z);
            // parameters_[i+3] = x;
            // parameters_[i+4] = y;
            // parameters_[i+5] = z;
            // // 旋转
            // transform.getRotation().getRodrigues(x,y,z);
            // parameters_[i] = x;
            // parameters_[i+1] = y;
            // parameters_[i+2] = z;
            // // 平移
            // transform.getTranslation(x,y,z);
            // parameters_[i+3] = x;
            // parameters_[i+4] = y;
            // parameters_[i+5] = z;
            // 旋转
            parameters_[i]   = 0;
            parameters_[i+1] = 0;
            parameters_[i+2] = 0;
            // 平移
            parameters_[i+3] = 0;
            parameters_[i+4] = 0;
            parameters_[i+5] = 0;
        }
    };

    ~WarpProblem() {
        delete parameters_;
    }
    std::vector<double*> mutable_epsilon(const unsigned long *index_list) const
    {
        std::vector<double*> mutable_epsilon_(/*KNN_NEIGHBOURS*/1);
        for(int i = 0; i < 1/*KNN_NEIGHBOURS*/; i++)
            mutable_epsilon_[i] = &(parameters_[index_list[i] * 6]); // Blocks of 6
        return mutable_epsilon_;
    }

    std::vector<double*> mutable_epsilon_edge(const unsigned long i, const unsigned long j) const
    {
        std::vector<double*> mutable_epsilon_edge_(2);
        mutable_epsilon_edge_[0] = &(parameters_[i * 6]); // Blocks of 6
        mutable_epsilon_edge_[1] = &(parameters_[j * 6]); // Blocks of 6
        return mutable_epsilon_edge_;
    }

    std::vector<double*> mutable_epsilon(const std::vector<size_t>& index_list) const
    {
        std::vector<double*> mutable_epsilon_(KNN_NEIGHBOURS);
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
            mutable_epsilon_[i] = &(parameters_[index_list[i] * 6]); // Blocks of 6
        return mutable_epsilon_;
    }
    double *mutable_params()
    {
        return parameters_;
    }

    const double *params() const
    {
        return parameters_;
    }

    void updateWarp()
    {
        for(int i = 0; i < warpField_->getNodes()->size() * 6; i+=6)
        {
            //增量式更新, We use Euler angle 
            float tx, ty, tz, ta, tb, tc;
            warpField_->getNodes()->at(i / 6).transform.getTranslation(tx, ty, tz);
            warpField_->getNodes()->at(i / 6).transform.getEuler(ta,tb,tc);
            //
            warpField_->getNodes()->at(i / 6).transform.encodeRotation(parameters_[i]*0.0174533 + ta, parameters_[i+1]*0.0174533 + tb, parameters_[i+2]*0.0174533 + tc);
            warpField_->getNodes()->at(i / 6).transform.encodeTranslation(parameters_[i+3]+tx,parameters_[i+4]+ty,parameters_[i+5]+tz);
            // std::cout<<i/6<<","<<parameters_[i]<<", "<<parameters_[i+1]<<", "<<parameters_[i+2]<<", "<<parameters_[i+3]<<", "<<parameters_[i+5]<<", "<<parameters_[i+5]<<std::endl;
        }
        std::cout<<"updated"<<std::endl;
    }

private:
    double *parameters_;
    kfusion::WarpField *warpField_;
};

#endif //KFUSION_OPTIMISATION_H
