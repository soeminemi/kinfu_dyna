#pragma once

#include <kfusion/types.hpp>
#include <kfusion/utils/dual_quaternion.hpp>

namespace kfusion
{
    class WarpField;
    namespace cuda
    {
        class KF_EXPORTS TsdfVolume
        {
        public:
            TsdfVolume(const cv::Vec3i& dims);
            virtual ~TsdfVolume();

            void create(const Vec3i& dims);

            Vec3i getDims() const;
            Vec3f getVoxelSize() const;

            const CudaData data() const;
            CudaData data();

            Vec3f getSize() const;
            void setSize(const Vec3f& size);

            float getTruncDist() const;
            void setTruncDist(float distance);

            void setDepthScale(float depthScale);
            float getDepthScale() const;

            int getMaxWeight() const;
            void setMaxWeight(int weight);

            Affine3f getPose() const;
            void setPose(const Affine3f& pose);

            float getRaycastStepFactor() const;
            void setRaycastStepFactor(float factor);

            float getGradientDeltaFactor() const;
            void setGradientDeltaFactor(float factor);

            Vec3i getGridOrigin() const;
            void setGridOrigin(const Vec3i& origin);
            
            //--- For dynamic fusion
            std::vector<float> psdf(const std::vector<Vec3f>& warped, Dists& depth_img, const Intr& intr);
//            float psdf(const std::vector<Vec3f>& warped, Dists& dists, const Intr& intr);
            float weighting(const std::vector<float>& dist_sqr, int k) const;
            void surface_fusion(WarpField& warp_field,
                                std::vector<Vec3f> warped,
                                std::vector<Vec3f> canonical,
                                cuda::Depth &depth,
                                const Affine3f& camera_pose,
                                const Intr& intr);
            //--- END For dynamic fusion
            virtual void clear();
            virtual void applyAffine(const Affine3f& affine);
            virtual void integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr);
            virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals);
            virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Cloud& points, Normals& normals);

            void swap(CudaData& data);

            DeviceArray<Point> fetchCloud(DeviceArray<Point>& cloud_buffer) const;
            void fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<Normal>& normals) const;
            void computePoints(cv::Mat &init_frame);
            void compute_points();
            void compute_normals();
            void get_points(cv::Mat &points_mat);

            struct Entry
            {
                typedef unsigned short half;

                half tsdf;
                unsigned short weight;

                static float half2float(half value);
                static half float2half(float value);
            };
        private:
            CudaData data_;

            float trunc_dist_;
            int max_weight_;
            Vec3i dims_;
            Vec3f size_;
            Affine3f pose_;
            float depth_scale_;

            float gradient_delta_factor_;
            float raycast_step_factor_;

            cuda::DeviceArray<Point> *cloud_buffer_;
            cuda::DeviceArray<Point> *normal_buffer_;
            cuda::DeviceArray<Point> *cloud_;
            cv::Mat *cloud_host_;
            cv::Mat *normal_host_;
        };
    }
}
