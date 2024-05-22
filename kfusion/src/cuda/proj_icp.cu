#include "device.hpp"
#include "texture_binder.hpp"
#include <stdio.h>

namespace kfusion
{
    namespace device
    {
        texture<ushort, 2> dprev_tex;
        texture<Normal, 2> nprev_tex;
        texture<Point,  2> vprev_tex;

        struct ComputeIcpHelper::Policy
        {
            enum
            {
                CTA_SIZE_X = 32,
                CTA_SIZE_Y = 8,
                CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

                B = 6, COLS = 6, ROWS = 6, DIAG = 6,
                UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG,
                TOTAL = UPPER_DIAG_MAT + B,

                FINAL_REDUCE_CTA_SIZE = 256,
                FINAL_REDUCE_STRIDE = FINAL_REDUCE_CTA_SIZE
            };
        };

        __kf_device__
        float2 ComputeIcpHelper::proj(const float3& p) const
        {
            float2 coo;
            coo.x = __fmaf_rn(f.x, __fdividef(p.x, p.z), c.x);
            coo.y = __fmaf_rn(f.y, __fdividef(p.y, p.z), c.y);
            return coo;
        }

        __kf_device__
        float3 ComputeIcpHelper::reproj(float u, float v, float z)  const
        {
            float x = z * (u - c.x) * finv.x;
            float y = z * (v - c.y) * finv.y;
            return make_float3(x, y, z);
        }

#if defined USE_DEPTH
        __kf_device__
        int ComputeIcpHelper::find_coresp(int x, int y, float3& nd, float3& d, float3& s) const
        {
            int src_z = dcurr(y, x);
            if (src_z == 0)
                return 40;

            s = aff * reproj(x, y, src_z * 0.001f);

            float2 coo = proj(s);
            if (s.z <= 0 || coo.x < 0 || coo.y < 0 || coo.x >= cols || coo.y >= rows)
                return 80;

            int dst_z = tex2D(dprev_tex, coo.x, coo.y);
            if (dst_z == 0)
                return 120;

            d = reproj(coo.x, coo.y, dst_z * 0.001f);

            float dist2 = norm_sqr(s - d);
            if (dist2 > dist2_thres)
                return 160;

            float3 ns = aff.R * tr(ncurr(y, x));
            nd = tr(tex2D(nprev_tex, coo.x, coo.y));

            float cosine = fabs(dot(ns, nd));
            if (cosine < min_cosine)
                return 200;
            return 0;
        }
#else
        // 通过将voxel的体素投影到深度图像平面上，找到对应点，并判断法向是否接近，一次判断是否为对应点
        __kf_device__
        int ComputeIcpHelper::find_coresp(int x, int y, float3& nd, float3& d, float3& s) const //d, and s is the correspond locations at previous and current map
        {
            s = tr(vcurr(y, x));//Get the coor of (y,z)
            if (isnan(s.x))
                return 40;

            s = aff * s; // Transform to current camera coor

            float2 coo = proj(s); //project to current depth map/points map
            if (s.z <= 0 || coo.x < 0 || coo.y < 0 || coo.x >= cols || coo.y >= rows)
                return 80;

            d = tr(tex2D(vprev_tex, coo.x, coo.y)); //Get the previouos point coor at coo
            if (isnan(d.x))
                return 120;

            float dist2 = norm_sqr(s - d); //distance of s and d
            if (dist2 > dist2_thres)
                return 160;

            float3 ns = aff.R * tr(ncurr(y, x));
            nd = tr(tex2D(nprev_tex, coo.x, coo.y));

            float cosine = fabs(dot(ns, nd)); // similarity of normals
            if (cosine < min_cosine)
                return 200;
            return 0;
        }
#endif

        __kf_device__
        void ComputeIcpHelper::partial_reduce(const float row[7], PtrStep<float>& partial_buf) const
        {
            volatile __shared__ float smem[Policy::CTA_SIZE];
            int tid = Block::flattenedThreadId ();

            float  *pos = partial_buf.data + blockIdx.x + gridDim.x * blockIdx.y;
            size_t step = partial_buf.step / sizeof(float);

#define STOR \
            if (tid == 0) \
            { \
                *pos = smem[0]; \
                pos += step; \
            }


            __syncthreads ();
            smem[tid] = row[0] * row[0];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[0] * row[1];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[0] * row[2];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[0] * row[3];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[0] * row[4];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR
            __syncthreads ();
            smem[tid] = row[0] * row[5];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[0] * row[6];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR
////////////////////////////////

            __syncthreads ();
            smem[tid] = row[1] * row[1];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[1] * row[2];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[1] * row[3];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[1] * row[4];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[1] * row[5];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[1] * row[6];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR
////////////////////////////////

            __syncthreads ();
            smem[tid] = row[2] * row[2];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[2] * row[3];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[2] * row[4];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[2] * row[5];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[2] * row[6];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR
////////////////////////////////

            __syncthreads ();
            smem[tid] = row[3] * row[3];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[3] * row[4];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[3] * row[5];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[3] * row[6];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR
///////////////////////////////////////////////////

            __syncthreads ();
            smem[tid] = row[4] * row[4];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[4] * row[5];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[4] * row[6];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

        STOR

///////////////////////////////////////////////////

            __syncthreads ();
            smem[tid] = row[5] * row[5];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR

            __syncthreads ();
            smem[tid] = row[5] * row[6];
            __syncthreads ();

            Block::reduce<Policy::CTA_SIZE>(smem, plus ());

         STOR
        }

        __global__ void icp_helper_kernel(const ComputeIcpHelper helper, PtrStep<float> partial_buf, int *valid_nums)
        {
            int x = threadIdx.x + blockIdx.x * ComputeIcpHelper::Policy::CTA_SIZE_X;
            int y = threadIdx.y + blockIdx.y * ComputeIcpHelper::Policy::CTA_SIZE_Y;

            float3 n, d, s; //缓存
            // 只是按照投影方式寻找对应点，对应点不一定是对的！！！
            int filtered = (x < helper.cols && y < helper.rows) ? helper.find_coresp (x, y, n, d, s) : 1; //d为前一帧对应点坐标，s为当前帧对应点坐标，返回0表示两者存在对应关系
            //if (x < helper.cols && y < helper.rows) mask(y, x) = filtered;
            // find correspondence : filtered == 0
            float row[7];

            if (!filtered) //存在对应关系
            {
                *(float3*)&row[0] = cross (s, n);  // 叉积，A矩阵及b并列，计算方式见点到面ICP算法
                *(float3*)&row[3] = n;
                row[6] = dot (n, d - s);
                // atomicAdd(valid_nums, 1); //计算对应点数量，调试用
            }
            else
                row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

            helper.partial_reduce(row, partial_buf);
        }

        __global__ void icp_final_reduce_kernel(const PtrStep<float> partial_buf, const int length, float* final_buf)
        {
            const float *beg = partial_buf.ptr(blockIdx.x);
            const float *end = beg + length;

            int tid = threadIdx.x;

            float sum = 0.f;
            for (const float *t = beg + tid; t < end; t += ComputeIcpHelper::Policy::FINAL_REDUCE_STRIDE)
                sum += *t;

            __shared__ float smem[ComputeIcpHelper::Policy::FINAL_REDUCE_CTA_SIZE];

            smem[tid] = sum;

            __syncthreads();

            Block::reduce<ComputeIcpHelper::Policy::FINAL_REDUCE_CTA_SIZE>(smem, plus());

            if (tid == 0)
            {
                final_buf[blockIdx.x] = smem[0];
            }
        }
    }
}
// 计算对应点，并获取估计平移和旋转的矩阵，SVD--
void kfusion::device::ComputeIcpHelper::operator()(const Depth& dprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t s)
{
    dprev_tex.filterMode = cudaFilterModePoint;
    nprev_tex.filterMode = cudaFilterModePoint;
    TextureBinder dprev_binder(dprev, dprev_tex);
    TextureBinder nprev_binder(nprev, nprev_tex);

    dim3 block(Policy::CTA_SIZE_X, Policy::CTA_SIZE_Y);
    dim3 grid(divUp ((int)cols, block.x), divUp ((int)rows, block.y));

    int partials_count = (int)(grid.x * grid.y);
    allocate_buffer(buffer, partials_count);
    int *valid_nums;
    cudaMalloc(&valid_nums, sizeof(int));
    icp_helper_kernel<<<grid, block, 0, s>>>(*this, buffer,valid_nums);
    cudaSafeCall ( cudaGetLastError () );

    int b = Policy::FINAL_REDUCE_CTA_SIZE;
    int g = Policy::TOTAL;
    icp_final_reduce_kernel<<<g, b, 0, s>>>(buffer, partials_count, buffer.ptr(Policy::TOTAL));
    cudaSafeCall ( cudaGetLastError () );

    cudaSafeCall ( cudaMemcpyAsync(data, buffer.ptr(Policy::TOTAL), Policy::TOTAL * sizeof(float), cudaMemcpyDeviceToHost, s) );
    cudaSafeCall ( cudaGetLastError () );
}
// // 计算Ax=b 
// void kfusion::device::ComputeIcpHelper::operator()(const Points& vprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t s)
// {
//     dprev_tex.filterMode = cudaFilterModePoint;
//     nprev_tex.filterMode = cudaFilterModePoint;
//     TextureBinder vprev_binder(vprev, vprev_tex);
//     TextureBinder nprev_binder(nprev, nprev_tex);

//     dim3 block(Policy::CTA_SIZE_X, Policy::CTA_SIZE_Y);
//     dim3 grid(divUp ((int)cols, block.x), divUp ((int)rows, block.y));

//     int partials_count = (int)(grid.x * grid.y);
//     allocate_buffer(buffer, partials_count);

//     icp_helper_kernel<<<grid, block, 0, s>>>(*this, buffer);
//     cudaSafeCall ( cudaGetLastError () );

//     int b = Policy::FINAL_REDUCE_CTA_SIZE;
//     int g = Policy::TOTAL;
//     icp_final_reduce_kernel<<<g, b, 0, s>>>(buffer, partials_count, buffer.ptr(Policy::TOTAL));
//     cudaSafeCall ( cudaGetLastError () );

//     cudaSafeCall ( cudaMemcpyAsync(data, buffer.ptr(Policy::TOTAL), Policy::TOTAL * sizeof(float), cudaMemcpyDeviceToHost, s) );
//     cudaSafeCall ( cudaGetLastError () );
// }

//计算Ax = b，先获取矩阵，再通过矩阵乘法获取ATAx=ATb
void kfusion::device::ComputeIcpHelper::operator()(const Points& vprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t s)
{
    dprev_tex.filterMode = cudaFilterModePoint;
    nprev_tex.filterMode = cudaFilterModePoint;
    TextureBinder vprev_binder(vprev, vprev_tex);
    TextureBinder nprev_binder(nprev, nprev_tex);

    dim3 block(Policy::CTA_SIZE_X, Policy::CTA_SIZE_Y);
    dim3 grid(divUp ((int)cols, block.x), divUp ((int)rows, block.y));

    int partials_count = (int)(grid.x * grid.y);
    allocate_buffer(buffer, partials_count);
    int *valid_nums;
    int init_sum = 0;
    cudaMalloc(&valid_nums, sizeof(int));
    cudaMemcpy(valid_nums, &init_sum, sizeof(int), cudaMemcpyHostToDevice);
    icp_helper_kernel<<<grid, block, 0, s>>>(*this, buffer,valid_nums);
    cudaSafeCall ( cudaGetLastError () );
    int result;
    cudaMemcpy(&result, valid_nums, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("valid_nums: %d\n",result);
    cudaFree(valid_nums);
    int b = Policy::FINAL_REDUCE_CTA_SIZE;
    int g = Policy::TOTAL;
    icp_final_reduce_kernel<<<g, b, 0, s>>>(buffer, partials_count, buffer.ptr(Policy::TOTAL));
    cudaSafeCall ( cudaGetLastError () );

    cudaSafeCall ( cudaMemcpyAsync(data, buffer.ptr(Policy::TOTAL), Policy::TOTAL * sizeof(float), cudaMemcpyDeviceToHost, s) );
    cudaSafeCall ( cudaGetLastError () );
}


void kfusion::device::ComputeIcpHelper::allocate_buffer(DeviceArray2D<float>& buffer, int partials_count)
{ 
    if (partials_count < 0)
    {
        const int input_cols = 640;
        const int input_rows = 480;

        int gx = divUp (input_cols, Policy::CTA_SIZE_X);
        int gy = divUp (input_rows, Policy::CTA_SIZE_Y);

        partials_count = gx * gy;
    }

    int min_rows = Policy::TOTAL + 1;
    int min_cols = max(partials_count, Policy::TOTAL);

    if (buffer.rows() < min_rows || buffer.cols() < min_cols)
        buffer.create (min_rows, min_cols);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ComputeIcpHelper::PageLockHelper

kfusion::device::ComputeIcpHelper::PageLockHelper::PageLockHelper() : data(0)
{ cudaSafeCall( cudaMallocHost((void **)&data, Policy::TOTAL * sizeof(float)) );  }

kfusion::device::ComputeIcpHelper::PageLockHelper::~PageLockHelper()
{   cudaSafeCall( cudaFreeHost(data) ); data = 0; }
