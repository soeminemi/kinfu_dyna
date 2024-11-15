#include "precomp.hpp"
#include "internal.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Kinfu/types implementation

kfusion::Intr::Intr () {}
kfusion::Intr::Intr (float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {
  // fx = fy = cx = cy = 0;
  k1 = k2 = p1 = p2 = 0;  
}
kfusion::Intr::Intr (float fx_, float fy_, float cx_, float cy_, float k1_, float k2_, float k3_, float p1_, float p2_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_), k1(k1_), k2(k2_), k3(k3_), p1(p1_), p2(p2_) {}
      
kfusion::Intr kfusion::Intr::operator()(int level_index) const
{
  int div = 1 << level_index;
  return (Intr (fx / div, fy / div, cx / div, cy / div, k1, k2 ,k3, p1, p2));
}

std::ostream& operator << (std::ostream& os, const kfusion::Intr& intr)
{
  return os << "([f = " << intr.fx << ", " << intr.fy << "] [cp = " << intr.cx << ", " << intr.cy << "])";
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume host implementation

kfusion::device::TsdfVolume::TsdfVolume(elem_type* _data, int3 _dims, float3 _voxel_size, float _trunc_dist, int _max_weight)
: data(_data), dims(_dims), voxel_size(_voxel_size), trunc_dist(_trunc_dist), max_weight(_max_weight) {}

kfusion::device::TsdfVolume::TsdfVolume(const TsdfVolume& other)
  : data(other.data), dims(other.dims), voxel_size(other.voxel_size), trunc_dist(other.trunc_dist), max_weight(other.max_weight) {}

//kfusion::device::TsdfVolume::elem_type* kfusionl::device::TsdfVolume::operator()(int x, int y, int z)
//{ return data + x + y*dims.x + z*dims.y*dims.x; }
//
//const kfusion::device::TsdfVolume::elem_type* kfusionl::device::TsdfVolume::operator() (int x, int y, int z) const
//{ return data + x + y*dims.x + z*dims.y*dims.x; }
//
//kfusion::device::TsdfVolume::elem_type* kfusionl::device::TsdfVolume::beg(int x, int y) const
//{ return data + x + dims.x * y; }
//
//kfusion::device::TsdfVolume::elem_type* kfusionl::device::TsdfVolume::zstep(elem_type *const ptr) const
//{ return data + dims.x * dims.y; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Projector host implementation

kfusion::device::Projector::Projector(float fx, float fy, float cx, float cy) : f(make_float2(fx, fy)), c(make_float2(cx, cy)) 
{
  k.x = 0;
  k.y = 0;
  k.z = 0;
}

kfusion::device::Projector::Projector(float fx, float fy, float cx, float cy, float k1, float k2, float k3) : f(make_float2(fx, fy)), c(make_float2(cx, cy)), k(make_float3(k1, k2, k3)) {}

//float2 kfusion::device::Projector::operator()(const float3& p) const
//{
//  float2 coo;
//  coo.x = p.x * f.x / p.z + c.x;
//  coo.y = p.y * f.y / p.z + c.y;
//  return coo;
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Reprojector host implementation

kfusion::device::Reprojector::Reprojector(float fx, float fy, float cx, float cy) : finv(make_float2(1.f/fx, 1.f/fy)), c(make_float2(cx, cy)) 
{
  k.x = 0;k.y = 0;k.z=0;
}
kfusion::device::Reprojector::Reprojector(float fx, float fy, float cx, float cy, float k1, float k2, float k3) :  finv(make_float2(1.f/fx, 1.f/fy)), c(make_float2(cx, cy)) {
  k.x = k1;
  k.y = k2;
  k.z = k3;
}
//float3 kfusion::device::Reprojector::operator()(int u, int v, float z) const
//{
//  float x = z * (u - c.x) * finv.x;
//  float y = z * (v - c.y) * finv.y;
//  return make_float3(x, y, z);
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Host implementation of packing/unpacking tsdf volume element

//ushort2 kfusion::device::pack_tsdf(float tsdf, int weight) { throw "Not implemented"; return ushort2(); }
//float kfusion::device::unpack_tsdf(ushort2 value, int& weight) { throw "Not implemented"; return 0; }
//float kfusion::device::unpack_tsdf(ushort2 value) { throw "Not implemented"; return 0; }

