/*
 * Copyright (C) 2023 Adrien ARNAUD
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "fusion/RenderDepthMap.hpp"
#include "io/Ply.hpp"
#include "../utils/Ptr.inl"

#include <vector>

namespace fusion
{
__device__ __forceinline__ static geometry::Vec3f getPoint(
    const uint16_t depth,
    const size_t u,
    const size_t v,
    const double scale,
    const float cx,
    const float cy,
    const float fx,
    const float fy)
{
  const double z = double(depth) / scale;
  const double x = (u - cx) * z / double(fx);
  const double y = (v - cy) * z / double(fy);
  return geometry::Vec3f{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
}

__global__ static void extractPointsKernel(
    const uint16_t* __restrict__ depth,
    geometry::Vec3f* __restrict__ points,
    float* __restrict__ footprints,
    const float scale,
    const geometry::Mat3d k,
    const size_t w,
    const size_t h,
    const size_t stride)
{
  const double cx = k.c02;
  const double cy = k.c12;
  const double fx = k.c00;
  const double fy = k.c11;
  for(size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
  {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
    {
      const uint16_t d = depth[i * stride + j];
      if(d > 0)
      {
        const auto p0 = getPoint(d, j, i, scale, cx, cy, fx, fy);
        const auto p1 = getPoint(d, j + 1, i + 1, scale, cx, cy, fx, fy);
        footprints[i * w + j] = geometry::Vec3f::Dist(p0, p1);
        points[i * w + j] = p0;
      }
      else
      {
        footprints[i * w + j] = std::numeric_limits<float>::max();
        points[i * w + j] = geometry::Vec3f{std::numeric_limits<float>::max()};
      }
    }
  }
}
__global__ static void renderDepthMapKernel(
    const geometry::Vec3f* __restrict__ points,
    const float* __restrict__ footprints,
    geometry::Vec3i* __restrict__ triangles,
    int* __restrict__ triangleCount,
    const float maxDisparity,
    const size_t w,
    const size_t h)
{
  for(size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < w - 1; j += blockDim.x * gridDim.x)
  {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < h - 1; i += blockDim.y * gridDim.y)
    {
      static constexpr float disparityThr = 10.0f;

      const int i0 = i * w + j;
      const int i1 = (i + 1) * w + j;
      const int i2 = (i + 1) * w + j + 1;
      const int i3 = i * w + j + 1;

      const auto p0 = points[i0];
      const auto p1 = points[i1];
      const auto p2 = points[i2];
      const auto p3 = points[i3];

      // p0, p1, p2 triangle
      if(geometry::all(geometry::notEqual(p0, geometry::Vec3f{std::numeric_limits<float>::max()}))
         && geometry::all(
             geometry::notEqual(p1, geometry::Vec3f{std::numeric_limits<float>::max()}))
         && geometry::all(
             geometry::notEqual(p2, geometry::Vec3f{std::numeric_limits<float>::max()})))
      {
        const float fp = footprints[i0];
        if((std::abs(p0.z - p1.z) <= disparityThr * fp)
           && (std::abs(p0.z - p2.z) <= disparityThr * fp)
           && (std::abs(p1.z - p2.z) <= disparityThr * fp))
        {
          const int index = atomicAdd(triangleCount, 1);
          triangles[index] = {i0, i1, i2};
        }
      }

      // p0, p2, p3 triangle
      if(geometry::all(geometry::notEqual(p0, geometry::Vec3f{std::numeric_limits<float>::max()}))
         && geometry::all(
             geometry::notEqual(p2, geometry::Vec3f{std::numeric_limits<float>::max()}))
         && geometry::all(
             geometry::notEqual(p3, geometry::Vec3f{std::numeric_limits<float>::max()})))
      {
        const float fp = footprints[i0];
        if((std::abs(p0.z - p2.z) <= disparityThr * fp)
           && (std::abs(p0.z - p3.z) <= disparityThr * fp)
           && (std::abs(p2.z - p3.z) <= disparityThr * fp))
        {
          const int index = atomicAdd(triangleCount, 1);
          triangles[index] = {i0, i2, i3};
        }
      }
    }
  }
}
__global__ static void estimateNormalsKernel(
    const geometry::Vec3f* __restrict__ points,
    geometry::Vec3f* __restrict__ normals,
    const geometry::Vec3i* __restrict__ triangles,
    const int* __restrict__ triangleCount)
{
  for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size_t(*triangleCount);
      idx += blockDim.x * gridDim.x)
  {
    const auto& t = triangles[idx];
    const auto& p0 = points[t.x];
    const auto& p1 = points[t.y];
    const auto& p2 = points[t.z];

    const auto n = geometry::Vec3f::Cross(p1 - p0, p2 - p0);
    atomicAdd(reinterpret_cast<float*>(&normals[t.x].x), n.x);
    atomicAdd(reinterpret_cast<float*>(&normals[t.x].y), n.y);
    atomicAdd(reinterpret_cast<float*>(&normals[t.x].z), n.z);

    atomicAdd(reinterpret_cast<float*>(&normals[t.y].x), n.x);
    atomicAdd(reinterpret_cast<float*>(&normals[t.y].y), n.y);
    atomicAdd(reinterpret_cast<float*>(&normals[t.y].z), n.z);

    atomicAdd(reinterpret_cast<float*>(&normals[t.z].x), n.x);
    atomicAdd(reinterpret_cast<float*>(&normals[t.z].y), n.y);
    atomicAdd(reinterpret_cast<float*>(&normals[t.z].z), n.z);
  }
}

RenderDepthMap::RenderDepthMap(const size_t width, const size_t height)
    : RGBDFrame{width, height}
    , triangles_{2 * (width_ - 1) * (height_ - 1)}
    , points_{width_ * height_}
    , normals_{width_ * height_}
    , footprints_{width_ * height_}
{
  size_.set(0, cudaStream_t{0});
}

void RenderDepthMap::resize(const size_t width, const size_t height)
{
  RGBDFrame::resize(width, height);
  triangles_.resize(2 * (width_ - 1) * 2 * (height_ - 1));
  points_.resize(width_ * height_);
  normals_.resize(width_ * height_);
  footprints_.resize(width_ * height_);
  size_.set(0, cudaStream_t{0});
}

void RenderDepthMap::render(
    const float scale, const geometry::Mat3d& intrinsics, const cudaStream_t& stream)
{
  size_.set(0, stream);
  extractPointsKernel<<<dim3(32, 32), dim3(8, 8), 0, stream>>>(
      depthData_.data(), points_, footprints_, scale, intrinsics, width_, height_,
      depthData_.stride());
  renderDepthMapKernel<<<dim3(32, 32), dim3(8, 8), 0, stream>>>(
      points_, footprints_, triangles_, size_, 0.0f, width_, height_);
}

void RenderDepthMap::estimateNormals(const cudaStream_t& stream)
{
  normals_.set(geometry::Vec3f(0.0), stream);
  estimateNormalsKernel<<<1024, 32, 0, stream>>>(points_, normals_, triangles_, size_);
}

void RenderDepthMap::exportPLY(const std::string& filename, const cudaStream_t& stream)
{
  CpuPtr<int32_t, true> size{1};
  CpuPtr<geometry::Vec3i, true> triangles{triangles_.size()};
  CpuPtr<geometry::Vec3f, true> points{points_.size()};
  CpuPtr<geometry::Vec3f, true> normals{normals_.size()};
  CpuPtr<float, true> footprints{footprints_.size()};

  // TODO : make it more async
  size.downloadFrom(size_, stream);
  triangles.downloadFrom(triangles_, stream);
  points.downloadFrom(points_, stream);
  normals.downloadFrom(normals_, stream);
  footprints.downloadFrom(footprints_, stream);
  gpuErrcheck(cudaStreamSynchronize(stream));

  std::vector<geometry::Vec3f> xyz;
  std::vector<geometry::Vec3<uint8_t>> rgb;
  std::vector<geometry::Vec3f> n;
  xyz.reserve(points_.size());
  rgb.reserve(points_.size());
  n.reserve(points_.size());

  constexpr float voxelSize = 0.005f;
  auto clamp = [](const float x, const float a, const float b) {
    return std::max(std::min(x, b), a);
  };
  auto getColor = [voxelSize, clamp](const float footprint) {
    auto val = clamp(footprint / voxelSize / 4.0f, 0.0f, 1.0f);
    return 255.0f * geometry::Vec3f{val, 1.0f - val, 0.0f};
  };

  for(size_t i = 0; i < points.size(); ++i)
  {
    xyz.emplace_back(points[i]);
    rgb.emplace_back(getColor(footprints[i]));
    n.emplace_back(geometry::Vec3f::Normalize(normals[i]));
  }

  std::vector<geometry::Vec3i> tris;
  tris.reserve(*size);
  for(size_t i = 0; i < size_t(*size); ++i)
  {
    const auto& t = triangles[i];
    tris.emplace_back(t);
  }

  io::Ply::saveSurface(filename, xyz, rgb, n, tris);
}
} // namespace fusion