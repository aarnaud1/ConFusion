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

#include "fusion/OrderedPointCloud.hpp"
#include "io/Ply.hpp"
#include "common.inl"

namespace fusion
{
__global__ static void transformPointCloudKernel(
    geometry::Vec3f* __restrict__ points,
    geometry::Vec3f* __restrict__ normals,
    uint8_t* __restrict__ masks,
    const geometry::Mat4d& m,
    const size_t n)
{
  for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
  {
    if(masks[idx])
    {
      points[idx] = m * geometry::Vec4d(points[idx], 1.0);
      normals[idx] = m.GetRotation() * geometry::Vec3d(normals[idx]);
    }
  }
}

__global__ static void estimateNormalsKernel(
    geometry::Vec3f* __restrict__ points,
    geometry::Vec3f* __restrict__ normals,
    uint8_t* __restrict__ masks,
    const size_t w,
    const size_t h)
{
  for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
  {
    for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
    {
      const size_t index = i * w + j;
      if(!masks[index])
      {
        continue;
      }

      auto n = geometry::Vec3f{0};
      size_t count = 0;

      const int index00 = (j < (int(w) - 1)) ? i * int(w) + j + 1 : -1;
      const int index01 = (i > 0) ? (i - 1) * int(w) + j : -1;
      const int index10 = (j > 0) ? i * int(w) + j - 1 : -1;
      const int index11 = (i < (h - 1)) ? (i + 1) * int(w) + j : -1;

      if(index00 >= 0 && index01 >= 0 && masks[index00] && masks[index01])
      {
        n += geometry::Vec3f::Cross(
            points[index00] - points[index], points[index01] - points[index]);
        count++;
      }
      if(index01 >= 0 && index10 >= 0 && masks[index01] && masks[index10])
      {
        n += geometry::Vec3f::Cross(
            points[index01] - points[index], points[index10] - points[index]);
        count++;
      }
      if(index10 >= 0 && index11 >= 0 && masks[index10] && masks[index11])
      {
        n += geometry::Vec3f::Cross(
            points[index10] - points[index], points[index11] - points[index]);
        count++;
      }
      if(index11 >= 0 && index00 >= 0 && masks[index11] && masks[index00])
      {
        n += geometry::Vec3f::Cross(
            points[index11] - points[index], points[index00] - points[index]);
        count++;
      }
      normals[index] = (count > 0) ? geometry::Vec3f::Normalize(n) : geometry::Vec3f{0};
    }
  }
}

OrderedPointCloud::OrderedPointCloud(const size_t width, const size_t height)
    : width_{width}
    , height_{height}
    , points_{width_ * height_}
    , normals_{width_ * height_}
    , colors_{width_ * height_}
    , masks_{width_ * height_}
{}

void OrderedPointCloud::resize(const size_t width, const size_t height)
{
  width_ = width;
  height_ = height;
  const size_t res = width_ * height_;
  points_.resize(res);
  normals_.resize(res);
  colors_.resize(res);
  masks_.resize(res);
}

void OrderedPointCloud::copy(OrderedPointCloud& dst, const cudaStream_t& stream)
{
  if(dst.width() != width_ && dst.height() != height_)
  {
    throw std::runtime_error("Src and dst cloud sizes mismatch");
  }
  points_.copyTo(dst.points_, stream);
  colors_.copyTo(dst.colors_, stream);
  normals_.copyTo(dst.normals_, stream);
  masks_.copyTo(dst.masks_, stream);
}

void OrderedPointCloud::transform(const geometry::Mat4d& m, const cudaStream_t& stream)
{
  transformPointCloudKernel<<<1024, 32, 0, stream>>>(
      points_, normals_, masks_, m, width_ * height_);
}

void OrderedPointCloud::estimateNormals(const cudaStream_t& stream)
{
  estimateNormalsKernel<<<dim3(32, 32, 1), dim3(8, 8, 1), 0, stream>>>(
      points_, normals_, masks_, width_, height_);
}

// void OrderedPointCloud::render(Surface& surf, const cudaStream_t& stream) {}

void OrderedPointCloud::exportPLY(const std::string& filename, const cudaStream_t& stream)
{
  CpuPtr<geometry::Vec3f, true> pointsCpu(points_.size());
  CpuPtr<geometry::Vec3f, true> colorsCpu(colors_.size());
  CpuPtr<geometry::Vec3f, true> normalsCpu(normals_.size());
  CpuPtr<uint8_t, true> masksCpu(masks_.size());

  // TODO : make it more async
  pointsCpu.downloadFrom(points_, stream);
  colorsCpu.downloadFrom(colors_, stream);
  normalsCpu.downloadFrom(normals_, stream);
  masksCpu.downloadFrom(masks_, stream);
  cudaStreamSynchronize(stream);

  std::vector<geometry::Vec3f> xyz;
  std::vector<geometry::Vec3<uint8_t>> rgb;
  std::vector<geometry::Vec3f> norm;
  xyz.reserve(points_.size());
  rgb.reserve(colors_.size());
  norm.reserve(normals_.size());
  for(size_t i = 0; i < points_.size(); ++i)
  {
    if(masksCpu[i])
    {
      xyz.emplace_back(pointsCpu[i]);
      norm.emplace_back(normalsCpu[i]);
      rgb.emplace_back(255.0f * colorsCpu[i]);
    }
  }

  io::Ply::savePoints(filename, xyz, rgb, norm);
}
} // namespace fusion