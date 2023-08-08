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

#include "fusion/DepthMap.hpp"

namespace fusion
{
__global__ static void bilateralFilterKernel(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    const size_t w,
    const size_t h,
    const size_t srcStride,
    const size_t dstStride,
    const float sigmaSpace,
    const float sigmaColor)
{
  static constexpr size_t D = 13;
  static constexpr size_t k_h = D / 2;
  static constexpr size_t k_w = D / 2;

  const float sigma_space2_inv_half = 0.5f / (sigmaSpace * sigmaSpace);
  const float sigma_color2_inv_half = 0.5f / (sigmaColor * sigmaColor);

  for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
  {
    for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
    {
      int value = src[i * srcStride + j];
      float sum1 = 0.0f;
      float sum2 = 0.0f;
#pragma unroll
      for(size_t ii = 0; ii < D; ii++)
      {
#pragma unroll
        for(size_t jj = 0; jj < D; jj++)
        {
          const int iIndex = (i + ii - k_h);
          const int jIndex = (j + jj - k_w);

          if(iIndex < 0 || iIndex >= h || jIndex < 0 || jIndex >= h)
          {
            continue;
          }

          int tmp = src[iIndex * srcStride + jIndex];
          const float dc = value - tmp;
          const float dx = (float) jj - (float) k_w;
          const float dy = (float) ii - (float) k_h;
          const float space2 = dx * dx + dy * dy;
          const float color2 = dc * dc;
          const float weight =
              std::exp(-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

          sum1 += tmp * weight;
          sum2 += weight;
        }
      }

      uint16_t res(sum1 / sum2);
      dst[i * dstStride + j] = res;
    }
  }
}
__global__ static void initRgbdFrameKernel(
    const uint8_t* __restrict__ rgb,
    const uint16_t* __restrict__ depth,
    geometry::Vec3<uint8_t>* __restrict__ dstRgb,
    uint16_t* __restrict__ dstDepth,
    const size_t w,
    const size_t h,
    const size_t rgbStride,
    const size_t depthStride)
{
  for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
  {
    for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
    {
      dstDepth[i * depthStride + j] = depth[i * w + j];
      dstRgb[i * rgbStride + j] = geometry::Vec3<uint8_t>{
          rgb[3 * (i * w + j)], rgb[3 * (i * w + j) + 1], rgb[3 * (i * w + j) + 2]};
    }
  }
}
template <typename T>
__global__ static void copyImage(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const size_t w,
    const size_t h,
    const size_t srcStride,
    const size_t dstStride)
{
  for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
  {
    for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
    {
      dst[i * dstStride + j] = src[i * srcStride + j];
    }
  }
}

__global__ static void extractColorsKernelRGB(
    const geometry::Vec3<uint8_t>* __restrict__ src,
    geometry::Vec3f* __restrict__ dst,
    const size_t w,
    const size_t h,
    const size_t stride)
{
  for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
  {
    for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
    {
      dst[i * w + j] = (1.0f / 255.0f) * geometry::Vec3f(src[i * stride + j]);
    }
  }
}
__global__ static void extractColorsKernelBGR(
    const geometry::Vec3<uint8_t>* __restrict__ src,
    geometry::Vec3f* __restrict__ dst,
    const size_t w,
    const size_t h,
    const size_t stride)
{
  for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
  {
    for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
    {
      const auto& c = src[i * stride + j];
      dst[i * w + j] = (1.0f / 255.0f) * geometry::Vec3f(c.z, c.y, c.x);
    }
  }
}

__global__ static void extractPointsKernel(
    const uint16_t* __restrict__ depth,
    geometry::Vec3f* __restrict__ points,
    uint8_t* __restrict__ masks,
    const size_t w,
    const size_t h,
    const size_t stride,
    const double scale,
    const double near,
    const double far,
    const geometry::Mat3d k,
    const geometry::Mat4d m)
{
  const auto invM = geometry::Mat4d::Inverse(m);
  const double cx = k.c02;
  const double cy = k.c12;
  const double invFx = 1.0 / k.c00;
  const double invFy = 1.0 / k.c11;

  for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
  {
    for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
    {
      const uint16_t d = depth[i * stride + j];
      const double z = double(d) / scale;
      const bool mask = (d > 0) && ((z > near) || (z < far));
      if(mask)
      {
        const double x = (double(j) - cx) * (z * invFx);
        const double y = (double(i) - cy) * (z * invFy);
        points[i * w + j] = invM * geometry::Vec3d{x, y, z};
      }
      else
      {
        points[i * w + j] = geometry::Vec3f{0};
      }
      masks[i * w + j] = uint8_t(mask);
    }
  }
}

RGBDFrame::RGBDFrame(const size_t width, const size_t height)
    : width_{width}, height_{height}, rgbData_{width_, height_}, depthData_{width_, height_}
{}
RGBDFrame::RGBDFrame(const RGBDFrame& cp)
    : width_{cp.width_}, height_{cp.height_}, rgbData_{width_, height_}, depthData_{width_, height_}
{
  copyImage<<<1024, 32>>>(
      cp.rgbData_.data(), rgbData_.data(), width_, height_, cp.rgbData_.stride(),
      rgbData_.stride());
  copyImage<<<1024, 32>>>(
      cp.depthData_.data(), depthData_.data(), width_, height_, cp.depthData_.stride(),
      depthData_.stride());
}
RGBDFrame& RGBDFrame::operator=(const RGBDFrame& cp)
{
  width_ = cp.width_;
  height_ = cp.height_;
  rgbData_.resize(width_, height_);
  depthData_.resize(width_, height_);

  copyImage<<<1024, 32>>>(
      cp.rgbData_.data(), rgbData_.data(), width_, height_, cp.rgbData_.stride(),
      rgbData_.stride());
  copyImage<<<1024, 32>>>(
      cp.depthData_.data(), depthData_.data(), width_, height_, cp.depthData_.stride(),
      depthData_.stride());

  return *this;
}

void RGBDFrame::resize(const size_t w, const size_t h)
{
  width_ = w;
  height_ = h;
  rgbData_.resize(w, h);
  depthData_.resize(w, h);
}

void RGBDFrame::init(
    const GpuPtr<uint8_t>& rgb, const GpuPtr<uint16_t>& depth, const cudaStream_t& stream)
{
  initRgbdFrameKernel<<<1024, 32, 0, stream>>>(
      rgb.data(), depth.data(), rgbData_.data(), depthData_.data(), width_, height_,
      rgbData_.stride(), depthData_.stride());
}

void RGBDFrame::filterDepth(
    const float sigmaS,
    const float sigmaC,

    GpuImg<uint16_t>& tmp,
    const cudaStream_t& stream)
{
  bilateralFilterKernel<<<dim3(32, 32, 1), dim3(8, 4, 1), 0, stream>>>(
      depthData_.data(), tmp.data(), width_, height_, depthData_.stride(), tmp.stride(), sigmaS,
      sigmaC);
  copyImage<<<1024, 32, 0, stream>>>(
      tmp.data(), depthData_.data(), width_, height_, tmp.stride(), depthData_.stride());
}

void RGBDFrame::copy(RGBDFrame& dst, const cudaStream_t& stream) const
{
  if(dst.width_ != width_ || dst.height_ != height_)
  {
    throw std::runtime_error("Error when copying images, sizes mismatch");
  }
  copyImage<<<1024, 32, 0, stream>>>(
      rgbData_.data(), dst.rgbData_.data(), width_, height_, rgbData_.stride(),
      dst.rgbData_.stride());
  copyImage<<<1024, 32, 0, stream>>>(
      depthData_.data(), dst.depthData_.data(), width_, height_, depthData_.stride(),
      dst.depthData_.stride());
}

void RGBDFrame::extractPoints(
    OrderedPointCloud& dst,
    const double scale,
    const geometry::Mat3d& intrinsics,
    const geometry::Mat4d& extrinsics,
    const double near,
    const double far,
    const ColorFormat format,
    const cudaStream_t& stream)
{
  switch(format)
  {
    case ColorFormat::RGB:
      extractColorsKernelRGB<<<1024, 32, 0, stream>>>(
          rgbData_.data(), dst.colors(), width_, height_, rgbData_.stride());
      break;
    case ColorFormat::BGR:
      extractColorsKernelBGR<<<1024, 32, 0, stream>>>(
          rgbData_.data(), dst.colors(), width_, height_, rgbData_.stride());
      break;
    default:
      throw(std::runtime_error("Unknown color format"));
      break;
  }
  extractPointsKernel<<<1024, 32, 0, stream>>>(
      depthData_.data(), dst.points(), dst.masks(), width_, height_, depthData_.stride(), scale,
      near, far, intrinsics, extrinsics);
}

void RGBDFrame::exportPNG(const std::string& /*filename*/)
{
  throw std::runtime_error("RGBDFrame::exportPNG() not implemented yet");
}
} // namespace fusion