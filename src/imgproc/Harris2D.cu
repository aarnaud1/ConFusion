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

#include "imgproc/Harris2D.hpp"
#include "imgProcCommon.inl"

namespace fusion
{
template <typename T>
__global__ __launch_bounds__(tileWidth* tileHeight) static void extractGradientsKernel(
    const T* __restrict__ inputImg,
    geometry::Vec2f* __restrict__ gradients,
    const size_t w,
    const size_t h,
    const size_t inputStride,
    const size_t outputStride)
{
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += blockDim.x * gridDim.x)
  {
    for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < w; j += blockDim.y * gridDim.y)
    {
      const T p = inputImg[i * inputStride + j];

      const float px00 = readPixel(inputImg, i - 1, j - 1, w, h, inputStride, p);
      const float px01 = readPixel(inputImg, i - 1, j + 1, w, h, inputStride, p);
      const float px10 = readPixel(inputImg, i, j - 1, w, h, inputStride, p);
      const float px11 = readPixel(inputImg, i, j + 1, w, h, inputStride, p);
      const float px20 = readPixel(inputImg, i + 1, j - 1, w, h, inputStride, p);
      const float px21 = readPixel(inputImg, i + 1, j + 1, w, h, inputStride, p);
      const float gx = 0.125f * ((px01 - px00) + 2.0f * (px11 - px10) + (px21 - px20));

      const float py00 = readPixel(inputImg, i - 1, j - 1, w, h, inputStride, p);
      const float py01 = readPixel(inputImg, i + 1, j - 1, w, h, inputStride, p);
      const float py10 = readPixel(inputImg, i - 1, j, w, h, inputStride, p);
      const float py11 = readPixel(inputImg, i + 1, j, w, h, inputStride, p);
      const float py20 = readPixel(inputImg, i - 1, j + 1, w, h, inputStride, p);
      const float py21 = readPixel(inputImg, i + 1, j + 1, w, h, inputStride, p);
      const float gy = 0.125f * ((py01 - py00) + 2.0f * (py11 - py10) + (py21 - py20));

      const float len = geometry::Vec2f{gx, gy}.Len();
      const float theta =
          (gx > 0.0f) ? ((gx >= 0.0f) ? std::atan(gy / gx) : (M_PI - std::atan(-gy / gx))) : 0.0f;
      gradients[i * outputStride + j] = geometry::Vec2f{len, theta};
    }
  }
}

__global__ __launch_bounds__(tileWidth* tileHeight) static void extractEdgesKernel(
    const geometry::Vec2f* __restrict__ gradients,
    float* __restrict__ edges,
    const size_t w,
    const size_t h,
    const size_t inputStride,
    const size_t outputStride)
{
//   static const std::array<geometry::Vec2i, 16> indices = {
// 
//   };
//   for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += blockDim.x * gridDim.x)
//   {
//     for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < w; j += blockDim.y * gridDim.y)
//     {
//       const geometry::Vec2f p = gradients[i * inputStride + j];
//       const geometry::Vec2f neighbours[8] = {
//           readPixel(gradients, i - 1, j - 1, w, h, inputStride, p),
//           readPixel(gradients, i - 1, j, w, h, inputStride, p),
//           readPixel(gradients, i - 1, j + 1, w, h, inputStride, p),
//           readPixel(gradients, i, j - 1, w, h, inputStride, p),
//           readPixel(gradients, i, j + 1, w, h, inputStride, p),
//           readPixel(gradients, i + 1, j - 1, w, h, inputStride, p),
//           readPixel(gradients, i + 1, j, w, h, inputStride, p),
//           readPixel(gradients, i + 1, j + 1, w, h, inputStride, p)};
// 
//       // TODO : look for the two nn from the direction of the gradient dependding on the orientation
//       // id
// 
//       const size_t refOrientation = assignOrientation(p.y);
//       float maxVal = 0.05f; // Set a threshold value here
//       int n = 0;
// #pragma unroll
//       for(size_t neighbId = 0; neighbId < 8; ++neighbId)
//       {
//         const auto& neighb = neighbours[neighbId];
//         const size_t orientation = assignOrientation(neighb.y);
//         if(orientation == refOrientation)
//         {
//           maxVal = std::max(maxVal, neighb.x);
//           n++;
//         }
//       }
//       edges[i * outputStride + j] = (p.x > 0.01f || (p.x >= maxVal && n >= 2)) ? p.x : 0.0f;
//     }
//   }
}

template <typename T>
__global__ static void harrisDetectorKernel(
    const geometry::Vec2f* __restrict__ gradients,
    T* __restrict__ output,
    const size_t w,
    const size_t h,
    const size_t inputStride,
    const size_t outputStride,
    const float k)
{
  static constexpr int blockSize = 3;
  for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += blockDim.x * gridDim.x)
  {
    for(size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < w; j += blockDim.y * gridDim.y)
    {
      double IxxSum = 0.0f;
      double IyySum = 0.0f;
      double IxySum = 0.0f;
      float n = 0.0f;
#pragma unroll
      for(int ii = -blockSize; ii <= blockSize; ++ii)
      {
#pragma unroll
        for(int jj = blockSize; jj <= blockSize; ++jj)
        {
          const int v = i + ii;
          const int u = j + jj;
          if(v >= 0 && v < h && u >= 0 && u < w)
          {
            const size_t index = (i + ii) * inputStride + j + jj;
            const auto grad = gradients[index];
            IxxSum += double(grad.x) * double(grad.x);
            IyySum += double(grad.y) * double(grad.y);
            IxySum += double(grad.x) * double(grad.y);
          }
        }
      }
      const double trM = IxxSum + IyySum;
      const double detM = IxxSum * IyySum - IxySum * IxySum;
      const double response = detM - double(k) * trM * trM;
      output[i * outputStride + j] = T(response);
    }
  }
}

// -------------------------------------------------------------------------------------------------

template <typename T>
Harris2D<T>::Harris2D(const float k, const size_t width, const size_t height)
    : kThr_{k}
    , width_{width}
    , height_{height}
    , gradients_{width_, height_}
    , edges_{width_, height_}
    , responses_{width_, height}
{}

template <typename T>
Harris2D<T>::Harris2D(const Harris2D<T>& cp)
    : kThr_{cp.kThr_}
    , width_{cp.width_}
    , height_{cp.height_}
    , gradients_{width_, height_}
    , edges_{width_, height_}
    , responses_{width_, height_}
{}

template <typename T>
Harris2D<T>& Harris2D<T>::operator=(const Harris2D<T>& cp)
{
  width_ = cp.width_;
  height_ = cp.height_;
  gradients_.resize(width_, height_);
  edges_.resize(width_, height_);
  responses_.resize(width_, height_);
  return *this;
}

template <typename T>
void Harris2D<T>::resize(const size_t width, const size_t height)
{
  width_ = width;
  height_ = height;
  gradients_.resize(width_, height_);
  edges_.resize(width_, height_);
  responses_.resize(width_, height_);
}

template <typename T>
void Harris2D<T>::compute(const Image2D<ImageFormat::R, T>& input, const cudaStream_t& stream)
{
  if(input.width() != gradients_.width() || input.height() != gradients_.height())
  {
    throw std::runtime_error("Input sizes mismatch for Harris detector");
  }
  extractGradientsKernel<<<dim3(16, 16), dim3(32, 32), 0, stream>>>(
      input.img().data(), gradients_.img().data(), input.width(), input.height(),
      input.img().stride(), gradients_.img().stride());
  extractEdgesKernel<<<dim3(16, 16), dim3(32, 32), 0, stream>>>(
      gradients_.img().data(), edges_.img().data(), gradients_.img().width(),
      gradients_.img().height(), gradients_.img().stride(), edges_.img().stride());
  // performNmsSuppression<<<dim3(16, 16), dim3(32, 32), 0, stream>>>(
  //     gradients_.img().data(), edges_.img().data(), edges_.img().width(),
  //     edges_.img().height(), gradients_.img().stride(), edges_.img().stride());
  // harrisDetectorKernel<<<dim3(16, 16), dim3(32, 32), 0, stream>>>(
  //     gradients_.img().data(), responses_.img().data(), gradients_.img().width(),
  //     gradients_.img().height(), gradients_.img().stride(), responses_.img().stride(), kThr_);
}

// -------------------------------------------------------------------------------------------------
// Explicit class intanciation
template class Harris2D<uint8_t>;
template class Harris2D<uint16_t>;
template class Harris2D<uint32_t>;
template class Harris2D<int8_t>;
template class Harris2D<int16_t>;
template class Harris2D<int32_t>;
template class Harris2D<float>;
template class Harris2D<double>;
} // namespace fusion