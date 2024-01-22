/*
 * Copyright (C) 2024 Adrien ARNAUD
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

#pragma once

#include "common.hpp"
#include "math/geometry.hpp"

namespace cfs
{
static constexpr size_t tileWidth = 32;
static constexpr size_t tileHeight = 32;
static constexpr size_t maxThreadsBlock = tileWidth * tileHeight;

template <typename T>
ATTR_DEV_INL static T readPixel(
    const T* __restrict__ data,
    const int i,
    const int j,
    const size_t w,
    const size_t h,
    const size_t stride,
    const T defaultValue)
{
    return (i >= 0 && i < h && j >= 0 && j < w) ? data[size_t(i) * stride + size_t(j)]
                                                : defaultValue;
}

template <typename T>
__global__ static void __launch_bounds__(maxThreadsBlock) computeGradientsKernel(
    const T* __restrict__ inputImg,
    float* __restrict__ gradients,
    const size_t w,
    const size_t h,
    const size_t inputStride,
    const size_t outputStride)
{
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += blockDim.x * gridDim.x)
    {
        for(size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < w; j += blockDim.y * gridDim.y)
        {
            const float p = float(inputImg[i * inputStride + j]);
            const float px0 = (i > 0) ? float(inputImg[(i - 1) * inputStride + j]) : p;
            const float px1 = (i < (h - 1)) ? float(inputImg[(i + 1) * inputStride + j]) : p;

            const float py0 = (j > 0) ? float(inputImg[i * inputStride + j - 1]) : p;
            const float py1 = (j < (w - 1)) ? float(inputImg[i * inputStride + j + 1]) : p;

            const auto g = 0.5f * math::Vec2f{px1 - px0, py1 - py0};
            const auto val = math::Vec2f::Dot(g, g);
            gradients[i * outputStride + j] = val;
        }
    }
}

template <typename T>
__global__ static void __launch_bounds__(maxThreadsBlock) computeGradientsKernel(
    const T* __restrict__ inputImg,
    math::Vec2f* __restrict__ gradients,
    const size_t w,
    const size_t h,
    const size_t inputStride,
    const size_t outputStride)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += blockDim.x * gridDim.x)
    {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < w; j += blockDim.y * gridDim.y)
        {
            const float p = float(inputImg[i * inputStride + j]);
            const float px0 = (i > 0) ? float(inputImg[(i - 1) * inputStride + j]) : p;
            const float px1 = (i < (h - 1)) ? float(inputImg[(i + 1) * inputStride + j]) : p;

            const float py0 = (j > 0) ? float(inputImg[i * inputStride + j - 1]) : p;
            const float py1 = (j < (w - 1)) ? float(inputImg[i * inputStride + j + 1]) : p;
            gradients[i * outputStride + j] = 0.5f * math::Vec2f{px1 - px0, py1 - py0};
        }
    }
}

template <typename T>
__global__ static void __launch_bounds__(maxThreadsBlock) performNmsSuppression(
    const T* __restrict__ inputImg,
    T* __restrict__ outputImg,
    const size_t w,
    const size_t h,
    const size_t inputStride,
    const size_t outputStride)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += blockDim.x * gridDim.x)
    {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < w; j += blockDim.y * gridDim.y)
        {
            T maxVal = std::numeric_limits<T>::lowest();
#pragma unroll
            for(int ii = -1; ii <= 1; ++ii)
            {
#pragma unroll
                for(int jj = -1; jj <= 1; ++jj)
                {
                    const int v = i + ii;
                    const int u = j + jj;
                    if(v >= 0 && v < h && u >= 0 && u < w)
                    {
                        const size_t index = (i + ii) * inputStride + j + jj;
                        const T tmp = inputImg[index];
                        maxVal = std::max(tmp, maxVal);
                    }
                }
            }
            outputImg[i * outputStride + j]
                = (inputImg[i * inputStride + j] == maxVal) ? maxVal : 0;
        }
    }
}

template <typename T>
ATTR_HOST_DEV_INL T clamp(const T x, const T a, const T b)
{
    return (x < a) ? a : ((x > b) ? b : x);
}

ATTR_HOST_DEV_INL size_t assignOrientation(const float theta)
{
    // return size_t(clamp(float(std::floor(4.0f * theta / M_PI)) + 4.0f, 0.0f, 8.0f));
    const float alpha = theta - M_PI / 4.0f;
    return clamp(std::floor(alpha / 4.0f) + 4.0f, 0.0f, 8.0f);
}
} // namespace cfs