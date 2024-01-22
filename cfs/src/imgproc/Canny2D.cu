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

#include "imgProcCommon.inl"
#include "imgproc/Canny2D.hpp"
#include "math/geometry.hpp"

namespace cfs
{
__global__ static void __launch_bounds__(maxThreadsBlock) extractEdgesKernel(
    const float* __restrict__ gradients,
    uint8_t* __restrict__ edges,
    const size_t w,
    const size_t h,
    const size_t gradientStride,
    const size_t edgesStride,
    const float minThr,
    const float maxThr)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < int(h); i += blockDim.x * gridDim.x)
    {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < int(w); j += blockDim.y * gridDim.y)
        {
            const float g = gradients[i * gradientStride + j];
            if(g > maxThr)
            {
                edges[i * edgesStride + j] = 255u;
            }
            else if(g < minThr)
            {
                edges[i * edgesStride + j] = 0u;
            }
            else
            {
                uint8_t val = 0u;
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
                            const auto guv = gradients[u * gradientStride + v];
                            if(guv > minThr)
                            {
                                val = 255u;
                            }
                        }
                    }
                }
                edges[i * edgesStride + j] = val;
            }
        }
    }
}

template <typename T>
void Canny2D<T>::getGradients(
    const Image2D<ImageFormat::R, T>& input,
    Image2D<ImageFormat::R, float>& gradients,
    const cudaStream_t& stream)
{
    if(input.width() != gradients.width() || input.height() != gradients.height())
    {
        throw std::runtime_error("Input and output sizes mismatch for edge detector");
    }
    computeGradientsKernel<<<dim3(16, 16), dim3(32, 32), 0, stream>>>(
        input.img().data(),
        gradients.img().data(),
        input.width(),
        input.height(),
        input.img().stride(),
        gradients.img().stride());
}

template <typename T>
void Canny2D<T>::extractEdges(
    const Image2D<ImageFormat::R, float>& gradients,
    Image2D<ImageFormat::R, uint8_t>& edges,
    const cudaStream_t& stream)
{
    if(gradients.width() != edges.width() || gradients.height() != edges.height())
    {
        throw std::runtime_error("Input and output sizes mismatch for edge detector");
    }
    extractEdgesKernel<<<dim3(16, 16), dim3(32, 32), 0, stream>>>(
        gradients.img().data(),
        edges.img().data(),
        gradients.width(),
        gradients.height(),
        gradients.img().stride(),
        edges.img().stride(),
        minThr_,
        maxThr_);
}

// Explicit class intanciation
template class Canny2D<uint8_t>;
template class Canny2D<uint16_t>;
template class Canny2D<uint32_t>;
template class Canny2D<int8_t>;
template class Canny2D<int16_t>;
template class Canny2D<int32_t>;
template class Canny2D<float>;
template class Canny2D<double>;

} // namespace cfs