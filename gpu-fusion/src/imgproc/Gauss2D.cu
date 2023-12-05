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

#include "common.hpp"
#include "imgproc/Gauss2D.hpp"
#include "math/vec2.hpp"

#include <cmath>

// #define GAUSS_DEBUG_LOG

namespace fusion
{
template <typename T>
__global__ static void applyKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const double* __restrict__ kernel,
    const size_t w,
    const size_t h,
    const size_t inputStride,
    const size_t outputStride,
    const size_t k,
    const size_t kStride)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += blockDim.x * gridDim.x)
    {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < w; j += blockDim.y * gridDim.y)
        {
            double acc = 0.0;
            double kSum = 0.0;
            for(int ii = -int(k); ii < int(k); ++ii)
            {
                for(int jj = -int(k); jj < int(k); ++jj)
                {
                    if(math::any(math::lessThan(math::Vec2i{i + ii, j + jj}, math::Vec2i{0, 0})))
                    {
                        continue;
                    }
                    if(math::any(math::greaterThanEqual(
                           math::Vec2i{i + ii, j + jj}, math::Vec2i{int(h), int(w)})))
                    {
                        continue;
                    }

                    const double kVal = kernel[ii * kStride + jj];
                    acc += kVal * double(input[(i + ii) * inputStride + j + jj]);
                    kSum += kVal;
                }
            }
            acc /= kSum;
            output[i * outputStride + j] = T(acc);
        }
    }
}
template <typename T>
__global__ static void applyKernel(
    const math::Vec3<T>* __restrict__ input,
    math::Vec3<T>* __restrict__ output,
    const double* __restrict__ kernel,
    const size_t w,
    const size_t h,
    const size_t inputStride,
    const size_t outputStride,
    const size_t k,
    const size_t kStride)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += blockDim.x * gridDim.x)
    {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < w; j += blockDim.y * gridDim.y)
        {
            math::Vec3d acc = 0.0;
            double kSum = 0.0;
            for(int ii = -int(k); ii < int(k); ++ii)
            {
                for(int jj = -int(k); jj < int(k); ++jj)
                {
                    if(math::any(math::lessThan(math::Vec2i{i + ii, j + jj}, math::Vec2i{0, 0})))
                    {
                        continue;
                    }
                    if(math::any(math::greaterThanEqual(
                           math::Vec2i{i + ii, j + jj}, math::Vec2i{int(h), int(w)})))
                    {
                        continue;
                    }

                    const double kVal = kernel[ii * kStride + jj];
                    acc += kVal * math::Vec3d(input[(i + ii) * inputStride + j + jj]);
                    kSum += kVal;
                }
            }
            acc /= kSum;
            output[i * outputStride + j] = math::Vec3<T>(acc);
        }
    }
}

template <ImageFormat format, typename T>
Gauss2D<format, T>::Gauss2D(const double sigma, const size_t k)
    : k_{k}
    , kernelStride_{2 * k + 1}
    , kernelSize_{kernelStride_ * kernelStride_}
    , kernelCpu_{kernelSize_}
    , kernelGpu_{kernelSize_}
{
    double sum = 0.0;
    const double sigmaSq = sigma * sigma;
    for(size_t i = 0; i < kernelStride_; ++i)
    {
        for(size_t j = 0; j < kernelStride_; ++j)
        {
            const auto u = math::Vec2d{double(i) - k_, double(j) - k};
            const double val = std::exp(-u.LenSq() / (2.0 * sigmaSq));
            kernelCpu_[i * kernelStride_ + j] = val;
            sum += val;
        }
    }
    const auto fact = 1.0 / (sum * sigma * std::sqrt(2.0 * M_PI));
    for(auto& val : kernelCpu_)
    {
        val *= fact;
    }

#ifdef GAUSS_DEBUG_LOG
    char debugLine[1024];
    int offset = sprintf(debugLine, "Kernel : \n");
    for(size_t i = 0; i < kernelStride_; ++i)
    {
        offset += sprintf(debugLine + offset, "|");
        for(size_t j = 0; j < kernelStride_; ++j)
        {
            offset += sprintf(debugLine + offset, " %5.12f ", kernelCpu_[i * kernelStride_ + j]);
        }
        offset += sprintf(debugLine + offset, "|\n");
    }
    utils::Log::message("%s", debugLine);
#endif

    kernelCpu_.uploadTo(kernelGpu_, cudaStream_t{0});
    cudaDeviceSynchronize();
}

template <ImageFormat format, typename T>
void Gauss2D<format, T>::filter(
    const Image2D<format, T>& input, Image2D<format, T>& output, const cudaStream_t& stream)
{
    const auto w = input.width();
    const auto h = input.height();
    if(input.width() != output.width() || input.height() != output.height())
    {
        throw std::runtime_error("Input and output dimensions mismatch");
    }
    applyKernel<<<dim3(32, 32), dim3(16, 16), 0, stream>>>(
        input.img().data(),
        output.img().data(),
        kernelGpu_,
        w,
        h,
        input.img().stride(),
        output.img().stride(),
        k_,
        kernelStride_);
}

// Explicit class intanciation
template class Gauss2D<ImageFormat::R, uint8_t>;
template class Gauss2D<ImageFormat::RGB, uint8_t>;

template class Gauss2D<ImageFormat::R, uint16_t>;
template class Gauss2D<ImageFormat::RGB, uint16_t>;

template class Gauss2D<ImageFormat::R, uint32_t>;
template class Gauss2D<ImageFormat::RGB, uint32_t>;

template class Gauss2D<ImageFormat::R, float>;
template class Gauss2D<ImageFormat::RGB, float>;

} // namespace fusion