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

#include "DepthUtils.inl"
#include "Ptr.inl"
#include "utils/DepthUtils.hpp"

#include <cub/cub.cuh>

namespace fusion
{
namespace utils
{
    // ---------------------------------------------------------------------------------------------
    // --------------------------------------- Filtering -------------------------------------------
    // ---------------------------------------------------------------------------------------------

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
                        const float weight = std::exp(
                            -(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

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
        math::Vec3<uint8_t>* __restrict__ dstRgb,
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
                dstRgb[i * rgbStride + j] = math::Vec3<uint8_t>{
                    rgb[3 * (i * w + j)], rgb[3 * (i * w + j) + 1], rgb[3 * (i * w + j) + 2]};
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // ----------------------------------- Points extraction ---------------------------------------
    // ---------------------------------------------------------------------------------------------

    __global__ static void extractPointsKernel(
        const uint16_t* __restrict__ depth,
        math::Vec3f* __restrict__ points,
        const float scale,
        const math::Mat3f k,
        const size_t w,
        const size_t h,
        const size_t stride)
    {
        const double cx = k.c02;
        const double cy = k.c12;
        const double fx = k.c00;
        const double fy = k.c11;

        for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < w * h;
            idx += blockDim.x * gridDim.x)
        {
            const size_t j = idx % w;
            const size_t i = idx / w;
            const uint16_t d = depth[i * stride + j];
            if(d > 0)
            {
                const auto p0 = getPoint(d, j, i, scale, cx, cy, fx, fy);
                const auto p1 = getPoint(d, j + 1, i + 1, scale, cx, cy, fx, fy);
                points[i * w + j] = p0;
            }
            else
            {
                points[i * w + j] = math::Vec3f{std::numeric_limits<float>::max()};
            }
        }
    }

    __global__ __launch_bounds__(blockSize) static void extractPointsKernel(
        const uint16_t* __restrict__ depth,
        math::Vec3f* __restrict__ points,
        math::Vec3f* __restrict__ bbox,
        const float scale,
        const math::Mat3f k,
        const size_t w,
        const size_t h,
        const size_t stride)
    {
        struct Vec3ReduceMin
        {
            __device__ __forceinline__ math::Vec3f operator()(
                const math::Vec3f v0, const math::Vec3f v1)
            {
                return getMinPoint(v0, v1);
            }
        };
        struct Vec3ReduceMax
        {
            __device__ __forceinline__ math::Vec3f operator()(
                const math::Vec3f v0, const math::Vec3f v1)
            {
                return getMinPoint(v0, v1);
            }
        };
        typedef cub::BlockReduce<
            math::Vec3<float>,
            blockSizeX,
            cub::BLOCK_REDUCE_WARP_REDUCTIONS,
            blockSizeY>
            BlockReduce;
        __shared__ typename BlockReduce::TempStorage tmpStorage;

        const double cx = k.c02;
        const double cy = k.c12;
        const double fx = k.c00;
        const double fy = k.c11;

        math::Vec3f minPoint{std::numeric_limits<float>::max()};
        math::Vec3f maxPoint{std::numeric_limits<float>::lowest()};
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
        {
            for(size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < h;
                i += blockDim.y * gridDim.y)
            {
                const uint16_t d = depth[i * stride + j];
                if(d > 0)
                {
                    const auto p0 = getPoint(d, j, i, scale, cx, cy, fx, fy);
                    const auto p1 = getPoint(d, j + 1, i + 1, scale, cx, cy, fx, fy);
                    points[i * w + j] = p0;
                    minPoint = getMinPoint(p0, minPoint);
                    maxPoint = getMaxPoint(p0, maxPoint);
                }
                else
                {
                    points[i * w + j] = math::Vec3f{std::numeric_limits<float>::max()};
                }
            }
        }

        // Perform reduction
        auto localMin = BlockReduce(tmpStorage).Reduce(minPoint, Vec3ReduceMin());
        __syncthreads();

        auto localMax = BlockReduce(tmpStorage).Reduce(maxPoint, Vec3ReduceMax());
        __syncthreads();

        if(threadIdx.x * blockDim.x + threadIdx.y == 0)
        {
            atomicMinFloat(reinterpret_cast<float*>(bbox) + 0, localMin.x);
            atomicMinFloat(reinterpret_cast<float*>(bbox) + 1, localMin.y);
            atomicMinFloat(reinterpret_cast<float*>(bbox) + 2, localMin.z);

            atomicMaxFloat(reinterpret_cast<float*>(bbox) + 3, localMax.x);
            atomicMaxFloat(reinterpret_cast<float*>(bbox) + 4, localMax.y);
            atomicMaxFloat(reinterpret_cast<float*>(bbox) + 5, localMax.z);
        }
    }

    __global__ __launch_bounds__(blockSize) static void extractPointsKernel(
        const uint16_t* __restrict__ depth,
        math::Vec3f* __restrict__ points,
        float* __restrict__ footprints,
        const float scale,
        const math::Mat3f k,
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
            for(size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < h;
                i += blockDim.y * gridDim.y)
            {
                const uint16_t d = depth[i * stride + j];
                if(d > 0)
                {
                    const auto p0 = getPoint(d, j, i, scale, cx, cy, fx, fy);
                    const auto p1 = getPoint(d, j + 1, i + 1, scale, cx, cy, fx, fy);
                    footprints[i * w + j] = math::Vec3f::Dist(p0, p1);
                    points[i * w + j] = p0;
                }
                else
                {
                    footprints[i * w + j] = std::numeric_limits<float>::max();
                    points[i * w + j] = math::Vec3f{std::numeric_limits<float>::max()};
                }
            }
        }
    }

    __global__ __launch_bounds__(blockSize) static void extractPointsKernel(
        const uint16_t* __restrict__ depth,
        math::Vec3f* __restrict__ points,
        float* __restrict__ footprints,
        math::Vec3f* __restrict__ bbox,
        const float scale,
        const math::Mat3f k,
        const size_t w,
        const size_t h,
        const size_t stride)
    {
        struct Vec3ReduceMin
        {
            __device__ __forceinline__ math::Vec3f operator()(
                const math::Vec3f v0, const math::Vec3f v1)
            {
                return getMinPoint(v0, v1);
            }
        };
        struct Vec3ReduceMax
        {
            __device__ __forceinline__ math::Vec3f operator()(
                const math::Vec3f v0, const math::Vec3f v1)
            {
                return getMinPoint(v0, v1);
            }
        };
        typedef cub::BlockReduce<
            math::Vec3<float>,
            blockSizeX,
            cub::BLOCK_REDUCE_WARP_REDUCTIONS,
            blockSizeY>
            BlockReduce;
        __shared__ typename BlockReduce::TempStorage tmpStorage;

        const double cx = k.c02;
        const double cy = k.c12;
        const double fx = k.c00;
        const double fy = k.c11;

        math::Vec3f minPoint{std::numeric_limits<float>::max()};
        math::Vec3f maxPoint{std::numeric_limits<float>::lowest()};
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
        {
            for(size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < h;
                i += blockDim.y * gridDim.y)
            {
                const uint16_t d = depth[i * stride + j];
                if(d > 0)
                {
                    const auto p0 = getPoint(d, j, i, scale, cx, cy, fx, fy);
                    const auto p1 = getPoint(d, j + 1, i + 1, scale, cx, cy, fx, fy);
                    footprints[i * w + j] = math::Vec3f::Dist(p0, p1);
                    points[i * w + j] = p0;
                    minPoint = getMinPoint(p0, minPoint);
                    maxPoint = getMaxPoint(p0, maxPoint);
                }
                else
                {
                    footprints[i * w + j] = std::numeric_limits<float>::max();
                    points[i * w + j] = math::Vec3f{std::numeric_limits<float>::max()};
                }
            }
        }

        // Perform reduction
        // TODO : check performances
        auto localMin = BlockReduce(tmpStorage).Reduce(minPoint, Vec3ReduceMin());
        __syncthreads();

        auto localMax = BlockReduce(tmpStorage).Reduce(maxPoint, Vec3ReduceMax());
        __syncthreads();

        if(threadIdx.x * blockDim.x + threadIdx.y == 0)
        {
            atomicMinFloat(reinterpret_cast<float*>(bbox) + 0, localMin.x);
            atomicMinFloat(reinterpret_cast<float*>(bbox) + 1, localMin.x);
            atomicMinFloat(reinterpret_cast<float*>(bbox) + 2, localMin.x);

            atomicMaxFloat(reinterpret_cast<float*>(bbox) + 3, localMax.x);
            atomicMaxFloat(reinterpret_cast<float*>(bbox) + 4, localMax.x);
            atomicMaxFloat(reinterpret_cast<float*>(bbox) + 5, localMax.x);
        }
    }

    // ---------------------------------------------------------------------------------------------

    __global__ __launch_bounds__(blockSize) static void renderDepthMapKernel(
        const math::Vec3f* __restrict__ points,
        const float* __restrict__ footprints,
        math::Vec3i* __restrict__ triangles,
        int* __restrict__ triangleCount,
        const float maxDisparity,
        const size_t w,
        const size_t h)
    {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < w - 1;
            j += blockDim.x * gridDim.x)
        {
            for(size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < h - 1;
                i += blockDim.y * gridDim.y)
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
                if(math::all(math::notEqual(p0, math::Vec3f{std::numeric_limits<float>::max()}))
                   && math::all(math::notEqual(p1, math::Vec3f{std::numeric_limits<float>::max()}))
                   && math::all(math::notEqual(p2, math::Vec3f{std::numeric_limits<float>::max()})))
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
                if(math::all(math::notEqual(p0, math::Vec3f{std::numeric_limits<float>::max()}))
                   && math::all(math::notEqual(p2, math::Vec3f{std::numeric_limits<float>::max()}))
                   && math::all(math::notEqual(p3, math::Vec3f{std::numeric_limits<float>::max()})))
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

    __global__ __launch_bounds__(blockSize) static void estimateNormalsKernel(
        const math::Vec3f* __restrict__ points,
        math::Vec3f* __restrict__ normals,
        const math::Vec3i* __restrict__ triangles,
        const int* __restrict__ triangleCount)
    {
        for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size_t(*triangleCount);
            idx += blockDim.x * gridDim.x)
        {
            const auto& t = triangles[idx];
            const auto& p0 = points[t.x];
            const auto& p1 = points[t.y];
            const auto& p2 = points[t.z];

            const auto n = math::Vec3f::Cross(p1 - p0, p2 - p0);
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

    __global__ static void estimateNormalsKernel(
        math::Vec3f* __restrict__ points,
        math::Vec3f* __restrict__ normals,
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

                auto n = math::Vec3f{0};
                size_t count = 0;

                const int index00 = (j < (int(w) - 1)) ? i * int(w) + j + 1 : -1;
                const int index01 = (i > 0) ? (i - 1) * int(w) + j : -1;
                const int index10 = (j > 0) ? i * int(w) + j - 1 : -1;
                const int index11 = (i < (h - 1)) ? (i + 1) * int(w) + j : -1;

                if(index00 >= 0 && index01 >= 0 && masks[index00] && masks[index01])
                {
                    n += math::Vec3f::Cross(
                        points[index00] - points[index], points[index01] - points[index]);
                    count++;
                }
                if(index01 >= 0 && index10 >= 0 && masks[index01] && masks[index10])
                {
                    n += math::Vec3f::Cross(
                        points[index01] - points[index], points[index10] - points[index]);
                    count++;
                }
                if(index10 >= 0 && index11 >= 0 && masks[index10] && masks[index11])
                {
                    n += math::Vec3f::Cross(
                        points[index10] - points[index], points[index11] - points[index]);
                    count++;
                }
                if(index11 >= 0 && index00 >= 0 && masks[index11] && masks[index00])
                {
                    n += math::Vec3f::Cross(
                        points[index11] - points[index], points[index00] - points[index]);
                    count++;
                }
                normals[index] = (count > 0) ? math::Vec3f::Normalize(n) : math::Vec3f{0};
            }
        }
    }

    __global__ static void transformPointCloudKernel(
        math::Vec3f* __restrict__ points,
        math::Vec3f* __restrict__ normals,
        uint8_t* __restrict__ masks,
        const math::Mat4f& m,
        const size_t n)
    {
        for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x)
        {
            if(masks[idx])
            {
                points[idx] = m * points[idx];
                normals[idx] = m.GetRotation() * normals[idx];
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // -------------------------- Color unpacking --------------------------------------------------
    // ---------------------------------------------------------------------------------------------

    __global__ static void extractColorsKernelRGB(
        const math::Vec3<uint8_t>* __restrict__ src,
        math::Vec3f* __restrict__ dst,
        const size_t w,
        const size_t h,
        const size_t stride)
    {
        for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
        {
            for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
            {
                dst[i * w + j] = (1.0f / 255.0f) * math::Vec3f(src[i * stride + j]);
            }
        }
    }
    __global__ static void extractColorsKernelBGR(
        const math::Vec3<uint8_t>* __restrict__ src,
        math::Vec3f* __restrict__ dst,
        const size_t w,
        const size_t h,
        const size_t stride)
    {
        for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += blockDim.x * gridDim.x)
        {
            for(int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += blockDim.y * gridDim.y)
            {
                const auto& c = src[i * stride + j];
                dst[i * w + j] = (1.0f / 255.0f) * math::Vec3f(c.z, c.y, c.x);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------

    void initRGBDFrame(
        const GpuPtr<uint16_t>& depthData,
        const GpuPtr<uint8_t>& rgbData,
        GpuImg<uint16_t>& depth,
        GpuImg<math::Vec3<uint8_t>>& rgb,
        const size_t w,
        const size_t h,
        const cudaStream_t& stream)
    {
        const size_t res = w * h;
        if(depthData.size() < res)
        {
            throw std::runtime_error("initRGBDFrame() : depthData too small");
        }
        if(rgbData.size() < 3 * res)
        {
            throw std::runtime_error("initRGBDFrame() : colorData too small");
        }
        initRgbdFrameKernel<<<32, dim3(blockSizeX, blockSizeY), 0, stream>>>(
            rgbData, depthData, rgb, depth, w, h, rgb.stride(), depth.stride());
    }

    void extractPoints(
        const GpuImg<uint16_t>& depth,
        GpuPtr<math::Vec3f>& points,
        const math::Mat3f& k,
        const float depthScale,
        const cudaStream_t& stream)
    {
        static constexpr size_t threads = 512;
        const size_t res = depth.width() * depth.height();
        if(res > points.size())
        {
            throw std::runtime_error("extractPoints() : points buffer not large enough");
        }
        extractPointsKernel<<<utils::div_up(res, threads), threads, 0, stream>>>(
            depth, points, depthScale, k, depth.width(), depth.height(), depth.stride());
    }

    void extractPoints(
        const GpuImg<uint16_t>& depth,
        GpuPtr<math::Vec3f>& points,
        GpuPtr<math::Vec3f>& bbox,
        const math::Mat3f& k,
        const float depthScale,
        const cudaStream_t& stream)
    {
        const size_t res = depth.width() * depth.height();
        if(res > points.size())
        {
            throw std::runtime_error("extractPoints() : points buffer not large enough");
        }
        extractPointsKernel<<<32, dim3(blockSizeX, blockSizeY), 0, stream>>>(
            depth, points, bbox, depthScale, k, depth.width(), depth.height(), depth.stride());
    }

    void extractPoints(
        const GpuImg<uint16_t>& depth,
        GpuPtr<math::Vec3f>& points,
        GpuPtr<float>& footprints,
        const math::Mat3f& k,
        const float depthScale,
        const cudaStream_t& stream)
    {
        const size_t res = depth.width() * depth.height();
        if(res > points.size())
        {
            throw std::runtime_error("extractPoints() : points buffer not large enough");
        }
        if(res > footprints.size())
        {
            throw std::runtime_error("extractPoints() : footprints buffer not large enough");
        }
        extractPointsKernel<<<32, dim3(blockSizeX, blockSizeY), 0, stream>>>(
            depth,
            points,
            footprints,
            depthScale,
            k,
            depth.width(),
            depth.height(),
            depth.stride());
    }

    void extractPoints(
        const GpuImg<uint16_t>& depth,
        GpuPtr<math::Vec3f>& points,
        GpuPtr<float>& footprints,
        GpuPtr<math::Vec3f>& bbox,
        const math::Mat3f& k,
        const float depthScale,
        const cudaStream_t& stream)
    {
        const size_t res = depth.width() * depth.height();
        if(res > points.size())
        {
            throw std::runtime_error("extractPoints() : points buffer not large enough");
        }
        if(res > footprints.size())
        {
            throw std::runtime_error("extractPoints() : footprints buffer not large enough");
        }
        extractPointsKernel<<<32, dim3(blockSizeX, blockSizeY), 0, stream>>>(
            depth,
            points,
            footprints,
            bbox,
            depthScale,
            k,
            depth.width(),
            depth.height(),
            depth.stride());
    }

    void extractColors(
        const GpuImg<math::Vec3b>& rgb,
        GpuPtr<math::Vec3f>& color,
        const bool useBgr,
        const cudaStream_t& stream)
    {
        if(color.size() < rgb.width() * rgb.height())
        {
            throw std::runtime_error("extractColors() : buffer too small");
        }
        if(useBgr)
        {
            extractColorsKernelBGR<<<32, dim3(blockSizeX, blockSizeY), 0, stream>>>(
                rgb, color, rgb.width(), rgb.height(), rgb.stride());
        }
        else
        {
            extractColorsKernelRGB<<<32, dim3(blockSizeX, blockSizeY), 0, stream>>>(
                rgb, color, rgb.width(), rgb.height(), rgb.stride());
        }
    }
} // namespace utils
} // namespace fusion