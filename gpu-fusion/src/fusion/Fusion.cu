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

#include "fusion/Fusion.hpp"

#include <cub/cub.cuh>
#include <cuda.h>
#include <omp.h>

#include "io/Ply.hpp"

#include "../utils/DepthUtils.inl"
#include "../utils/Ptr.inl"

// TODO : move in raycaster
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fusion
{
namespace gpu
{
    ATTR_HOST_DEV_INL static float getDepth(
        const uint16_t* depth,
        const math::Vec2f uv,
        const float depthScale,
        const size_t width,
        const size_t height)
    {
        if(uv.x < 0.0f || uv.x > (width - 1) || uv.y < 0.0f || uv.y > (height - 1))
        {
            return 0.0f;
        }

        const float u0 = std::floor(uv.x);
        const float u1 = u0 + 1.0f;
        const float v0 = std::floor(uv.y);
        const float v1 = v0 + 1.0f;

#ifdef __CUDA_ARCH__
        const float d0 = float(__ldg(depth + size_t(v0 * width + u0))) / depthScale;
        const float d1 = float(__ldg(depth + size_t(v0 * width + u1))) / depthScale;
        const float d2 = float(__ldg(depth + size_t(v1 * width + u0))) / depthScale;
        const float d3 = float(__ldg(depth + size_t(v1 * width + u1))) / depthScale;
#else
        const float d0 = float(depth[int(v0 * width + u0)]) / depthScale;
        const float d1 = float(depth[int(v0 * width + u1)]) / depthScale;
        const float d2 = float(depth[int(v1 * width + u0)]) / depthScale;
        const float d3 = float(depth[int(v1 * width + u1)]) / depthScale;
#endif

        static constexpr float deltaEps = 0.05f; // TODO : use a param
        const float w3 = (d3 == 0) ? 0.0f : (uv.x - u0) * (uv.y - v0);
        const float w2 = (d2 == 0) ? 0.0f : (u1 - uv.x) * (uv.y - v0);
        const float w1 = (d1 == 0) ? 0.0f : (uv.x - u0) * (v1 - uv.y);
        const float w0 = (d0 == 0) ? 0.0f : (u1 - uv.x) * (v1 - uv.y);
        const float w = w0 + w1 + w2 + w3;
        const float d = (w > 0.0f) ? (w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3) / w : 0.0f;

        return (d > 0.0f
                && std::max(
                       std::abs(d - d0),
                       std::max(std::abs(d - d1), std::max(std::abs(d - d2), std::abs(d - d3))))
                       <= deltaEps)
                   ? d
                   : 0.0f;
    }

    ATTR_HOST_DEV_INL static float getScale(const float depth, const float voxelRes)
    {
        return log2f(depth / voxelRes);
    }

    template <typename T>
    __global__ static void clearBuffer(T* data, const size_t n, const T val = T(0))
    {
        for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x)
        {
            data[idx] = val;
        }
    }

    template <size_t THREADS>
    __global__ static void getBounds(
        const math::Vec3f* points,
        const math::Mat4f m,
        math::Vec3f* minPoint,
        math::Vec3f* maxPoint,
        const size_t pointCount)
    {
        struct Vec3ReduceMin
        {
            __device__ __forceinline__ math::Vec3f operator()(
                const math::Vec3f v0, const math::Vec3f v1)
            {
                return utils::getMinPoint(v0, v1);
            }
        };
        struct Vec3ReduceMax
        {
            __device__ __forceinline__ math::Vec3f operator()(
                const math::Vec3f v0, const math::Vec3f v1)
            {
                return utils::getMinPoint(v0, v1);
            }
        };
        typedef cub::BlockReduce<math::Vec3<float>, THREADS, cub::BLOCK_REDUCE_WARP_REDUCTIONS>
            BlockReduce;
        __shared__ typename BlockReduce::TempStorage tmpStorage;

        math::Vec3f localMin{std::numeric_limits<float>::max()};
        math::Vec3f localMax{std::numeric_limits<float>::lowest()};
        for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < pointCount;
            idx += blockDim.x * gridDim.x)
        {
            const auto& p0 = points[idx];
            if(p0.x == std::numeric_limits<float>::max())
            {
                continue;
            }
            const auto p = m * p0;

            localMin = utils::getMinPoint(p, localMin);
            localMax = utils::getMaxPoint(p, localMax);
        }

        auto tmpMin = BlockReduce(tmpStorage).Reduce(localMin, Vec3ReduceMin());
        __syncthreads();

        auto tmpMax = BlockReduce(tmpStorage).Reduce(localMax, Vec3ReduceMax());
        __syncthreads();

        if(threadIdx.x == 0)
        {
            utils::atomicMinFloat(reinterpret_cast<float*>(minPoint) + 0, localMin.x);
            utils::atomicMinFloat(reinterpret_cast<float*>(minPoint) + 1, localMin.y);
            utils::atomicMinFloat(reinterpret_cast<float*>(minPoint) + 2, localMin.z);

            utils::atomicMaxFloat(reinterpret_cast<float*>(maxPoint) + 0, localMax.x);
            utils::atomicMaxFloat(reinterpret_cast<float*>(maxPoint) + 1, localMax.y);
            utils::atomicMaxFloat(reinterpret_cast<float*>(maxPoint) + 2, localMax.z);
        }
    }

    template <size_t THREADS>
    __global__ static void listIntersectingBlocks(
        const math::Vec3f* points,
        const math::Mat4f m,
        uint64_t* blockIds,
        uint32_t* blockCount,
        const float voxelSize,
        const size_t pointCount)
    {
        __shared__ uint64_t localIds[THREADS];
        __shared__ uint32_t localCount;
        __shared__ uint32_t globalId;

        for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < pointCount;
            idx += blockDim.x * gridDim.x)
        {
            if(threadIdx.x == 0)
            {
                localCount = 0;
            }
            localIds[threadIdx.x] = ~uint64_t(0);
            __syncthreads();

            const auto& p0 = points[idx];
            if(p0.x != std::numeric_limits<float>::max())
            {
                const auto p = m * p0;
                const uint32_t id = atomicAdd(&localCount, 1);
                localIds[id] = utils::encode(utils::getId(p, voxelSize));
            }
            __syncthreads();

            if(threadIdx.x == 0 && localCount > 0)
            {
                globalId = atomicAdd(blockCount, localCount);
            }
            __syncthreads();

            if(threadIdx.x < localCount)
            {
                blockIds[globalId + threadIdx.x] = localIds[threadIdx.x];
            }
            __syncthreads();
        }
    }

    __global__ static void sortBlockList(
        const uint64_t* blockIds,
        const uint32_t* blockCount,
        uint64_t* sortedBlockIds,
        uint8_t* tmpData,
        const size_t tmpDataSize)
    {
        size_t tmpStorageSize = tmpDataSize;
        cub::DeviceRadixSort::SortKeys(
            tmpData, tmpStorageSize, blockIds, sortedBlockIds, blockCount[0]);
    }

    __global__ static void reduceBlockList(
        const uint64_t* inputList,
        uint64_t* reducedList,
        uint32_t* listSize,
        uint8_t* tmpData,
        const size_t tmpDataSize)
    {
        const uint32_t count = listSize[0];
        size_t tmpStorageSize = tmpDataSize;
        cub::DeviceSelect::Unique(tmpData, tmpStorageSize, inputList, reducedList, listSize, count);
    }

    __global__ static void getIntersectingBlocksLaunch(
        const GpuFrameData frame,
        const math::Mat4f m,
        uint64_t* blockIds,
        uint32_t* blockCount,
        uint64_t* sortedBlockIds,
        uint8_t* tmpData,
        const size_t tmpDataSize,
        const float voxelSize)
    {
        static constexpr size_t threads = 256;

        const size_t pointCount = frame.width * frame.height;
        blockCount[0] = 0;

        clearBuffer<<<512, 256>>>(blockIds, pointCount, ~uint64_t(0));
        listIntersectingBlocks<threads>
            <<<512, threads>>>(frame.points, m, blockIds, blockCount, voxelSize, pointCount);
        sortBlockList<<<1, 1>>>(blockIds, blockCount, sortedBlockIds, tmpData, tmpDataSize);
        reduceBlockList<<<1, 1>>>(sortedBlockIds, blockIds, blockCount, tmpData, tmpDataSize);
    }

    __global__ static void integrateBatchKernel(
        const GpuFrameData* frames,
        const math::Mat4f* poses,
        const math::Mat3f k,
        const BlockHeader* headers,
        BlockPool blockPool,
        const float depthScale,
        const float tau,
        const float maxDepth,
        const float voxelRes,
        const size_t blockCount,
        const size_t batchSize)
    {
        const double cx = k.c02;
        const double cy = k.c12;
        const double fx = k.c00;
        const double fy = k.c11;

        for(size_t batchId = 0; batchId < batchSize; ++batchId)
        {
            const auto& frame = frames[batchId];
            const math::Mat4f invPose = math::Mat4f::Inverse(poses[batchId]);

            __syncthreads();
            for(size_t blockIndex = blockIdx.x; blockIndex < blockCount; blockIndex += gridDim.x)
            {
                const auto& header = headers[blockIndex];
                const math::Vec3i& blockId = header.blockId;
                const size_t memId = header.memId;
                float* __restrict__ sdfPtr = blockPool.sdf + memId * blockVolume;
                float* __restrict__ weightPtr = blockPool.weights + memId * blockVolume;
                math::Vec3f* __restrict__ colorPtr = blockPool.colors + memId * blockVolume;

                for(int idx = threadIdx.x; idx < blockVolume; idx += blockDim.x)
                {
                    const int i = idx % blockSize;
                    const int j = (idx / blockSize) % blockSize;
                    const int k = idx / (blockSize * blockSize);
                    const auto voxelId = math::Vec3i{i, j, k};
                    const math::Vec3f p
                        = utils::getVoxelPos(int(blockSize) * blockId + voxelId, voxelRes);
                    const math::Vec3f poseInFrame = invPose * p;
                    const math::Vec2f uv = utils::getDepthPos(poseInFrame, cx, cy, fx, fy);
                    const float depth
                        = getDepth(frame.depth, uv, depthScale, frame.width, frame.height);
                    const math::Vec3f ray = utils::getRay(uv, cx, cy, fx, fy);

                    // Integrate voxel
                    if(depth > 0.0f && depth <= maxDepth)
                    {
                        const float signVal
                            = (math::Vec3f::Dot(poseInFrame - depth * ray, ray) < 0.0f) ? 1.0f
                                                                                        : -1.0f;
                        const float dist = signVal * math::Vec3f::Len(depth * ray - poseInFrame);
                        // if(abs(dist) > tau)
                        // {
                        //     continue; // TODO : add w threshold in export
                        // }
                        const float val = dist / tau;
                        const float w = std::max(expf(-val * val), 0.0001f);
                        const float prevW = __ldg(weightPtr + idx);
                        const float newWeight = w + prevW;

                        // TODO : add other fields
                        const float prevDist
                            = (__ldg(sdfPtr + idx) != INVALID_TSDF) ? __ldg(sdfPtr + idx) : 0.0f;

                        sdfPtr[idx] = (w * dist + prevW * prevDist) / newWeight;
                        weightPtr[idx] = newWeight;

                        const float colorFact = std::min(1.0f, newWeight / 100.0f);
                        colorPtr[idx] = math::Vec3f{1.0f - colorFact, colorFact, 0.0f};
                    }
                }
            }
        }
    }

    // TODO : move in raytracer
    ATTR_HOST_DEV_INL static int findBlock(
        const math::Vec3i& blockId,
        const math::Vec3i* blockIdMap,
        const int* memIdMap,
        const math::Vec2i* mapOffsets,
        const size_t hashSize)
    {
        const int index = utils::hashIndex(blockId, hashSize);
        const auto& off = mapOffsets[index];
        if(off.y == 0)
        {
            return -1;
        }

        for(int i = 0; i < off.y; ++i)
        {
            if(blockIdMap[off.x + i] == blockId)
            {
                return memIdMap[off.x + i];
            }
        }

        return -1;
    }

#define DEFAULT_RAYCAST_POINT                                                                      \
    math::Vec3f { std::numeric_limits<float>::max() }

    __global__ static void raycastVolume(
        math::Vec3f* raycastedPoints,
        const math::Mat4f m,
        const BlockHeader* headers,
        BlockPool blockPool,
        const math::Vec3i* blockIdMap,
        const int* memIdMap,
        const math::Vec2i* mapOffsets,
        const float voxelSize,
        const float near,
        const float far,
        const size_t blockCount,
        const size_t hashSize,
        const size_t imgSize)
    {
        const size_t halfSize = imgSize >> 1;
        for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < imgSize * imgSize;
            idx += blockDim.x * gridDim.x)
        {
            const int u = idx % imgSize;
            const int v = idx / imgSize;
            const math::Vec3f ray = m
                                    * math::Vec3f::Normalize(math::Vec3f{
                                        float(u - halfSize) / float(halfSize),
                                        float(halfSize - v) / float(halfSize),
                                        -1.0f});

            float step = near;
            bool found = false;
            while(!found && step < far)
            {
                auto p = step * ray;
                const auto blockId = utils::getId(p, voxelSize);
                const size_t voxelId = math::Vec3i::Dot(
                    utils::getVoxelId(p, voxelSize), {1, blockSize, blockSize * blockSize});
                const int memId = findBlock(blockId, blockIdMap, memIdMap, mapOffsets, hashSize);
                if(memId == -1)
                {
                    step += voxelSize;
                    continue;
                }

                const float sdf = blockPool.sdf[memId * blockVolume + voxelId];
                if(sdf == INVALID_TSDF)
                {
                    step += voxelSize;
                    continue;
                }

                if(abs(sdf) <= voxelSize)
                {
                    found = true;
                    continue;
                }

                if(sdf >= 0.0f)
                {
                    step += std::min(abs(sdf), 1e-2f * voxelSize);
                }
                else
                {
                    break; // No intersection
                }
            }
            raycastedPoints[idx] = found ? step * ray : DEFAULT_RAYCAST_POINT;
        }
    }

    ATTR_HOST_DEV_INL static math::Vec3f getNormal(
        const math::Vec3f* points, const int u, const int v, const size_t imgSize)
    {
        const size_t index = v * imgSize + u;
        if(points[index] == DEFAULT_RAYCAST_POINT)
        {
            return math::Vec3f{0.0f};
        }

        auto n = math::Vec3f{0};
        size_t count = 0;

        const int index00 = (u < (int(imgSize) - 1)) ? v * int(imgSize) + u + 1 : -1;
        const int index01 = (v > 0) ? (v - 1) * int(imgSize) + u : -1;
        const int index10 = (u > 0) ? v * int(imgSize) + u - 1 : -1;
        const int index11 = (v < int(imgSize - 1)) ? (v + 1) * int(imgSize) + u : -1;

        const bool mask00 = (index00 > 0) && (points[index00] != DEFAULT_RAYCAST_POINT);
        const bool mask01 = (index01 > 0) && (points[index01] != DEFAULT_RAYCAST_POINT);
        const bool mask10 = (index10 > 0) && (points[index10] != DEFAULT_RAYCAST_POINT);
        const bool mask11 = (index11 > 0) && (points[index11] != DEFAULT_RAYCAST_POINT);

        if(index00 >= 0 && index01 >= 0 && mask00 && mask01)
        {
            n += math::Vec3f::Cross(
                points[index00] - points[index], points[index01] - points[index]);
            count++;
        }
        if(index01 >= 0 && index10 >= 0 && mask01 && mask10)
        {
            n += math::Vec3f::Cross(
                points[index01] - points[index], points[index10] - points[index]);
            count++;
        }
        if(index10 >= 0 && index11 >= 0 && mask10 && mask11)
        {
            n += math::Vec3f::Cross(
                points[index10] - points[index], points[index11] - points[index]);
            count++;
        }
        if(index11 >= 0 && index00 >= 0 && mask11 && mask00)
        {
            n += math::Vec3f::Cross(
                points[index11] - points[index], points[index00] - points[index]);
            count++;
        }
        return (count > 0) ? math::Vec3f::Normalize(n) : math::Vec3f{0};
    }

    __global__ static void renderImage(
        const math::Vec3f* raycastedPoints, uint8_t* img, const size_t imgSize)
    {
        for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < imgSize * imgSize;
            idx += blockDim.x * gridDim.x)
        {
            const int u = idx % imgSize;
            const int v = idx / imgSize;

            const math::Vec3f n = getNormal(raycastedPoints, u, v, imgSize);
            const uint32_t r = 255.0f * abs(n.x);
            const uint32_t g = 255.0f * abs(n.y);
            const uint32_t b = 255.0f * abs(n.z);
            img[3 * idx] = b;
            img[3 * idx + 1] = g;
            img[3 * idx + 2] = r;
        }
    }

    __global__ static void renderVolume(
        uint8_t* img,
        math::Vec3f* raycastedPoints,
        const math::Mat4f m,
        const BlockHeader* headers,
        BlockPool blockPool,
        const math::Vec3i* blockIdMap,
        const int* memIdMap,
        const math::Vec2i* mapOffsets,
        const float voxelSize,
        const float near,
        const float far,
        const size_t blockCount,
        const size_t hashSize,
        const size_t imgSize)
    {
        static constexpr size_t threads = 512;
        const size_t imgRes = imgSize * imgSize;
        raycastVolume<<<utils::div_up(imgRes, threads), threads>>>(
            raycastedPoints,
            m,
            headers,
            blockPool,
            blockIdMap,
            memIdMap,
            mapOffsets,
            voxelSize,
            near,
            far,
            blockCount,
            hashSize,
            imgSize);
        renderImage<<<utils::div_up(imgRes, threads), threads>>>(raycastedPoints, img, imgSize);
    }
} // namespace gpu

// -------------------------------------------------------------------------------------------------

Fusion::Fusion(const FusionParameters& params) : params_{params}, volume_{params_}
{
    auto logMemory = [&]() {
        const auto [available, total] = getAvailableGpuMemory();
        utils::Log::info(
            "Fusion",
            "Available GPU memory : %llu [MB] over %llu [MB]",
            available / 1024ULL / 1024ULL,
            total / 1024ULL / 1024ULL);
    };

    utils::Log::message(
        "Running fusion with :\n"
        "  - finestVoxelSize : %f [m]\n"
        "  - tau             : %f [m]\n"
        "  - near            : %f [m]\n"
        "  - far             : %f [m]\n",
        params_.voxelRes,
        params_.tau,
        params_.near,
        params_.far);

    logMemory();
    allocateMemory(params_.maxWidth, params_.maxHeight);
    logMemory();

    gpuErrcheck(cudaStreamCreate(&mainStream_));
    for(size_t batchId = 0; batchId < maxBatchSize; ++batchId)
    {
        gpuErrcheck(cudaStreamCreate(&subStreams_[batchId]));
        gpuErrcheck(cudaEventCreateWithFlags(
            &fireEvents_[batchId], cudaEventDisableTiming | cudaEventBlockingSync));
        gpuErrcheck(cudaEventCreateWithFlags(
            &waitEvents_[batchId], cudaEventDisableTiming | cudaEventBlockingSync));
    }
}

Fusion::~Fusion()
{
    gpuErrcheck(cudaStreamDestroy(mainStream_));
    for(size_t batchId = 0; batchId < maxBatchSize; ++batchId)
    {
        gpuErrcheck(cudaStreamDestroy(subStreams_[batchId]));
        gpuErrcheck(cudaEventDestroy(fireEvents_[batchId]));
        gpuErrcheck(cudaEventDestroy(waitEvents_[batchId]));
    }
}

void Fusion::integrateFrames(
    const std::vector<CpuFrameType>& frames, const std::vector<math::Mat4f>& poses)
{
    const size_t frameCount = frames.size();
    for(size_t batchOffset = 0; batchOffset < frameCount; batchOffset += maxBatchSize)
    {
        const size_t batchSize = std::min(maxBatchSize, frameCount - batchOffset);

        // Prepare frames
        notifySubStreamsStart(batchSize);
        prepareFrames(frames, poses, batchOffset, batchSize);
        waitForSubStreams(batchSize);

        // Get intersecting ids
        notifySubStreamsStart(batchSize);
        computeIntersectingBlocks(poses, batchOffset, batchSize);
        waitForSubStreams(batchSize);

        cudaStreamSynchronize(mainStream_);

        const auto blocks = getIntersectingBlocks(batchSize);
        utils::Log::info("Fusion", "Blocks intersecting : %zu", blocks.size());

        if(blocks.size() > maxBlockCount)
        {
            utils::Log::warning(
                "Fusion", "Max block count less than the current block count, skipping batch");
            continue;
        }
        volume_.streamBlocks(blocks);

        // Integrate frames batch
        performIntegration(blocks, poses, batchOffset, batchSize);
        if constexpr(renderFrames)
        {
            raycastFrames(poses, batchOffset, batchSize);
        }
    }
}

void Fusion::prepareFrames(
    const std::vector<CpuFrameType>& frames,
    const std::vector<math::Mat4f>& /*poses*/,
    const size_t batchOffset,
    const size_t batchSize)
{
    utils::Log::info(
        "Fusion",
        "Preparing frames %zu - %zu over %zu",
        batchOffset,
        batchOffset + batchSize,
        frames.size());

    for(size_t batchId = 0; batchId < batchSize; ++batchId)
    {
        const size_t frameId = batchOffset + batchId;
        const cudaStream_t& stream = subStreams_[batchId];

        waitForStart(batchId);
        frames[frameId].copyTo(framesHost_[batchId], stream);
        framesHost_[batchId].uploadTo(frames_[batchId], stream);
        frames_[batchId].prepare(params_.intrinsics, params_.depthScale, stream);
        notifyWorkDone(batchId);
    }
}

void Fusion::computeIntersectingBlocks(
    const std::vector<math::Mat4f>& poses, const size_t batchOffset, const size_t batchSize)
{
    for(size_t batchId = 0; batchId < batchSize; ++batchId)
    {
        const size_t frameId = batchOffset + batchId;
        const cudaStream_t& stream = subStreams_[batchId];

        waitForStart(batchId);
        gpu::getIntersectingBlocksLaunch<<<1, 1, 0, stream>>>(
            frames_[batchId].getData(),
            poses[frameId],
            blockList_[batchId],
            blockCounts_[batchId],
            sortedBlockIds_[batchId],
            tmpData_ + batchId * tmpDataSize_,
            tmpDataSize_,
            params_.voxelRes);

        blockCountsHost_[batchId].downloadFrom(blockCounts_[batchId], 1, stream);
        blockListHost_[batchId].downloadFrom(
            blockList_[batchId], blockList_[batchId].size(), stream);
        notifyWorkDone(batchId);
    }
}

std::vector<math::Vec3i> Fusion::getIntersectingBlocks(const size_t batchSize)
{
    utils::Timer timer("Fusion::getIntersectingBlocks()");

    const int w = int(params_.tau / (float(blockSize) * params_.voxelRes)) + 1;

    std::set<math::Vec3i> intersectingIds;
    for(size_t batchId = 0; batchId < batchSize; ++batchId)
    {
        const size_t blockCount = size_t(blockCountsHost_[batchId][0]);
        for(size_t i = 0; i < std::min(blockCount, maxBlockCount); ++i)
        {
            const auto blockId = utils::decode(blockListHost_[batchId][i]);
            for(int xOff = -w; xOff <= w; ++xOff)
            {
                for(int yOff = -w; yOff <= w; ++yOff)
                {
                    for(int zOff = -w; zOff <= w; ++zOff)
                    {
                        intersectingIds.emplace(blockId + math::Vec3i{xOff, yOff, zOff});
                    }
                }
            }
        }
    }
    std::vector<math::Vec3i> blocks;
    blocks.reserve(intersectingIds.size());

    for(const auto& id : intersectingIds)
    {
        blocks.emplace_back(id);
    }

    return blocks;
}

void Fusion::performIntegration(
    const BlockIdList& blockIds,
    const std::vector<math::Mat4f>& poses,
    const size_t batchOffset,
    const size_t batchSize)
{
    utils::Timer timer("fusion::performIntegration()");
    for(size_t batchId = 0; batchId < batchSize; ++batchId)
    {
        framesToIntegrateHost_[batchId] = frames_[batchId].getData();
        posesToIntegrateHost_[batchId] = poses[batchOffset + batchId];
    }
    framesToIntegrateHost_.uploadTo(framesToIntegrate_, batchSize, mainStream_);
    posesToIntegrateHost_.uploadTo(posesToIntegrate_, batchSize, mainStream_);

    const size_t blockCount = blockIds.size();
    const auto headers = volume_.getHeaders(blockIds);
    for(size_t id = 0; id < blockCount; ++id)
    {
        headersHost_[id] = headers[id];
    }
    headersHost_.uploadTo(headers_, blockCount, mainStream_);
    gpu::integrateBatchKernel<<<blockCount, 1024, 0, mainStream_>>>(
        framesToIntegrate_,
        posesToIntegrate_,
        params_.intrinsics,
        headers_,
        volume_.blockPool(),
        params_.depthScale,
        params_.tau,
        params_.far,
        params_.voxelRes,
        blockCount,
        batchSize);
    gpuErrcheck(cudaStreamSynchronize(mainStream_));
}

void Fusion::raycastFrames(
    const std::vector<math::Mat4f>& poses, const size_t batchOffset, const size_t batchSize)
{
    static constexpr size_t imgSize = 2048;

    static constexpr float fov = 45.0f;
    static constexpr float near = 0.1f;
    static constexpr float far = 10.0f;

    const float tanFov = std::tan(0.5f * (fov * M_PI) / 180.0f);
    const math::Vec3f dir(0.0f, 0.0f, -1.0f);

    BlockIdList blockList;

    GpuPtr<math::Vec3f> raycastedPoints{imgSize * imgSize};
    GpuPtr<uint8_t> image{3 * imgSize * imgSize};
    std::vector<uint8_t> imgData;
    imgData.resize(3 * imgSize * imgSize);

    for(size_t batchId = 0; batchId < batchSize; ++batchId)
    {
        const size_t frameId = batchOffset + batchId;
        const auto& pose = poses[frameId];

        math::Vec3f corners[8];

        corners[0] = dir * near + math::Vec3f(-far * tanFov, -far * tanFov, 0.0f);
        corners[1] = dir * near + math::Vec3f(far * tanFov, -far * tanFov, 0.0f);
        corners[2] = dir * near + math::Vec3f(-far * tanFov, far * tanFov, 0.0f);
        corners[3] = dir * near + math::Vec3f(far * tanFov, far * tanFov, 0.0f);

        corners[4] = dir * far + math::Vec3f(-far * tanFov, -far * tanFov, 0.0f);
        corners[5] = dir * far + math::Vec3f(far * tanFov, -far * tanFov, 0.0f);
        corners[6] = dir * far + math::Vec3f(-far * tanFov, far * tanFov, 0.0f);
        corners[7] = dir * far + math::Vec3f(far * tanFov, far * tanFov, 0.0f);

        // Compute all the frustum points
        corners[0] = pose * params_.camToSensor /*worldToCam*/ * corners[0];
        corners[1] = pose * params_.camToSensor /*worldToCam*/ * corners[1];
        corners[2] = pose * params_.camToSensor /*worldToCam*/ * corners[2];
        corners[3] = pose * params_.camToSensor /*worldToCam*/ * corners[3];
        corners[4] = pose * params_.camToSensor /*worldToCam*/ * corners[4];
        corners[5] = pose * params_.camToSensor /*worldToCam*/ * corners[5];
        corners[6] = pose * params_.camToSensor /*worldToCam*/ * corners[6];
        corners[7] = pose * params_.camToSensor /*worldToCam*/ * corners[7];

        float minX = std::numeric_limits<float>::max();
        float minY = std::numeric_limits<float>::max();
        float minZ = std::numeric_limits<float>::max();

        float maxX = -std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();
        float maxZ = -std::numeric_limits<float>::max();

        for(size_t i = 0; i < 8; i++)
        {
            minX = std::min(minX, corners[i].x);
            minY = std::min(minY, corners[i].y);
            minZ = std::min(minZ, corners[i].z);

            maxX = std::max(maxX, corners[i].x);
            maxY = std::max(maxY, corners[i].y);
            maxZ = std::max(maxZ, corners[i].z);
        }

        const BlockId b0 = utils::getId(math::Vec3f(minX, minY, minZ), params_.voxelRes);
        const BlockId b1 = utils::getId(math::Vec3f(maxX, maxY, maxZ), params_.voxelRes);

        blockList.clear();
        for(int i = b0.x; i <= b1.x; i++)
        {
            for(int j = b0.y; j <= b1.y; j++)
            {
                for(int k = b0.z; k <= b1.z; k++)
                {
                    const BlockId id(i, j, k);
                    blockList.emplace_back(id);
                }
            }
        }

        utils::Log::info("Fusion", "Block list size for frustum : %zu", blockList.size());
        const auto loadedBlocks = volume_.streamBlocks(blockList, false);
        const auto headers = volume_.getHeaders(loadedBlocks);
        utils::Log::info("Fusion", "Allocated blocks in frustum : %zu", headers.size());

        gpu::renderVolume<<<1, 1, 0, mainStream_>>>(
            image,
            raycastedPoints,
            pose * params_.camToSensor,
            headers_,
            volume_.blockPool(),
            volume_.blockIdMap(),
            volume_.memIdMap(),
            volume_.blockIdOffsets(),
            params_.voxelRes,
            0.1f,
            10.0f,
            loadedBlocks.size(),
            volume_.getHashSize(),
            imgSize);
        cudaDeviceSynchronize();
        image.downloadTo(imgData.data(), 3 * imgSize * imgSize, mainStream_);
        cudaDeviceSynchronize();

        cv::Mat outputImage(imgSize, imgSize, CV_8UC3, imgData.data());
        char imgName[512];
        static int imgCount = 0;
        snprintf(imgName, 512, "raycasted_%d.png", imgCount++);
        cv::imwrite(imgName, outputImage);
    }
}

void Fusion::allocateMemory(const size_t width, const size_t height)
{
    for(size_t batchId = 0; batchId < maxBatchSize; ++batchId)
    {
        frames_[batchId].resize(width, height);
        blockCounts_[batchId].resize(1);
        blockList_[batchId].resize(width * height);

        sortedBlockIds_[batchId].resize(width * height);

        framesHost_[batchId].resize(width, height);
        blockCountsHost_[batchId].resize(1);
        blockListHost_[batchId].resize(width * height);
    }

    size_t tmpSize = 0;
    cub::DeviceRadixSort::SortKeys<uint64_t, uint32_t>(
        nullptr, tmpSize, nullptr, nullptr, width * height);
    tmpDataSize_ = std::max(tmpDataSize_, tmpSize);

    cub::DeviceSelect::Unique<uint64_t*, uint64_t*, uint32_t*>(
        nullptr, tmpSize, nullptr, nullptr, nullptr, width * height);
    tmpDataSize_ = std::max(tmpDataSize_, tmpSize);

    tmpData_.resize(maxBatchSize * tmpDataSize_);

    volume_.allocateData(maxBlockCount);
}

void Fusion::exportFrame(const CpuFrameType& frame, const math::Mat4f& m, const char* filename)
{
    const double cx = params_.intrinsics.c02;
    const double cy = params_.intrinsics.c12;
    const double fx = params_.intrinsics.c00;
    const double fy = params_.intrinsics.c11;

    const size_t maxPointCount = frame.width() * frame.height();
    std::vector<math::Vec3f> points;
    points.reserve(maxPointCount);

    for(size_t j = 0; j < frame.height(); ++j)
    {
        for(size_t i = 0; i < frame.width(); ++i)
        {
            const uint16_t depth = frame.depth()[j * frame.width() + i];
            if(depth > 0)
            {
                const auto p = m * utils::getPoint(depth, j, i, params_.depthScale, cx, cy, fx, fy);
                points.push_back(p);
            }
        }
    }

    io::Ply::savePoints(filename, points);
}

std::tuple<size_t, size_t> Fusion::getAvailableGpuMemory()
{
    size_t availableMemory;
    size_t totalMemory;
    cudaSetDevice(0);
    cudaMemGetInfo(&availableMemory, &totalMemory);
    return {availableMemory, totalMemory};
}
} // namespace fusion