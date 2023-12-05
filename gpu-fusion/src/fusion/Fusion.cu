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

#include "fusion/Fusion.hpp"

#include <omp.h>
#include <cuda.h>
#include <cub/cub.cuh>

#include "../utils/DepthUtils.inl"

namespace fusion
{
namespace gpu
{
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
        const math::Mat4d m,
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
            const auto p = math::Vec3f{m * math::Vec3d{p0}};

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
        const math::Mat4d m,
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
                const auto p = math::Vec3f{m * math::Vec3d{p0}};
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
        const math::Mat4d m,
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
    const std::vector<CpuFrameType>& frames,
    const std::vector<math::Mat4d>& poses,
    const float depthScale)
{
    const size_t frameCount = frames.size();
    for(size_t batchOffset = 0; batchOffset < frameCount; batchOffset += maxBatchSize)
    {
        const size_t batchSize = std::min(maxBatchSize, frameCount - batchOffset);

        // Prepare frames
        notifySubStreamsStart(batchSize);
        prepareFrames(frames, poses, batchOffset, batchSize, depthScale);
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
    }
}

void Fusion::prepareFrames(
    const std::vector<CpuFrameType>& frames,
    const std::vector<math::Mat4d>& /*poses*/,
    const size_t batchOffset,
    const size_t batchSize,
    const float depthScale)
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
        frames_[batchId].prepare(params_.intrinsics, depthScale, stream);
        notifyWorkDone(batchId);
    }
}

void Fusion::computeIntersectingBlocks(
    const std::vector<math::Mat4d>& poses, const size_t batchOffset, const size_t batchSize)
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
    const int w = int(
        std::max(params_.tau - params_.voxelRes * float(blockSize / 2), 0.0f)
        / (float(blockSize) * params_.voxelRes));

    std::set<math::Vec3i> intersectingIds;
    for(size_t batchId = 0; batchId < batchSize; ++batchId)
    {
        const size_t blockCount = size_t(blockCountsHost_[batchId][0]);
        for(size_t i = 0; i < std::min(blockCount, maxBlockCount_); ++i)
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

std::tuple<size_t, size_t> Fusion::getAvailableGpuMemory()
{
    size_t availableMemory;
    size_t totalMemory;
    cudaSetDevice(0);
    cudaMemGetInfo(&availableMemory, &totalMemory);
    return {availableMemory, totalMemory};
}
} // namespace fusion