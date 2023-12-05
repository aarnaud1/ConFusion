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

#include "fusion/Volume.hpp"

#include "../utils/Ptr.inl"

#include <omp.h>

namespace fusion
{
namespace gpu
{
    __global__ static void copyBlocksForSave(
        math::Vec3i* blockIds,
        const Voxel* blockPoolData,
        const size_t* memIds,
        math::Vec3i* streamingBlockIds,
        Voxel* streamingBlockPoolData,
        const size_t count)
    {
        for(size_t id = blockIdx.x; id < count; id += gridDim.x)
        {
            const size_t memId = memIds[id];
            if(threadIdx.x == 0)
            {
                streamingBlockIds[id] = blockIds[memId];
                blockIds[memId] = DEFAULT_BLOCK_ID;
            }
            __syncthreads();

            for(size_t idx = threadIdx.x; idx < blockVolume; idx += blockDim.x)
            {
                streamingBlockPoolData[id * blockVolume + idx]
                    = blockPoolData[memId * blockVolume + idx];
            }
        }
    }

    __global__ static void copyBlocksFromLoad(
        const math::Vec3i* streamingBlockIds,
        const Voxel* streamingBlockPoolData,
        const size_t* memIds,
        math::Vec3i* blockIds,
        Voxel* blockPoolData,
        const size_t count)
    {
        for(size_t id = blockIdx.x; id < count; id += gridDim.x)
        {
            const size_t memId = memIds[id];
            if(threadIdx.x == 0)
            {
                blockIds[memId] = streamingBlockIds[id];
            }
            __syncthreads();

            for(size_t idx = threadIdx.x; idx < blockVolume; idx += blockDim.x)
            {
                blockPoolData[memId * blockVolume + idx]
                    = streamingBlockPoolData[id * blockVolume + idx];
            }
        }
    }
} // namespace gpu

Volume::Volume(const FusionParameters& params) : params_{params} {}

void Volume::allocateData(const size_t maxBlockCount)
{
    // Pool data
    blockIds_.resize(maxBlockCount);
    blockPool_.resize(blockVolume * maxBlockCount);

    // StreamingData
    for(size_t tId = 0; tId < threadCount; ++tId)
    {
        streamingBlockIds_[tId].resize(streamingBatchSize);
        streamingBlocks_[tId].resize(blockVolume * streamingBatchSize);
        streamingMemIds_[tId].resize(streamingBatchSize);

        streamingBlockIdsHost_[tId].resize(streamingBatchSize);
        streamingBlocksHost_[tId].resize(blockVolume * streamingBatchSize);
        streamingMemIdsHost_[tId].resize(streamingBatchSize);
    }
    blockCache_.resize(maxBlockCount);

    for(size_t tId = 0; tId < threadCount; ++tId)
    {
        gpuErrcheck(cudaStreamCreate(&loadStreams_[tId]));
        gpuErrcheck(cudaStreamCreate(&saveStreams_[tId]));
    }

    blockIds_.set(DEFAULT_BLOCK_ID, cudaStream_t{0});
}

Volume::~Volume()
{
    for(size_t tId = 0; tId < threadCount; ++tId)
    {
        gpuErrcheck(cudaStreamDestroy(saveStreams_[tId]));
        gpuErrcheck(cudaStreamDestroy(loadStreams_[tId]));
    }
}

void Volume::streamBlocks(const BlockIdList& blockList)
{
    // Allocate blocks
    const size_t newBlocks = addBlocks(blockList);
    utils::Log::info("Volume", "%zu new blocks, total block count : %zu", newBlocks, blockCount());

    const auto memInfo = blockCache_.addBlocks(blockList);

    std::vector<size_t> evictedIndices;
    evictedIndices.reserve(blockCache_.getEvictions().size());
    for(const auto& id : blockCache_.getEvictions())
    {
        evictedIndices.push_back(memIds_[id]);
    }
    blockCache_.getEvictions().clear();

    // First save evicted blocks
    saveBlocks(evictedIndices);

    // Stream blocks
    std::vector<size_t> indicesToLoad;
    indicesToLoad.reserve(memInfo.size());

    std::vector<math::Vec3i> blockIdsToLoad;
    blockIdsToLoad.reserve(memInfo.size());

    for(size_t i = 0; i < memInfo.size(); ++i)
    {
        const auto& [memId, inCache] = memInfo[i];
        const auto blockId = blockList[i];

        memIds_[blockId] = memId;
        if(!inCache)
        {
            blockIdsToLoad.push_back(blockId);
            indicesToLoad.push_back(memId);
        }
    }
    streamBlocks(blockIdsToLoad, indicesToLoad);
}

bool Volume::addBlock(const math::Vec3i& blockId)
{
    const bool allocated = find(blockId);
    if(!allocated)
    {
        voxelBlocks_[blockId] = std::make_unique<VoxelBlock>(params_.voxelRes, 0);
        memIds_[blockId] = -1;
        return true;
    }
    return false;
}

size_t Volume::addBlocks(const std::vector<math::Vec3i>& blockIds)
{
    size_t ret = 0;
    for(const auto& blockId : blockIds)
    {
        ret += addBlock(blockId) ? 1 : 0;
    }
    return ret;
}

void Volume::removeBlock(const math::Vec3i& blockId) noexcept
{
    voxelBlocks_.erase(blockId);
    memIds_.erase(blockId);
}

void Volume::removeBlocks(const std::vector<math::Vec3i>& blockIds) noexcept
{
    for(const auto& blockId : blockIds)
    {
        removeBlock(blockId);
    }
}

void Volume::saveBlocks(const std::vector<size_t>& indices)
{
#pragma omp parallel for num_threads(threadCount)
    for(size_t batchOffset = 0; batchOffset < indices.size(); batchOffset += streamingBatchSize)
    {
        const size_t tId = omp_get_thread_num();
        const size_t batchSize = std::min(streamingBatchSize, indices.size() - batchOffset);

        const auto start = indices.begin() + batchOffset;
        const auto end = start + batchSize;
        std::copy(start, end, streamingMemIdsHost_[tId].begin());

        streamingMemIdsHost_[tId].uploadTo(streamingMemIds_[tId], batchSize, saveStreams_[tId]);
        gpu::copyBlocksForSave<<<streamingBatchSize, 256, 0, saveStreams_[tId]>>>(
            blockIds_,
            blockPool_,
            streamingMemIds_[tId],
            streamingBlockIds_[tId],
            streamingBlocks_[tId],
            batchSize);
        streamingBlockIdsHost_[tId].downloadFrom(
            streamingBlockIds_[tId], batchSize, saveStreams_[tId]);
        streamingBlocksHost_[tId].downloadFrom(
            streamingBlocks_[tId], batchSize * blockVolume, saveStreams_[tId]);
        cudaStreamSynchronize(saveStreams_[tId]);

        // Check that all blocks are allocated
        for(size_t id = 0; id < batchSize; ++id)
        {
            const auto blockId = streamingBlockIdsHost_[tId][id];
            const auto& blockPtr = voxelBlocks_[blockId];
            if(blockPtr == nullptr)
            {
                throw std::runtime_error("Trying to save unallocated block");
            }
        }

        for(size_t id = 0; id < batchSize; ++id)
        {
            const auto blockId = streamingBlockIdsHost_[tId][id];
            const auto* __restrict__ ptr = streamingBlocksHost_[tId] + id * blockVolume;
            auto* __restrict__ blockPtr = voxelBlocks_[blockId]->voxels().data();
            memIds_[blockId] = -1;
            memcpy(blockPtr, ptr, blockVolume * sizeof(Voxel));
        }
    }
}

void Volume::streamBlocks(
    const std::vector<math::Vec3i>& blockIds, const std::vector<size_t>& indices)
{
#pragma omp parallel for num_threads(threadCount)
    for(size_t batchOffset = 0; batchOffset < indices.size(); batchOffset += streamingBatchSize)
    {
        const size_t tId = omp_get_thread_num();
        const size_t batchSize = std::min(streamingBatchSize, indices.size() - batchOffset);

        const auto blockIdStart = blockIds.begin() + batchOffset;
        const auto blockIdEnd = blockIdStart + batchSize;
        std::copy(blockIdStart, blockIdEnd, streamingBlockIdsHost_[tId].begin());

        const auto memIdStart = indices.begin() + batchOffset;
        const auto memIdEnd = memIdStart + batchSize;
        std::copy(memIdStart, memIdEnd, streamingMemIdsHost_[tId].begin());

        // Check that all blocks are allocated
        for(size_t id = 0; id < batchSize; ++id)
        {
            const auto blockId = blockIds[id];
            const auto& blockPtr = voxelBlocks_[blockId];
            if(blockPtr == nullptr)
            {
                throw std::runtime_error("Trying to load unallocated block");
            }
        }

        for(size_t id = 0; id < batchSize; ++id)
        {
            const auto blockId = blockIds[id];
            const auto* __restrict__ blockPtr = voxelBlocks_[blockId]->voxels().data();
            auto* __restrict__ ptr = streamingBlocksHost_[tId] + id * blockVolume;
            memcpy(ptr, blockPtr, blockVolume * sizeof(Voxel));
        }

        streamingBlockIdsHost_[tId].uploadTo(streamingBlockIds_[tId], batchSize, loadStreams_[tId]);
        streamingMemIdsHost_[tId].uploadTo(streamingMemIds_[tId], batchSize, loadStreams_[tId]);
        streamingBlocksHost_[tId].uploadTo(
            streamingBlocks_[tId], batchSize * blockVolume, loadStreams_[tId]);
        gpu::copyBlocksFromLoad<<<streamingBatchSize, 256, 0, loadStreams_[tId]>>>(
            streamingBlockIds_[tId],
            streamingBlocks_[tId],
            streamingMemIds_[tId],
            blockIds_,
            blockPool_,
            batchSize);
        cudaStreamSynchronize(loadStreams_[tId]);
    }
}
} // namespace fusion