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

#include "fusion/Volume.hpp"

#include <omp.h>
#include <algorithm>

#include "marching_cubes/MarchingCubes.hpp"
#include "io/Ply.hpp"

#include "../utils/Ptr.inl"

namespace fusion
{
namespace gpu
{
    __global__ static void copyBlocksForSave(
        math::Vec3i *blockIds,
        const BlockPool blockPool,
        const size_t *memIds,
        math::Vec3i *streamingBlockIds,
        BlockPool streamingBlockPool,
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
                streamingBlockPool.sdf[id * blockVolume + idx]
                    = blockPool.sdf[memId * blockVolume + idx];
                streamingBlockPool.weights[id * blockVolume + idx]
                    = blockPool.weights[memId * blockVolume + idx];
                streamingBlockPool.colors[id * blockVolume + idx]
                    = blockPool.colors[memId * blockVolume + idx];
                streamingBlockPool.gradients[id * blockVolume + idx]
                    = blockPool.gradients[memId * blockVolume + idx];
            }
        }
    }

    __global__ static void copyBlocksFromLoad(
        const math::Vec3i *streamingBlockIds,
        const BlockPool streamingBlockPool,
        const size_t *memIds,
        math::Vec3i *blockIds,
        BlockPool blockPool,
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
                blockPool.sdf[memId * blockVolume + idx]
                    = streamingBlockPool.sdf[id * blockVolume + idx];
                blockPool.weights[memId * blockVolume + idx]
                    = streamingBlockPool.weights[id * blockVolume + idx];
                blockPool.colors[memId * blockVolume + idx]
                    = streamingBlockPool.colors[id * blockVolume + idx];
                blockPool.gradients[memId * blockVolume + idx]
                    = streamingBlockPool.gradients[id * blockVolume + idx];
            }
        }
    }
} // namespace gpu

Volume::Volume(const FusionParameters &params) : params_{params} {}

void Volume::allocateData(const size_t maxBlockCount)
{
    // Pool data
    blockIds_.resize(maxBlockCount);
    tsdfBlockPool_.resize(blockVolume * maxBlockCount);
    weightsBlockPool_.resize(blockVolume * maxBlockCount);
    colorsBlockPool_.resize(blockVolume * maxBlockCount);
    gradientsBlockPool_.resize(blockVolume * maxBlockCount);

    // GPU hash data
    blockIdMap_.resize(maxBlockCount);
    memIdMap_.resize(maxBlockCount);

    // StreamingData
    for(size_t tId = 0; tId < threadCount; ++tId)
    {
        streamingBlockIds_[tId].resize(streamingBatchSize);
        streamingSdf_[tId].resize(blockVolume * streamingBatchSize);
        streamingWeights_[tId].resize(blockVolume * streamingBatchSize);
        streamingColors_[tId].resize(blockVolume * streamingBatchSize);
        streamingGradients_[tId].resize(blockVolume * streamingBatchSize);
        streamingMemIds_[tId].resize(streamingBatchSize);

        streamingBlockIdsHost_[tId].resize(streamingBatchSize);
        streamingSdfHost_[tId].resize(blockVolume * streamingBatchSize);
        streamingWeightsHost_[tId].resize(blockVolume * streamingBatchSize);
        streamingColorsHost_[tId].resize(blockVolume * streamingBatchSize);
        streamingGradientsHost_[tId].resize(blockVolume * streamingBatchSize);
        streamingMemIdsHost_[tId].resize(streamingBatchSize);
    }

    blockIdMapHost_.resize(maxBlockCount);
    memIdMapHost_.resize(maxBlockCount);
    blockCache_.resize(maxBlockCount);

    for(size_t tId = 0; tId < threadCount; ++tId)
    {
        gpuErrcheck(cudaStreamCreate(&loadStreams_[tId]));
        gpuErrcheck(cudaStreamCreate(&saveStreams_[tId]));
    }
    gpuErrcheck(cudaStreamCreate(&baseStream_));

    blockIds_.set(DEFAULT_BLOCK_ID, cudaStream_t{0});
}

Volume::~Volume()
{
    gpuErrcheck(cudaStreamDestroy(baseStream_));
    for(size_t tId = 0; tId < threadCount; ++tId)
    {
        gpuErrcheck(cudaStreamDestroy(saveStreams_[tId]));
        gpuErrcheck(cudaStreamDestroy(loadStreams_[tId]));
    }
}

BlockIdList Volume::streamBlocks(const BlockIdList &blockList, const bool allocateMissingBlocks)
{
    utils::Timer timer("Volume::streamBlocks()");

    BlockIdList allocatedBlocks;
    allocatedBlocks.reserve(blockList.size());

    if(allocateMissingBlocks)
    {
        const size_t newBlocks = addBlocks(blockList);
        for(const auto &id : blockList)
        {
            allocatedBlocks.emplace_back(id);
        }
        utils::Log::info(
            "Volume", "%zu new blocks, total block count : %zu", newBlocks, blockCount());
    }
    else
    {
        for(const auto &id : blockList)
        {
            if(find(id))
            {
                allocatedBlocks.emplace_back(id);
            }
        }
    }
    if(allocatedBlocks.size() == 0)
    {
        utils::Log::info("Volume", "No blocks to stream");
        return allocatedBlocks;
    }

    const auto memInfo = blockCache_.addBlocks(allocatedBlocks);

    std::vector<size_t> evictedIndices;
    evictedIndices.reserve(blockCache_.getEvictions().size());
    for(const auto &id : blockCache_.getEvictions())
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
        const auto &[memId, inCache] = memInfo[i];
        const auto blockId = allocatedBlocks[i];

        memIds_[blockId] = memId;
        if(!inCache)
        {
            blockIdsToLoad.push_back(blockId);
            indicesToLoad.push_back(memId);
        }
    }
    uploadBlocks(blockIdsToLoad, indicesToLoad);
    updateHash(blockIdsToLoad);

    return allocatedBlocks;
}

bool Volume::addBlock(const math::Vec3i &blockId)
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

size_t Volume::addBlocks(const std::vector<math::Vec3i> &blockIds)
{
    size_t ret = 0;
    for(size_t i = 0; i < blockIds.size(); ++i)
    {
        const auto blockId = blockIds[i];
        const bool allocated = find(blockId);
        if(!allocated)
        {
            auto blockPtr = std::make_unique<VoxelBlock>(params_.voxelRes, 0);
            voxelBlocks_[blockId] = std::move(blockPtr);
            memIds_[blockId] = -1;
            ret++;
        }
    }
    return ret;
}

void Volume::removeBlock(const math::Vec3i &blockId) noexcept
{
    voxelBlocks_.erase(blockId);
    memIds_.erase(blockId);
}

void Volume::removeBlocks(const std::vector<math::Vec3i> &blockIds) noexcept
{
    for(const auto &blockId : blockIds)
    {
        removeBlock(blockId);
    }
}

std::vector<BlockHeader> Volume::getHeaders(const BlockIdList &blockIds) const
{
    std::vector<BlockHeader> ret;
    ret.reserve(blockIds.size());
    for(const auto blockId : blockIds)
    {
        if(voxelBlocks_.find(blockId) == voxelBlocks_.end())
        {
            throw(std::runtime_error("Block not allocated"));
        }
        ret.emplace_back(blockId, memIds_.at(blockId));
    }
    return ret;
}

// TODO : replace with GPU MC
void Volume::exportMesh(const char *filename)
{
    static constexpr float wThr = 10.0f;
    static constexpr size_t maxTriangleCount = 3 * blockVolume;

    utils::Log::info("Volume", "Exporting triangles...");

    synchronizeBlocks();
    std::vector<math::Vec3f> points;
    std::vector<math::Vec3f> colors;

    std::vector<math::Vec3i> blockIds;
    blockIds.reserve(voxelBlocks_.size());

    // NOTE : Structured binding with OpenMP block after produces a compilation error
    for(const auto &entry : voxelBlocks_)
    {
        blockIds.emplace_back(entry.first);
    }

#pragma omp parallel
    {
        size_t triangleCount = 0;
        std::vector<math::Vec3f> tmpPoints(3 * maxTriangleCount);
        std::vector<math::Vec3f> tmpColors(3 * maxTriangleCount);
        std::vector<math::Vec3f> tmpNormals(3 * maxTriangleCount);

#pragma omp for
        for(size_t i = 0; i < blockIds.size(); ++i)
        {
            const math::Vec3i blockId = blockIds[i];
            const BlockId b0 = blockId + BlockId(0, 0, 0);
            const math::Vec3f org
                = float(blockSize) * params_.voxelRes * math::Vec3f(b0.x, b0.y, b0.z);

            const BlockId bxx = blockId + BlockId(1, 0, 0);
            const BlockId byy = blockId + BlockId(0, 1, 0);
            const BlockId bzz = blockId + BlockId(0, 0, 1);
            const BlockId bxy = blockId + BlockId(1, 1, 0);
            const BlockId bxz = blockId + BlockId(1, 0, 1);
            const BlockId byz = blockId + BlockId(0, 1, 1);
            const BlockId bxyz = blockId + BlockId(1, 1, 1);

            float *__restrict__ pointsPtr = reinterpret_cast<float *>(tmpPoints.data());
            float *__restrict__ colorsPtr = reinterpret_cast<float *>(tmpColors.data());
            float *__restrict__ normalsPtr = reinterpret_cast<float *>(tmpNormals.data());

            const float *__restrict__ tsdf = voxelBlocks_[blockId]->sdf().data();
            const float *__restrict__ weights = voxelBlocks_[blockId]->weights().data();
            const float *__restrict__ rgb = (float *) voxelBlocks_[blockId]->colors().data();
            const float *__restrict__ grad = (float *) voxelBlocks_[blockId]->gradients().data();

            float *xx = nullptr;
            float *yy = nullptr;
            float *zz = nullptr;
            float *xy = nullptr;
            float *xz = nullptr;
            float *yz = nullptr;
            float *xyz = nullptr;

            float *cxx = nullptr;
            float *cyy = nullptr;
            float *czz = nullptr;
            float *cxy = nullptr;
            float *cxz = nullptr;
            float *cyz = nullptr;
            float *cxyz = nullptr;

            float *gxx = nullptr;
            float *gyy = nullptr;
            float *gzz = nullptr;
            float *gxy = nullptr;
            float *gxz = nullptr;
            float *gyz = nullptr;
            float *gxyz = nullptr;

            float *wxx = nullptr;
            float *wyy = nullptr;
            float *wzz = nullptr;
            float *wxy = nullptr;
            float *wxz = nullptr;
            float *wyz = nullptr;
            float *wxyz = nullptr;

            if(voxelBlocks_.find(bxx) != voxelBlocks_.end())
            {
                xx = voxelBlocks_[bxx]->sdf().data();
                wxx = voxelBlocks_[bxx]->weights().data();
                cxx = (float *) voxelBlocks_[bxx]->colors().data();
                gxx = (float *) voxelBlocks_[bxx]->gradients().data();
            }

            if(voxelBlocks_.find(byy) != voxelBlocks_.end())
            {
                yy = voxelBlocks_[byy]->sdf().data();
                wyy = voxelBlocks_[byy]->weights().data();
                cyy = (float *) voxelBlocks_[byy]->colors().data();
                gyy = (float *) voxelBlocks_[byy]->gradients().data();
            }

            if(voxelBlocks_.find(bzz) != voxelBlocks_.end())
            {
                zz = voxelBlocks_[bzz]->sdf().data();
                wzz = voxelBlocks_[bzz]->weights().data();
                czz = (float *) voxelBlocks_[bzz]->colors().data();
                gzz = (float *) voxelBlocks_[bzz]->gradients().data();
            }

            if(voxelBlocks_.find(bxy) != voxelBlocks_.end())
            {
                xy = voxelBlocks_[bxy]->sdf().data();
                wxy = voxelBlocks_[bxy]->weights().data();
                cxy = (float *) voxelBlocks_[bxy]->colors().data();
                gxy = (float *) voxelBlocks_[bxy]->gradients().data();
            }

            if(voxelBlocks_.find(bxz) != voxelBlocks_.end())
            {
                xz = voxelBlocks_[bxz]->sdf().data();
                wxz = voxelBlocks_[bxz]->weights().data();
                cxz = (float *) voxelBlocks_[bxz]->colors().data();
                gxz = (float *) voxelBlocks_[bxz]->gradients().data();
            }

            if(voxelBlocks_.find(byz) != voxelBlocks_.end())
            {
                yz = voxelBlocks_[byz]->sdf().data();
                wyz = voxelBlocks_[byz]->weights().data();
                cyz = (float *) voxelBlocks_[byz]->colors().data();
                gyz = (float *) voxelBlocks_[byz]->gradients().data();
            }

            if(voxelBlocks_.find(bxyz) != voxelBlocks_.end())
            {
                xyz = voxelBlocks_[bxyz]->sdf().data();
                wxyz = voxelBlocks_[bxyz]->weights().data();
                cxyz = (float *) voxelBlocks_[bxyz]->colors().data();
                gxyz = (float *) voxelBlocks_[bxyz]->gradients().data();
            }

            triangleCount = mc::extractMesh(
                tsdf,
                xx,
                yy,
                zz,
                xy,
                xz,
                yz,
                xyz,
                rgb,
                cxx,
                cyy,
                czz,
                cxy,
                cxz,
                cyz,
                cxyz,
                grad,
                gxx,
                gyy,
                gzz,
                gxy,
                gxz,
                gyz,
                gxyz,
                weights,
                wxx,
                wyy,
                wzz,
                wxy,
                wxz,
                wyz,
                wxyz,
                pointsPtr,
                colorsPtr,
                normalsPtr,
                wThr,
                blockSize,
                params_.voxelRes,
                reinterpret_cast<const float *>(&org));

#pragma omp critical
            {
                if(triangleCount > 0)
                {
                    points.insert(
                        points.end(), tmpPoints.begin(), tmpPoints.begin() + 3 * triangleCount);
                    colors.insert(
                        colors.end(), tmpColors.begin(), tmpColors.begin() + 3 * triangleCount);
                }
            } // End of critical region
        }
    } // End of parallel region

    std::vector<math::Vec3<uint8_t>> exportColors;
    exportColors.reserve(colors.size());

    for(size_t i = 0; i < colors.size(); ++i)
    {
        const auto &c = colors[i];
        exportColors.emplace_back(
            uint8_t(255.0f * c.x), uint8_t(255.0f * c.y), uint8_t(255.0f * c.z));
    }

    std::vector<math::Vec3i> triangles;
    triangles.reserve(points.size() / 3);
    for(size_t i = 0; i < points.size() / 3; ++i)
    {
        triangles.emplace_back(3 * i, 3 * i + 1, 3 * i + 2);
    }

    io::Ply::saveSurface(filename, points, exportColors, triangles);
    utils::Log::info("Volume", "Done exporting triangles");
}

void Volume::synchronizeBlocks()
{
    std::vector<size_t> indicesToSave;
    for(const auto &[blockId, memId] : memIds_)
    {
        if(memId > 0)
        {
            indicesToSave.push_back(memId);
        }
    }
    saveBlocks(indicesToSave, false); // Save but don't invalidate
}

void Volume::saveBlocks(const std::vector<size_t> &indices, const bool invalidate)
{
    utils::Timer timer("Volume::saveBlocks()");
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
            blockPool(),
            streamingMemIds_[tId],
            streamingBlockIds_[tId],
            streamingBlockPool(tId),
            batchSize);
        streamingBlockIdsHost_[tId].downloadFrom(
            streamingBlockIds_[tId], batchSize, saveStreams_[tId]);
        streamingSdfHost_[tId].downloadFrom(
            streamingSdf_[tId], batchSize * blockVolume, saveStreams_[tId]);
        streamingWeightsHost_[tId].downloadFrom(
            streamingWeights_[tId], batchSize * blockVolume, saveStreams_[tId]);
        streamingColorsHost_[tId].downloadFrom(
            streamingColors_[tId], batchSize * blockVolume, saveStreams_[tId]);
        streamingGradientsHost_[tId].downloadFrom(
            streamingGradients_[tId], batchSize * blockVolume, saveStreams_[tId]);
        cudaStreamSynchronize(saveStreams_[tId]);

        // Check that all blocks are allocated
        for(size_t id = 0; id < batchSize; ++id)
        {
            const auto blockId = streamingBlockIdsHost_[tId][id];
            const auto &blockPtr = voxelBlocks_[blockId];
            if(blockPtr == nullptr)
            {
                throw std::runtime_error("Trying to save unallocated block");
            }
        }

        for(size_t id = 0; id < batchSize; ++id)
        {
            const auto blockId = streamingBlockIdsHost_[tId][id];
            const auto *__restrict__ sdfPtrH = streamingSdfHost_[tId] + id * blockVolume;
            const auto *__restrict__ weightsPtrH = streamingWeightsHost_[tId] + id * blockVolume;
            const auto *__restrict__ colorsPtrH = streamingColorsHost_[tId] + id * blockVolume;
            const auto *__restrict__ gradientsPtrH
                = streamingGradientsHost_[tId] + id * blockVolume;

            auto *__restrict__ sdfPtr = voxelBlocks_[blockId]->sdf().data();
            auto *__restrict__ weightsPtr = voxelBlocks_[blockId]->weights().data();
            auto *__restrict__ colorsPtr = voxelBlocks_[blockId]->colors().data();
            auto *__restrict__ gradientsPtr = voxelBlocks_[blockId]->gradients().data();

            if(invalidate)
            {
                memIds_[blockId] = -1;
            }
            memcpy(sdfPtr, sdfPtrH, blockVolume * sizeof(float));
            memcpy(weightsPtr, weightsPtrH, blockVolume * sizeof(float));
            memcpy(colorsPtr, colorsPtrH, blockVolume * sizeof(math::Vec3f));
            memcpy(gradientsPtr, gradientsPtrH, blockVolume * sizeof(math::Vec3f));
        }
    }
}

void Volume::uploadBlocks(
    const std::vector<math::Vec3i> &blockIds, const std::vector<size_t> &indices)
{
    utils::Timer timer("Volume::uploadBlocks()");
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
            const auto &blockPtr = voxelBlocks_[blockId];
            if(blockPtr == nullptr)
            {
                throw std::runtime_error("Trying to load unallocated block");
            }
        }

        for(size_t id = 0; id < batchSize; ++id)
        {
            auto blockId = blockIds[id];
            auto *__restrict__ sdfPtrH = streamingSdfHost_[tId] + id * blockVolume;
            auto *__restrict__ weightsPtrH = streamingWeightsHost_[tId] + id * blockVolume;
            auto *__restrict__ colorsPtrH = streamingColorsHost_[tId] + id * blockVolume;
            auto *__restrict__ gradientsPtrH = streamingGradientsHost_[tId] + id * blockVolume;

            const auto *__restrict__ sdfPtr = voxelBlocks_[blockId]->sdf().data();
            const auto *__restrict__ weightsPtr = voxelBlocks_[blockId]->weights().data();
            const auto *__restrict__ colorsPtr = voxelBlocks_[blockId]->colors().data();
            const auto *__restrict__ gradientsPtr = voxelBlocks_[blockId]->gradients().data();

            memcpy(sdfPtrH, sdfPtr, blockVolume * sizeof(float));
            memcpy(weightsPtrH, weightsPtr, blockVolume * sizeof(float));
            memcpy(colorsPtrH, colorsPtr, blockVolume * sizeof(math::Vec3f));
            memcpy(gradientsPtrH, gradientsPtr, blockVolume * sizeof(math::Vec3f));
        }

        streamingBlockIdsHost_[tId].uploadTo(streamingBlockIds_[tId], batchSize, loadStreams_[tId]);
        streamingMemIdsHost_[tId].uploadTo(streamingMemIds_[tId], batchSize, loadStreams_[tId]);
        streamingSdfHost_[tId].uploadTo(
            streamingSdf_[tId], batchSize * blockVolume, loadStreams_[tId]);
        streamingWeightsHost_[tId].uploadTo(
            streamingWeights_[tId], batchSize * blockVolume, loadStreams_[tId]);
        streamingColorsHost_[tId].uploadTo(
            streamingColors_[tId], batchSize * blockVolume, loadStreams_[tId]);
        streamingGradientsHost_[tId].uploadTo(
            streamingGradients_[tId], batchSize * blockVolume, loadStreams_[tId]);

        gpu::copyBlocksFromLoad<<<streamingBatchSize, 256, 0, loadStreams_[tId]>>>(
            streamingBlockIds_[tId],
            streamingBlockPool(tId),
            streamingMemIds_[tId],
            blockIds_,
            blockPool(),
            batchSize);
        cudaStreamSynchronize(loadStreams_[tId]);
    }
}

void Volume::updateHash(const std::vector<math::Vec3i> &blockIds)
{
    utils::Timer("Volume::updateHash()");
    if(blockIds.empty())
    {
        return;
    }

    const size_t blockCount = blockIds.size();
    std::vector<math::Vec3i> sortedIds(blockIds);
    std::sort(sortedIds.begin(), sortedIds.end(), [this](const auto &x, const auto &y) {
        return utils::hashIndex(x, hashSize) < utils::hashIndex(y, hashSize);
    });

    std::vector<math::Vec2i> offsets(hashSize);
    std::fill(offsets.begin(), offsets.end(), math::Vec2i{0, 0});

    size_t currIndex = utils::hashIndex(sortedIds[0], hashSize);
    size_t currCount = 1;
    size_t currOffset = 0;
    for(size_t i = 1; i < blockCount; ++i)
    {
        const size_t index = utils::hashIndex(sortedIds[i], hashSize);
        if(index < currIndex)
        {
            throw std::runtime_error("List not sorted");
        }

        if(index > currIndex || i == (blockCount - 1))
        {
            offsets[currIndex] = {int(currOffset), int(currCount)};
            currOffset = i;
            currIndex = index;
            currCount = 0;
        }
        currCount++;
    }

    for(size_t i = 0; i < blockCount; ++i)
    {
        memIdMapHost_[i] = memIds_[sortedIds[i]];
    }
    std::copy(sortedIds.begin(), sortedIds.end(), blockIdMapHost_.begin());
    std::copy(offsets.begin(), offsets.end(), blockIdOffsetsHost_.begin());

    blockIdMapHost_.uploadTo(blockIdMap_, blockCount, baseStream_);
    memIdMapHost_.uploadTo(memIdMap_, blockCount, baseStream_);
    blockIdOffsetsHost_.uploadTo(blockIdOffsets_, hashSize, baseStream_);
    cudaStreamSynchronize(baseStream_);
}
} // namespace fusion