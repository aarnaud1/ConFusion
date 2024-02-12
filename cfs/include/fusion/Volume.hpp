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

#include "fusion/BlockCache.hpp"
#include "fusion/DepthFrame.hpp"
#include "fusion/FusionParameters.hpp"
#include "fusion/VoxelBlock.hpp"
#include "fusion/BlockAllocator.hpp"
#include "math/geometry.hpp"
#include "utils/Ptr.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace cfs
{
// TODO : std::unorderes_map is really not a good choicehere, we should consider using some third
// party hash map library or implement our own block map class.
typedef std::vector<math::Vec3i> BlockIdList;
typedef std::unordered_map<math::Vec3i, VoxelBlock, VolumeHash> BlockMap;
typedef std::unordered_map<math::Vec3i, int, VolumeHash> BlockMemoryMap;
typedef std::vector<std::unique_ptr<VoxelBlock>> BlockList;

struct BlockPool
{
    float* sdf;
    float* weights;
    math::Vec3f* colors;
    math::Vec3f* gradients;
};

class Volume
{
  public:
    Volume() = delete;
    Volume(const FusionParameters& params);
    Volume(const Volume&) = delete;
    Volume(Volume&&) = delete;
    Volume& operator=(const Volume&) = delete;
    Volume& operator=(Volume&&) = delete;

    ~Volume();

    void allocateData(const size_t maxBlocks);

    bool addBlock(const math::Vec3i& blockId);

    size_t addBlocks(const std::vector<math::Vec3i>& blockIds);

    void removeBlock(const math::Vec3i& blockId) noexcept;

    void removeBlocks(const std::vector<math::Vec3i>& blockIds) noexcept;

    BlockIdList streamBlocks(const BlockIdList& blockList, const bool allocateMissingBlocks = true);

    inline bool find(const math::Vec3i& blockId) const noexcept
    {
        return voxelBlocks_.find(blockId) != voxelBlocks_.end();
    }

    VoxelBlock& getBlock(const math::Vec3i& blockId) { return voxelBlocks_.at(blockId); }

    const VoxelBlock& getBlock(const math::Vec3i& blockId) const
    {
        return voxelBlocks_.at(blockId);
    }

    std::vector<BlockHeader> getHeaders(const BlockIdList& blockIds) const;

    void exportMesh(const char* filename);

    inline auto& blockIds() { return blockIds_; }
    inline const auto& blockIds() const { return blockIds_; }

    inline BlockPool blockPool()
    {
        BlockPool ret;
        ret.sdf = tsdfBlockPool_.data();
        ret.weights = weightsBlockPool_.data();
        ret.colors = colorsBlockPool_.data();
        ret.gradients = gradientsBlockPool_.data();
        return ret;
    }

    inline auto& blockIdMap() { return blockIdMap_; }
    inline const auto& blockIdMap() const { return blockIdMap_; }

    inline auto& memIdMap() { return memIdMap_; }
    inline const auto& memIdMap() const { return memIdMap_; }

    inline auto& blockIdOffsets() { return blockIdOffsets_; }
    inline const auto& blockIdOffsets() const { return blockIdOffsets_; }

    inline size_t blockCount() const { return voxelBlocks_.size(); }

    inline size_t getHashSize() const { return hashSize; }

  private:
    static constexpr size_t threadCount = 4;
    static constexpr size_t streamingBatchSize = 64;

    const FusionParameters& params_;

    std::array<cudaStream_t, threadCount> loadStreams_;
    std::array<cudaStream_t, threadCount> saveStreams_;
    cudaStream_t baseStream_;

    BlockMap voxelBlocks_;
    BlockMemoryMap memIds_;

    BlockCache blockCache_;
    BlockAllocator blockAllocator_;

    // Pool data
    GpuPtr<math::Vec3i> blockIds_;
    GpuPtr<float> tsdfBlockPool_;
    GpuPtr<float> weightsBlockPool_;
    GpuPtr<math::Vec3f> colorsBlockPool_;
    GpuPtr<math::Vec3f> gradientsBlockPool_;

    // GPU hash data
    static constexpr size_t hashSize = 2048;
    GpuPtr<math::Vec3i> blockIdMap_;
    GpuPtr<int> memIdMap_;
    GpuPtr<math::Vec2i> blockIdOffsets_{hashSize};

    // Streaming data
    std::array<GpuPtr<math::Vec3i>, threadCount> streamingBlockIds_;
    std::array<GpuPtr<float>, threadCount> streamingSdf_;
    std::array<GpuPtr<float>, threadCount> streamingWeights_;
    std::array<GpuPtr<math::Vec3f>, threadCount> streamingColors_;
    std::array<GpuPtr<math::Vec3f>, threadCount> streamingGradients_;
    std::array<GpuPtr<size_t>, threadCount> streamingMemIds_;

    std::array<CpuPtr<math::Vec3i, true>, threadCount> streamingBlockIdsHost_;
    std::array<CpuPtr<float, true>, threadCount> streamingSdfHost_;
    std::array<CpuPtr<float, true>, threadCount> streamingWeightsHost_;
    std::array<CpuPtr<math::Vec3f, true>, threadCount> streamingColorsHost_;
    std::array<CpuPtr<math::Vec3f, true>, threadCount> streamingGradientsHost_;
    std::array<CpuPtr<size_t, true>, threadCount> streamingMemIdsHost_;

    CpuPtr<math::Vec3i, true> blockIdMapHost_;
    CpuPtr<int, true> memIdMapHost_;
    CpuPtr<math::Vec2i, true> blockIdOffsetsHost_{hashSize};

    inline BlockPool streamingBlockPool(const size_t i)
    {
        BlockPool ret;
        ret.sdf = streamingSdf_[i].data();
        ret.weights = streamingWeights_[i].data();
        ret.colors = streamingColors_[i].data();
        ret.gradients = streamingGradients_[i].data();
        return ret;
    }

    void synchronizeBlocks();
    void saveBlocks(const std::vector<size_t>& indices, const bool invalidate = true);
    void uploadBlocks(const std::vector<math::Vec3i>& blockIds, const std::vector<size_t>& indices);
    void updateHash(const std::vector<math::Vec3i>& blockIds);
};
} // namespace cfs