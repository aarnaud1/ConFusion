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

#pragma once

#include "math/geometry.hpp"
#include "fusion/FusionParameters.hpp"
#include "fusion/VoxelBlock.hpp"
#include "fusion/DepthFrame.hpp"
#include "fusion/BlockCache.hpp"
#include "utils/Ptr.hpp"

#include <vector>
#include <memory>
#include <unordered_map>

namespace fusion
{
typedef std::vector<math::Vec3i> BlockIdList;
typedef std::unordered_map<math::Vec3i, std::unique_ptr<VoxelBlock>, VolumeHash> BlockMap;
typedef std::unordered_map<math::Vec3i, int, VolumeHash> BlockMemoryMap;
typedef std::vector<std::unique_ptr<VoxelBlock>> BlockList;

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

    void streamBlocks(const BlockIdList& blockList);

    inline bool find(const math::Vec3i& blockId) const noexcept
    {
        return voxelBlocks_.find(blockId) != voxelBlocks_.end();
    }
    VoxelBlock& getBlock(const math::Vec3i& blockId) { return *voxelBlocks_.at(blockId); }
    const VoxelBlock& getBlock(const math::Vec3i& blockId) const
    {
        return *voxelBlocks_.at(blockId);
    }

    inline size_t blockCount() const { return voxelBlocks_.size(); }

  private:
    static constexpr size_t threadCount = 4;
    static constexpr size_t streamingBatchSize = 64;

    const FusionParameters& params_;

    std::array<cudaStream_t, threadCount> loadStreams_;
    std::array<cudaStream_t, threadCount> saveStreams_;

    BlockMap voxelBlocks_;
    BlockMemoryMap memIds_;

    BlockCache blockCache_;

    // Pool data
    GpuPtr<math::Vec3i> blockIds_;
    GpuPtr<Voxel> blockPool_;

    // Streaming data
    std::array<GpuPtr<math::Vec3i>, threadCount> streamingBlockIds_;
    std::array<GpuPtr<Voxel>, threadCount> streamingBlocks_;
    std::array<GpuPtr<size_t>, threadCount> streamingMemIds_;

    std::array<CpuPtr<math::Vec3i, true>, threadCount> streamingBlockIdsHost_;
    std::array<CpuPtr<Voxel, true>, threadCount> streamingBlocksHost_;
    std::array<CpuPtr<size_t, true>, threadCount> streamingMemIdsHost_;

    void saveBlocks(const std::vector<size_t>& indices);
    void streamBlocks(const std::vector<math::Vec3i>& blockIds, const std::vector<size_t>& indices);
};
} // namespace fusion