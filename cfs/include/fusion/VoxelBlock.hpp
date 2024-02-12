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

#include "fusion/BlockUtils.hpp"
#include "math/geometry.hpp"

#define INVALID_TSDF std::numeric_limits<float>::max()
#define DEFAULT_WEIGHT 0.0f
#define DEFAULT_COLOR                                                                              \
    math::Vec3f { 0.0f, 0.0f, 0.0f }
#define DEFAULT_GRADIENT                                                                           \
    math::Vec3f { 0.0f, 0.0f, 0.0f }

namespace cfs
{
struct BlockHeader
{
    math::Vec3i blockId;
    size_t memId;

    BlockHeader() = default;
    inline BlockHeader(const math::Vec3i _blockId, const size_t _memId)
        : blockId{_blockId}, memId{_memId}
    {}
};

class VoxelBlock
{
  public:
    VoxelBlock() = default;
    VoxelBlock(const VoxelBlock&) = delete;
    VoxelBlock(VoxelBlock&&) = default;
    VoxelBlock& operator=(const VoxelBlock&) = delete;
    VoxelBlock& operator=(VoxelBlock&&) = default;

    ~VoxelBlock();

    inline auto* sdf() { return sdf_; }
    inline const auto* sdf() const { return sdf_; }

    inline auto weights() { return weights_; }
    inline const auto* weights() const { return weights_; }

    inline auto* colors() { return colors_; }
    inline const auto* colors() const { return colors_; }

    inline auto* gradients() { return gradients_; }
    inline const auto* gradients() const { return gradients_; }

    inline auto finestVoxelSize() const { return voxelSize_; }
    inline auto voxelSize() const { return float(1 << level_) * voxelSize_; }

    static size_t getMemorySize()
    {
        return blockVolume * sizeof(float) + blockVolume * sizeof(float)
               + blockVolume * sizeof(math::Vec3f) + blockVolume * sizeof(math::Vec3f);
    }

  private:
    friend class BlockAllocator;

    float voxelSize_{0.0f};
    size_t level_{0};

    float* sdf_{nullptr};
    float* weights_{nullptr};
    math::Vec3f* colors_{nullptr};
    math::Vec3f* gradients_{nullptr};
};
} // namespace cfs