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
#include "utils/Ptr.hpp"
#include "fusion/BlockUtils.hpp"

namespace fusion
{
struct Voxel
{
    float sdf;
    float weight;
    float sigma;
    math::Vec3f color;
};

struct BlockHeader
{
    math::Vec3i blockId;
    size_t memId;
};

class VoxelBlock
{
  public:
    VoxelBlock() = delete;
    VoxelBlock(const float voxelSize, const size_t level = 0);
    VoxelBlock(const VoxelBlock&) = default;
    VoxelBlock(VoxelBlock&&) = default;
    VoxelBlock& operator=(const VoxelBlock&) = default;
    VoxelBlock& operator=(VoxelBlock&&) = default;

    ~VoxelBlock();

    auto& voxels() { return voxels_; }
    const auto& voxels() const { return voxels_; }

    inline auto finestVoxelSize() const { return voxelSize_; }
    inline auto voxelSize() const { return float(1 << level_) * voxelSize_; }

    static size_t getMemorySize() { return blockVolume * sizeof(decltype(voxels_)::value_type); }

  private:
    const float voxelSize_;
    const size_t level_;

    CpuPtr<Voxel, false> voxels_;
};
} // namespace fusion