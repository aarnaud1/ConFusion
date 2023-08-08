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

#include "geometry/geometry.hpp"
#include "fusion/VoxelBlock.hpp"

#include <vector>
#include <memory>
#include <unordered_map>

namespace fusion
{
struct VolumeHash
{
  static constexpr size_t p1 = 73856093;
  static constexpr size_t p2 = 19349663;
  static constexpr size_t p3 = 83492791;

  inline std::size_t operator()(const geometry::Vec3i& key) const
  {
    return (key.x * p1 ^ key.y * p2 ^ key.z * p3);
  }
};

typedef std::vector<geometry::Vec3i> BlockIdList;
typedef std::unordered_map<geometry::Vec3i, std::unique_ptr<VoxelBlock>, VolumeHash> BlockMap;
typedef std::vector<std::unique_ptr<VoxelBlock>> BlockList;

class Volume
{
public:
  Volume() = delete;
  Volume(const float voxelSize);
  Volume(const Volume&) = delete;
  Volume(Volume&&) = delete;
  Volume& operator=(const Volume&) = delete;
  Volume& operator=(Volume&&) = delete;

  bool addBlock(const geometry::Vec3i& blockId);
  size_t addBlocks(const std::vector<geometry::Vec3i>& blockIds);

  void removeBlock(const geometry::Vec3i& blockId) noexcept;
  void removeBlocks(const std::vector<geometry::Vec3i>& blockIds) noexcept;

  inline bool find(const geometry::Vec3i& blockId) const noexcept
  {
    return voxelBlocks_.find(blockId) != voxelBlocks_.end();
  }
  VoxelBlock& getBlock(const geometry::Vec3i& blockId) { return *voxelBlocks_.at(blockId); }
  const VoxelBlock& getBlock(const geometry::Vec3i& blockId) const
  {
    return *voxelBlocks_.at(blockId);
  }

  inline size_t size() const { return voxelBlocks_.size(); }

private:
  const float voxelSize_;

  BlockMap voxelBlocks_;
};
} // namespace fusion