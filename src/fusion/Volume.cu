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

namespace fusion
{
Volume::Volume(const float voxelSize) : voxelSize_{voxelSize} {}

bool Volume::addBlock(const geometry::Vec3i& blockId)
{
  const bool existing = find(blockId);
  if(!existing)
  {
    voxelBlocks_[blockId] = std::make_unique<VoxelBlock>(voxelSize_, 0);
  }
  return existing;
}
size_t Volume::addBlocks(const std::vector<geometry::Vec3i>& blockIds)
{
  size_t ret = 0;
  for(const auto& blockId : blockIds)
  {
    ret += addBlock(blockId) ? 1 : 0;
  }
  return ret;
}

void Volume::removeBlock(const geometry::Vec3i& blockId) noexcept { voxelBlocks_.erase(blockId); }
void Volume::removeBlocks(const std::vector<geometry::Vec3i>& blockIds) noexcept
{
  for(const auto& blockId : blockIds)
  {
    voxelBlocks_.erase(blockId);
  }
}

} // namespace fusion