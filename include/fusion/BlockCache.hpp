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

#include "fusion/VoxelBlock.hpp"

#include <list>
#include <unordered_map>
#include <vector>
#include <set>

#define MAX_VOLUME_SIZE 1024 // TODO : Estimate it from available GPU memory

namespace fusion
{
class BlockCache
{
public:
  BlockCache() = delete;
  BlockCache(const size_t capacity);
  BlockCache(const BlockCache&) = delete;
  BlockCache(BlockCache&&) = delete;
  BlockCache& operator=(const BlockCache) = delete;
  BlockCache& operator=(BlockCache&&) = delete;

  void clear();

  size_t addBlock(const BlockId& blockId);
  std::vector<size_t> addBlocks(const std::vector<BlockId>& blockIds);

  auto& getEvictions() { return evictionList_; }
  const auto& getEvictions() const { return evictionList_; }

  auto capacity() const { return capacity_; }

private:
  struct CacheItem
  {
    BlockId blockId;
    size_t memId;
  };

  size_t capacity_;

  std::list<CacheItem> cache_;
  std::unordered_map<uint64_t, std::list<CacheItem>::iterator> map_;
  std::vector<size_t> availableIds_;

  std::set<BlockId> evictionList_;
};
} // namespace fusion