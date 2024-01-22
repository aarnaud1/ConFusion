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
#include "fusion/VoxelBlock.hpp"

#include <list>
#include <set>
#include <unordered_map>
#include <vector>

namespace cfs
{
class BlockCache
{
  public:
    BlockCache() = default;
    BlockCache(const size_t capacity);
    BlockCache(const BlockCache&) = delete;
    BlockCache(BlockCache&&) = delete;
    BlockCache& operator=(const BlockCache) = delete;
    BlockCache& operator=(BlockCache&&) = delete;

    void resize(const size_t capacity);

    void clear();

    std::pair<size_t, bool> addBlock(const BlockId& blockId);
    std::vector<std::pair<size_t, bool>> addBlocks(const std::vector<BlockId>& blockIds);

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
    std::unordered_map<BlockId, std::list<CacheItem>::iterator, VolumeHash> map_;
    std::vector<size_t> availableIds_;

    std::set<BlockId> evictionList_;
};
} // namespace cfs