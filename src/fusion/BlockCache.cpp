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

#include "fusion/BlockCache.hpp"

#include <numeric>

namespace fusion
{
BlockCache::BlockCache(const size_t capacity) : capacity_{capacity}
{
  availableIds_.resize(capacity_);
  std::iota(availableIds_.begin(), availableIds_.end(), 0);
}

void BlockCache::clear()
{
  map_.clear();
  cache_.clear();
  std::iota(availableIds_.begin(), availableIds_.end(), 0);
}

size_t BlockCache::addBlock(const BlockId& blockId)
{
  const uint64_t uuid = utils::encode(blockId);
  const auto it = map_.find(uuid);

  CacheItem entry;
  entry.blockId = blockId;

  if(it == map_.end())
  {
    if(availableIds_.size() > 0)
    {
      const size_t memId = availableIds_.back();
      availableIds_.pop_back();
      entry.memId = memId;
    }
    else
    {
      const auto last = cache_.back();
      evictionList_.insert(last.blockId);
      entry.memId = last.memId;
      cache_.erase(map_[uuid]);
    }
  }
  cache_.push_front(entry);
  map_[uuid] = cache_.begin();

  return entry.memId;
}

std::vector<size_t> BlockCache::addBlocks(const std::vector<BlockId>& blockIds)
{
  std::vector<size_t> ret;
  ret.reserve(blockIds.size());
  for(const auto& blockId : blockIds)
  {
    ret.emplace_back(addBlock(blockId));
  }
  return ret;
}
} // namespace fusion