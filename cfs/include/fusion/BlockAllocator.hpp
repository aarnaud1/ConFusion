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

#include <vector>

#include "fusion/VoxelBlock.hpp"

namespace cfs
{
class BlockAllocator
{
  public:
    BlockAllocator() = default;
    BlockAllocator(const size_t n)
    {
        const size_t chunkCount = utils::div_up(n, chunkSize);
        for(size_t i = 0; i < chunkCount; ++i)
        {
            allocateNewChunk();
        }
    }
    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator(BlockAllocator&&) = default;

    BlockAllocator& operator=(const BlockAllocator&) = delete;
    BlockAllocator& operator=(BlockAllocator&&) = default;

    VoxelBlock createBlock(const float voxelSize, const size_t level = 0)
    {
        if(top_ == size_)
        {
            allocateNewChunk();
        }
        const size_t off = top_ / chunkSize;
        const size_t pos = top_ % chunkSize;

        VoxelBlock ret{};
        ret.voxelSize_ = voxelSize;
        ret.level_ = level;

        ret.sdf_ = sdf_[off].data() + pos * blockVolume;
        ret.weights_ = weights_[off].data() + pos * blockVolume;
        ret.colors_ = colors_[off].data() + pos * blockVolume;
        ret.gradients_ = gradients_[off].data() + pos * blockVolume;

        top_++;

        return ret;
    }
    std::vector<VoxelBlock> createBlocks(
        const size_t /*blockCount*/, const float /*voxelSize*/, const size_t /*level = 0*/)
    {
        throw std::runtime_error("BlockAllocator::createBlocks() not implemented yet");
    }

  private:
    static constexpr size_t chunkSize = 256;

    size_t size_{0};
    size_t top_{0};

    std::vector<CpuPtr<float, false>> sdf_;
    std::vector<CpuPtr<float, false>> weights_;
    std::vector<CpuPtr<math::Vec3f, false>> colors_;
    std::vector<CpuPtr<math::Vec3f, false>> gradients_;

    void allocateNewChunk()
    {
        size_ += chunkSize;

        sdf_.emplace_back(chunkSize * blockVolume);
        weights_.emplace_back(chunkSize * blockVolume);
        colors_.emplace_back(chunkSize * blockVolume);
        gradients_.emplace_back(chunkSize * blockVolume);

        std::fill(sdf_.back().begin(), sdf_.back().end(), INVALID_TSDF);
        std::fill(weights_.back().begin(), weights_.back().end(), DEFAULT_WEIGHT);
        std::fill(colors_.back().begin(), colors_.back().end(), DEFAULT_COLOR);
        std::fill(gradients_.back().begin(), gradients_.back().end(), DEFAULT_GRADIENT);
    }
};
} // namespace cfs