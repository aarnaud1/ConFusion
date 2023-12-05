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

#include "common.hpp"
#include "utils/Ptr.hpp"
#include "utils/Img.hpp"
#include "math/geometry.hpp"
#include "fusion/FusionParameters.hpp"
#include "fusion/DepthFrame.hpp"
#include "fusion/VoxelBlock.hpp"
#include "fusion/Volume.hpp"

#include <memory>
#include <vector>
#include <string>
#include <array>

namespace fusion
{
using CpuFrameType = CpuFrame<false>;
using StagingFrameType = CpuFrame<true>;
using GpuFrameType = GpuFrame;

class Fusion final
{
  public:
    static constexpr size_t maxBatchSize = 16;
    static constexpr size_t maxBlockCount = 10000; // TODO : compute with available memory

    Fusion(const FusionParameters& params);
    Fusion(const Fusion&) = default;
    Fusion(Fusion&&) = delete;
    Fusion& operator=(const Fusion&) = delete;
    Fusion& operator=(Fusion&&) = default;

    ~Fusion();

    void integrateFrames(
        const std::vector<CpuFrameType>& frames,
        const std::vector<math::Mat4d>& poses,
        const float depthScale);

  private:
    FusionParameters params_;

    const size_t maxBlockCount_ = 100000; // TODO : compute

    Volume volume_;

    cudaStream_t mainStream_;
    std::array<cudaStream_t, maxBatchSize> subStreams_;
    std::array<cudaEvent_t, maxBatchSize> fireEvents_;
    std::array<cudaEvent_t, maxBatchSize> waitEvents_;

    // GPU buffers
    std::array<GpuFrameType, maxBatchSize> frames_;
    std::array<GpuPtr<uint32_t>, maxBatchSize> blockCounts_;
    std::array<GpuPtr<uint64_t>, maxBatchSize> blockList_;

    // Tmp data
    std::array<GpuPtr<uint64_t>, maxBatchSize> sortedBlockIds_;

    size_t tmpDataSize_{0};
    GpuPtr<uint8_t> tmpData_;

    // Staging buffers
    std::array<StagingFrameType, maxBatchSize> framesHost_;
    std::array<CpuPtr<uint32_t, true>, maxBatchSize> blockCountsHost_;
    std::array<CpuPtr<uint64_t, true>, maxBatchSize> blockListHost_;

    inline void notifySubStreamsStart(const size_t batchSize)
    {
        for(size_t batchId = 0; batchId < batchSize; ++batchId)
        {
            gpuErrcheck(cudaEventRecord(fireEvents_[batchId], mainStream_));
        }
    }
    inline void waitForSubStreams(const size_t batchSize)
    {
        for(size_t batchId = 0; batchId < batchSize; ++batchId)
        {
            gpuErrcheck(cudaStreamWaitEvent(mainStream_, waitEvents_[batchId]));
        }
    }
    inline void waitForStart(const size_t batchId)
    {
        gpuErrcheck(cudaStreamWaitEvent(subStreams_[batchId], fireEvents_[batchId]));
    }
    inline void notifyWorkDone(const size_t batchId)
    {
        gpuErrcheck(cudaEventRecord(waitEvents_[batchId], subStreams_[batchId]));
    }

    void prepareFrames(
        const std::vector<CpuFrameType>& frames,
        const std::vector<math::Mat4d>& poses,
        const size_t batchOffset,
        const size_t batchSize,
        const float depthScale);

    void computeIntersectingBlocks(
        const std::vector<math::Mat4d>& poses, const size_t batchOffset, const size_t batchSize);

    std::vector<math::Vec3i> getIntersectingBlocks(const size_t batchSize);

    void allocateMemory(const size_t width, const size_t height);

    static std::tuple<size_t, size_t> getAvailableGpuMemory();
};
} // namespace fusion
