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
#include "geometry/geometry.hpp"
#include "fusion/DepthMap.hpp"
#include "fusion/RenderDepthMap.hpp"
#include "fusion/VoxelBlock.hpp"
#include "fusion/Volume.hpp"

#include <memory>
#include <vector>
#include <string>

namespace fusion
{
struct FusionParameters
{
  float voxelRes; // Finest voxel size
  float tau;      // Truncation distance
};

class Fusion
{
public:
  using FramePtr = std::shared_ptr<RenderDepthMap>;

  Fusion(const FusionParameters& params);

  void integrateFrame(
      FramePtr& frame,
      const geometry::Mat4d& pose,
      const geometry::Mat4d& k,
      const float dpethScale);
  void integrateFrames(
      std::vector<FramePtr>& frames,
      const std::vector<geometry::Mat4d>& poses,
      const geometry::Mat3d& k,
      const float depthScale);

  void dumpBlocks(const std::string& dir);
  void preloadBlocks(const std::string& dir);

private:
  FusionParameters params_;

  BlockIdList getBlocksIntersecting(const FramePtr& frame, const geometry::Mat4d& pose);
  BlockIdList getBlocksintersecting(
      const std::vector<FramePtr>& frames, const geometry::Mat4d& pose);
};
} // namespace fusion
