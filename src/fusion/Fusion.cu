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

#include "fusion/Fusion.hpp"

namespace fusion
{
__global__ void getBlocksIntersectingKernel(
    const geometry::Vec3f* __restrict__ points,
    const geometry::Vec3f* __restrict__ normals,
    const geometry::Vec3f* __restrict__ triangles,
    geometry::Vec3i* __restrict__ blockIds,
    size_t* __restrict__ blockCount,
    const size_t maxBlockCount,
    const geometry::Mat4d m,
    const size_t pointCount,
    const size_t triangleCount,
    const float voxelRes,
    const float tau)
{}

Fusion::Fusion(const FusionParameters& params) : params_{params} {}

void Fusion::integrateFrame(
    FramePtr& frame, const geometry::Mat4d& pose, const geometry::Mat4d& k, const float dpethScale)
{}

void Fusion::integrateFrames(
    std::vector<FramePtr>& frames,
    const std::vector<geometry::Mat4d>& poses,
    const geometry::Mat3d& k,
    const float depthScale)
{}

BlockIdList Fusion::getBlocksIntersecting(const FramePtr& frame, const geometry::Mat4d& pose) {}
BlockIdList Fusion::getBlocksintersecting(
    const std::vector<FramePtr>& frames, const geometry::Mat4d& pose)
{}
} // namespace fusion