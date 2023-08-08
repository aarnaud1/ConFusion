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
#include "utils/Ptr.hpp"

namespace fusion
{
static constexpr size_t blockSize = 16;
static constexpr size_t blockVolume = blockSize * blockSize * blockSize;
static constexpr size_t blockShift = 4;

using BlockId = geometry::Vec3i;

namespace utils
{
  __host__ __device__ __forceinline__ static int mod(const int a, const int b)
  {
    return (a % b + b) % b;
  }
  __host__ __device__ __forceinline__ static geometry::Vec3i Mod(
      const geometry::Vec3i& index, const int val)
  {
    return {mod(index.x, val), mod(index.y, val), mod(index.z, val)};
  }
  __host__ __device__ __forceinline__ static int div(const int a, const int b)
  {
    return (a - mod(a, b)) / b;
  }
  __host__ __device__ __forceinline__ static geometry::Vec3i div(
      const geometry::Vec3i& index, const int val)
  {
    return {div(index.x, val), div(index.y, val), div(index.z, val)};
  }
  __host__ __device__ __forceinline__ static geometry::Vec3i getId(
      const geometry::Vec3f& v, const float voxelRes)
  {
    const int x = static_cast<int>(std::floor(v.x / voxelRes)) >> blockShift;
    const int y = static_cast<int>(std::floor(v.y / voxelRes)) >> blockShift;
    const int z = static_cast<int>(std::floor(v.z / voxelRes)) >> blockShift;
    return {x, y, z};
  }
  __host__ __device__ __forceinline__ static geometry::Vec3i getVoxelAbsolutePos(
      const geometry::Vec3i& blockId, const geometry::Vec3i& voxelId)
  {
    return int(blockSize) * blockId + voxelId;
  }
  __host__ __device__ __forceinline__ static geometry::Vec3f getVoxelPos(
      const geometry::Vec3i& id, const float voxelRes)
  {
    return voxelRes
           * geometry::Vec3f(
               static_cast<float>(id.x), static_cast<float>(id.y), static_cast<float>(id.z));
  }
  __host__ __device__ __forceinline__ static geometry::Vec3i getVoxelId(
      const geometry::Vec3f& p, const float voxelRes)
  {
    const int x = mod(static_cast<int>(std::floor(p.x / voxelRes)), blockSize);
    const int y = mod(static_cast<int>(std::floor(p.y / voxelRes)), blockSize);
    const int z = mod(static_cast<int>(std::floor(p.z / voxelRes)), blockSize);
    return {x, y, z};
  }
  __host__ __device__ __forceinline__ static uint64_t encode(const BlockId& blockId)
  {
    static constexpr uint32_t mask = 0x1ffff;
    return uint64_t(reinterpret_cast<uint32_t>(blockId.x & mask))
           | uint64_t(reinterpret_cast<uint32_t>(blockId.y & mask)) << 21
           | uint64_t(reinterpret_cast<uint32_t>(blockId.z & mask)) << 42;
  }
} // namespace utils

struct Voxel
{
  float sdf;
  float weight;
  float sigma;
  geometry::Vec3f color;
};

struct BlockHeader
{
  geometry::Vec3i blockId;
  int memId = -1;
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

private:
  const float voxelSize_;
  const size_t level_;

  CpuPtr<Voxel, false> voxels_;
};
} // namespace fusion