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

#include "attributes.hpp"
#include "math/geometry.hpp"
#include "utils/Ptr.hpp"

namespace fusion
{
static constexpr size_t blockSize = 16;
static constexpr size_t blockVolume = blockSize * blockSize * blockSize;
static constexpr size_t blockShift = 4;

using BlockId = math::Vec3i;

#define DEFAULT_BLOCK_ID                                                                           \
    math::Vec3i { ~int32_t(0) }

struct VolumeHash
{
    static constexpr size_t p1 = 73856093;
    static constexpr size_t p2 = 19349663;
    static constexpr size_t p3 = 83492791;

    ATTR_HOST_INL std::size_t operator()(const math::Vec3i& key) const
    {
        return (key.x * p1 ^ key.y * p2 ^ key.z * p3);
    }

    static ATTR_HOST_DEV_INL size_t hashIndex(const math::Vec3i& key)
    {
        return (key.x * p1 ^ key.y * p2 ^ key.z * p3);
    }
};

namespace utils
{
    ATTR_HOST_DEV_INL static size_t hashIndex(const math::Vec3i& id, const size_t n)
    {
        return VolumeHash::hashIndex(id) % n;
    }

    ATTR_HOST_DEV_INL static bool isValidBlock(const math::Vec3i& id)
    {
        return id != DEFAULT_BLOCK_ID;
    }

    ATTR_HOST_DEV_INL static int mod(const int a, const int b) { return (a % b + b) % b; }

    ATTR_HOST_DEV_INL static math::Vec3i Mod(const math::Vec3i& index, const int val)
    {
        return {mod(index.x, val), mod(index.y, val), mod(index.z, val)};
    }

    ATTR_HOST_DEV_INL static int div(const int a, const int b) { return (a - mod(a, b)) / b; }

    ATTR_HOST_DEV_INL static math::Vec3i div(const math::Vec3i& index, const int val)
    {
        return {div(index.x, val), div(index.y, val), div(index.z, val)};
    }

    ATTR_HOST_DEV_INL static math::Vec3i getId(const math::Vec3f& v, const float voxelRes)
    {
        const int x = static_cast<int>(std::floor(v.x / voxelRes)) >> blockShift;
        const int y = static_cast<int>(std::floor(v.y / voxelRes)) >> blockShift;
        const int z = static_cast<int>(std::floor(v.z / voxelRes)) >> blockShift;
        return {x, y, z};
    }

    ATTR_HOST_DEV_INL static math::Vec3i getVoxelAbsolutePos(
        const math::Vec3i& blockId, const math::Vec3i& voxelId)
    {
        return int(blockSize) * blockId + voxelId;
    }

    ATTR_HOST_DEV_INL static math::Vec3f getVoxelPos(const math::Vec3i& id, const float voxelRes)
    {
        return voxelRes
               * math::Vec3f(
                   static_cast<float>(id.x), static_cast<float>(id.y), static_cast<float>(id.z));
    }

    ATTR_HOST_DEV_INL static math::Vec3i getVoxelId(const math::Vec3f& p, const float voxelRes)
    {
        const int x = mod(static_cast<int>(std::floor(p.x / voxelRes)), blockSize);
        const int y = mod(static_cast<int>(std::floor(p.y / voxelRes)), blockSize);
        const int z = mod(static_cast<int>(std::floor(p.z / voxelRes)), blockSize);
        return {x, y, z};
    }

    ATTR_HOST_DEV_INL static uint64_t encode(const BlockId& blockId)
    {
        static constexpr uint32_t mask = 0x1ffff;
        return uint64_t(static_cast<uint32_t>(blockId.x) & mask)
               | uint64_t(static_cast<uint32_t>(blockId.y) & mask) << 21
               | uint64_t(static_cast<uint32_t>(blockId.z) & mask) << 42;
    }

    ATTR_HOST_DEV_INL static BlockId decode(const uint64_t indice)
    {
        static constexpr uint64_t mask = 0x1ffff;
        static constexpr uint32_t signBit = 0x10000;
        static constexpr uint32_t signMask = 0xfffe0000;
        const uint32_t x = static_cast<uint32_t>(indice & mask);
        const uint32_t y = static_cast<uint32_t>((indice >> 21) & mask);
        const uint32_t z = static_cast<uint32_t>((indice >> 42) & mask);
        return BlockId{
            static_cast<int32_t>((x & signBit) ? (x | signMask) : x),
            static_cast<int32_t>((y & signBit) ? (y | signMask) : y),
            static_cast<int32_t>((z & signBit) ? (z | signMask) : z)};
    }
} // namespace utils
} // namespace fusion