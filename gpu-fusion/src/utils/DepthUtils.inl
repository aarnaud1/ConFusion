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

#include "MathUtils.inl"

namespace fusion
{
namespace utils
{
    static constexpr size_t blockSizeX = 8;
    static constexpr size_t blockSizeY = 128;
    static constexpr size_t blockSize = blockSizeX * blockSizeY;

    ATTR_HOST_DEV_INL static math::Vec3f getPoint(
        const uint16_t depth,
        const size_t u,
        const size_t v,
        const float scale,
        const float cx,
        const float cy,
        const float fx,
        const float fy)
    {
        const double z = double(depth) / double(scale);
        const double x = (u - cx) * z / double(fx);
        const double y = (v - cy) * z / double(fy);
        return math::Vec3f{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
    }

    ATTR_HOST_DEV_INL static math::Vec3f getPoint(
        const float depth,
        const float u,
        const float v,
        const float scale,
        const float cx,
        const float cy,
        const float fx,
        const float fy)
    {
        const double z = double(depth) / float(scale);
        const double x = (u - cx) * z / double(fx);
        const double y = (v - cy) * z / double(fy);
        return math::Vec3f{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
    }

    ATTR_HOST_DEV_INL static math::Vec3f getRay(
        const float u,
        const float v,
        const float cx,
        const float cy,
        const float fx,
        const float fy)
    {
        const double z = 1.0f;
        const double x = (u - cx) / double(fx);
        const double y = (v - cy) / double(fy);
        return math::Vec3f::Normalize(
            math::Vec3f{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
    }

    ATTR_HOST_DEV_INL static math::Vec3f getRay(
        const math::Vec2f uv, const float cx, const float cy, const float fx, const float fy)
    {
        const double z = 1.0f;
        const double x = (uv.x - cx) / double(fx);
        const double y = (uv.y - cy) / double(fy);
        return math::Vec3f{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
    }

    ATTR_HOST_DEV_INL static math::Vec2f getDepthPos(
        const math::Vec3f& p, const float cx, const float cy, const float fx, const float fy)
    {
        return math::Vec2f{fx * p.x / p.z + cx, fy * p.y / p.z + cy};
    }
} // namespace utils
} // namespace fusion