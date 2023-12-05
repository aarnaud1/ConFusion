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
        const double scale,
        const float cx,
        const float cy,
        const float fx,
        const float fy)
    {
        const double z = double(depth) / scale;
        const double x = (u - cx) * z / double(fx);
        const double y = (v - cy) * z / double(fy);
        return math::Vec3f{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
    }
} // namespace utils
} // namespace fusion