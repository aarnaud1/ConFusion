#pragma once

#include "attributes.hpp"
#include "math/geometry.hpp"

namespace fusion
{
namespace utils
{
    ATTR_HOST_DEV_INL math::Vec3f getMaxPoint(const math::Vec3f& p0, const math::Vec3f& p1)
    {
        return math::Vec3f{std::max(p0.x, p1.x), std::max(p0.y, p1.y), std::max(p0.z, p1.z)};
    }

    ATTR_HOST_DEV_INL math::Vec3f getMinPoint(const math::Vec3f& p0, const math::Vec3f& p1)
    {
        return math::Vec3f{std::min(p0.x, p1.x), std::min(p0.y, p1.y), std::min(p0.z, p1.z)};
    }

    ATTR_DEV_INL float atomicMinFloat(float* addr, float value)
    {
        float old;
        old = (value >= 0)
                  ? __int_as_float(atomicMin((int*) addr, __float_as_int(value)))
                  : __uint_as_float(atomicMax((unsigned int*) addr, __float_as_uint(value)));
        return old;
    }

    ATTR_DEV_INL float atomicMaxFloat(float* addr, float value)
    {
        float old;
        old = (value >= 0)
                  ? __int_as_float(atomicMax((int*) addr, __float_as_int(value)))
                  : __uint_as_float(atomicMin((unsigned int*) addr, __float_as_uint(value)));
        return old;
    }
} // namespace utils
} // namespace fusion