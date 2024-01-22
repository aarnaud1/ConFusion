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

#include "common.hpp"
#include "math/vec2.hpp"

#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <type_traits>

namespace cfs
{
namespace math
{
    template <typename T>
    struct Vec3
    {
        static_assert(std::is_arithmetic_v<T>, "Only arithmetic types can be used for Vec3");

        using value_type = T;

        T x;
        T y;
        T z;

        // Constructors
        ATTR_HOST_DEV_INL constexpr Vec3() noexcept : x(0), y(0), z(0) {}
        ATTR_HOST_DEV_INL constexpr Vec3(const T x, const T y, const T z) noexcept
            : x(x), y(y), z(z)
        {}
        ATTR_HOST_DEV_INL constexpr Vec3(const T val) noexcept : x{val}, y{val}, z{val} {}

        Vec3(const Vec3 &cp) noexcept = default;
        Vec3(Vec3 &&cp) noexcept = default;
        Vec3 &operator=(const Vec3 &cp) noexcept = default;
        Vec3 &operator=(Vec3 &&cp) noexcept = default;

        // Conversion
        template <typename U>
        ATTR_HOST_DEV_INL constexpr operator Vec3<U>() const noexcept
        {
            return Vec3<U>{U(x), U(y), U(z)};
        }

        ATTR_HOST_DEV_INL bool operator==(const Vec3 &v0) const
        {
            return (x == v0.x) && (y == v0.y) && (z == v0.z);
        }
        ATTR_HOST_DEV_INL bool operator!=(const Vec3 &v0) const
        {
            return (x != v0.x) || (y != v0.y) || (z != v0.z);
        }

        ATTR_HOST_DEV_INL bool operator<(const Vec3 &v0) const
        {
            return (x < v0.x) || ((x == v0.x) && ((y < v0.y) || ((y == v0.y) && (z < v0.z))));
        }

        ATTR_HOST_DEV_INL bool operator<=(const Vec3 &v0) const
        {
            return (x < v0.x) || ((x == v0.x) && ((y < v0.y) || ((y == v0.y) && (z <= v0.z))));
        }

        ATTR_HOST_DEV_INL bool operator>(const Vec3 &v0) const { return v0 >= *this; }

        ATTR_HOST_DEV_INL bool operator>=(const Vec3 &v0) const { return v0 > *this; }

        ATTR_HOST_DEV_INL Vec3 &operator+=(const Vec3 &v0)
        {
            *this = *this + v0;
            return *this;
        }

        ATTR_HOST_DEV_INL Vec3 operator-() const
        {
            Vec3 ret;
            ret.x = -this->x;
            ret.y = -this->y;
            ret.z = -this->z;
            return ret;
        }

        ATTR_HOST_DEV_INL Vec3 &operator-=(const Vec3 &v0)
        {
            *this = *this - v0;
            return *this;
        }

        ATTR_HOST_DEV_INL Vec3 &operator*=(const T k)
        {
            *this = *this * k;
            return *this;
        }

        ATTR_HOST_DEV_INL Vec3 operator/=(const T k)
        {
            *this = *this / k;
            return *this;
        }

        ATTR_HOST_DEV_INL T Len() const { return std::sqrt(x * x + y * y + z * z); }

        ATTR_HOST_DEV_INL T LenSq() const { return x * x + y * y + z * z; }

        ATTR_HOST_DEV_INL T Dist(const Vec3 &v0) const
        {
            const T dx = this->x - v0.x;
            const T dy = this->y - v0.y;
            const T dz = this->z - v0.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        ATTR_HOST_DEV_INL T Dot(const Vec3 &v0) const
        {
            return this->x * v0.x + this->y * v0.y + this->z * v0.z;
        }

        static ATTR_HOST_DEV_INL T Len(const Vec3 &v0)
        {
            return std::sqrt(v0.x * v0.x + v0.y * v0.y + v0.z * v0.z);
        }

        static ATTR_HOST_DEV_INL T LenSq(const Vec3 &v0) { return v0.LenSq(); }

        ATTR_HOST_DEV_INL void Normalize()
        {
            const T l = std::sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
            *this = *this * (T(1) / l);
        }

        static ATTR_HOST_DEV_INL constexpr Vec3 Zero() { return {0, 0, 0}; }

        static ATTR_HOST_DEV_INL T Dist(const Vec3 &v0, const Vec3 &v1)
        {
            const T dx = v0.x - v1.x;
            const T dy = v0.y - v1.y;
            const T dz = v0.z - v1.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        static ATTR_HOST_DEV_INL Vec3 Normalize(const Vec3 &v0)
        {
            return Vec3(v0) * (T(1) / std::sqrt(v0.x * v0.x + v0.y * v0.y + v0.z * v0.z));
        }

        static ATTR_HOST_DEV_INL T Dot(const Vec3 &v0, const Vec3 &v1)
        {
            return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
        }

        static ATTR_HOST_DEV_INL Vec3 Cross(const Vec3 &v0, const Vec3 &v1)
        {
            Vec3 ret;

            ret.x = v0.y * v1.z - v0.z * v1.y;
            ret.y = v0.z * v1.x - v0.x * v1.z;
            ret.z = v0.x * v1.y - v0.y * v1.x;

            return ret;
        }

        static ATTR_HOST_DEV_INL Vec3 Reflect(const Vec3 &I, const Vec3 &N)
        {
            Vec3 ret;
            Vec3 normalized = Vec3::Normalize(N);
            T tmp = 2.0f * Vec3::Dot(normalized, I);
            ret = Vec3(I) - normalized * tmp;
            return ret;
        }
    };

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<T> operator+(const Vec3<T> &v0, const Vec3<T> &v1)
    {
        Vec3<T> ret;
        ret.x = v0.x + v1.x;
        ret.y = v0.y + v1.y;
        ret.z = v0.z + v1.z;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<T> operator-(const Vec3<T> &v0, const Vec3<T> &v1)
    {
        Vec3<T> ret;
        ret.x = v0.x - v1.x;
        ret.y = v0.y - v1.y;
        ret.z = v0.z - v1.z;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<T> operator*(const Vec3<T> &v0, const T k)
    {
        Vec3<T> ret;
        ret.x = k * v0.x;
        ret.y = k * v0.y;
        ret.z = k * v0.z;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<T> operator*(const T k, const Vec3<T> &v0)
    {
        Vec3<T> ret;
        ret.x = k * v0.x;
        ret.y = k * v0.y;
        ret.z = k * v0.z;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<T> operator/(const Vec3<T> &v0, const T k)
    {
        Vec3<T> ret;
        ret.x = v0.x / k;
        ret.y = v0.y / k;
        ret.z = v0.z / k;
        return ret;
    }

    // -----------------------------------------------------------------------------

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<bool> lessThan(const Vec3<T> &v0, const Vec3<T> &v1)
    {
        return Vec3<bool>{v0.x < v1.x, v0.y < v1.y, v0.z < v1.z};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<bool> lessThanEqual(const Vec3<T> &v0, const Vec3<T> &v1)
    {
        return Vec3<bool>{v0.x <= v1.x, v0.y <= v1.y, v0.z <= v1.z};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<bool> greaterThan(const Vec3<T> &v0, const Vec3<T> &v1)
    {
        return Vec3<bool>{v0.x > v1.x, v0.y > v1.y, v0.z > v1.z};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<bool> greaterThanEqual(const Vec3<T> &v0, const Vec3<T> &v1)
    {
        return Vec3<bool>{v0.x >= v1.x, v0.y >= v1.y, v0.z >= v1.z};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<bool> equal(const Vec3<T> &v0, const Vec3<T> &v1)
    {
        return Vec3<bool>{v0.x == v1.x, v0.y == v1.y, v0.z == v1.z};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<bool> notEqual(const Vec3<T> &v0, const Vec3<T> &v1)
    {
        return Vec3<bool>{v0.x != v1.x, v0.y != v1.y, v0.z != v1.z};
    }

    static ATTR_HOST_DEV_INL bool all(const Vec3<bool> &v) { return v.x && v.y && v.z; }

    static ATTR_HOST_DEV_INL bool any(const Vec3<bool> &v) { return v.x || v.y || v.z; }

    // -----------------------------------------------------------------------------

    template <typename T>
    static inline std::ostream &operator<<(std::ostream &os, const Vec3<T> &v)
    {
        os << "|" << v.x << " " << v.y << " " << v.z << "|";
        return os;
    }

    template <typename T>
    static inline std::istream &operator>>(std::istream &is, Vec3<T> &v)
    {
        is >> v.x >> v.y >> v.z;
        return is;
    }

    using Vec3b = Vec3<uint8_t>;
    using Vec3u = Vec3<uint32_t>;
    using Vec3i = Vec3<int32_t>;
    using Vec3f = Vec3<float>;
    using Vec3d = Vec3<double>;
} // namespace math
} // namespace cfs
