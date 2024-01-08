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
#include "common.hpp"

#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <type_traits>

namespace fusion
{
namespace math
{
    template <typename T>
    struct Vec4
    {
        static_assert(std::is_arithmetic_v<T>, "Only arithmetic types can be used for Vec4");

        using value_type = T;

        T x;
        T y;
        T z;
        T w;

        // Constructors
        ATTR_HOST_DEV_INL constexpr Vec4() noexcept : x(0), y(0), z(0), w(0) {}
        ATTR_HOST_DEV_INL constexpr Vec4(const T x, const T y, const T z, const T w) noexcept
            : x{x}, y{y}, z{z}, w{w}
        {}
        ATTR_HOST_DEV_INL constexpr Vec4(const T val) noexcept : x{val}, y{val}, z{val} {}
        ATTR_HOST_DEV_INL constexpr Vec4(Vec3<T> const &pos, const T w)
            : x(pos.x), y(pos.y), z(pos.z), w(w)
        {}
        ATTR_HOST_DEV_INL constexpr Vec4(Vec3<T> const &pos) : x(pos.x), y(pos.y), z(pos.z), w(1) {}

        Vec4(const Vec4 &) noexcept = default;
        Vec4(Vec4 &&cp) noexcept = default;
        Vec4 &operator=(const Vec4 &cp) noexcept = default;
        Vec4 &operator=(Vec4 &&cp) noexcept = default;

        // Conversion
        template <typename U>
        ATTR_HOST_DEV_INL constexpr operator Vec4<U>() const noexcept
        {
            return Vec4<U>{U(x), U(y), U(z), U(w)};
        }
        template <typename U>
        ATTR_HOST_DEV_INL constexpr operator Vec3<U>() const noexcept
        {
            return Vec3<U>{U(x), U(y), U(z)};
        }

        ATTR_HOST_DEV_INL bool operator==(Vec4 const &v0) const
        {
            return (x == v0.x) && (y == v0.y) && (z == v0.z) && (w == v0.w);
        }
        ATTR_HOST_DEV_INL bool operator!=(Vec4 const &v0) const
        {
            return (x != v0.x) || (y != v0.y) || (z != v0.z) || (w != v0.w);
        }

        // Swizzle
        ATTR_HOST_DEV_INL Vec3<T> xyz() const { return Vec3<T>(x, y, z); }

        ATTR_HOST_DEV_INL Vec4 &operator+=(Vec4 const &v0)
        {
            *this = *this + v0;
            return *this;
        }

        ATTR_HOST_DEV_INL Vec4 &operator-=(Vec4 const &v0)
        {
            *this = *this - v0;
            return *this;
        }

        // Allow matrix - vector multiplication in the form V *= M
        template <typename U>
        ATTR_HOST_DEV_INL Vec4 &operator*=(const U &m)
        {
            *this = m * *this;
            return *this;
        }

        ATTR_HOST_DEV_INL Vec4 &operator*=(const T k)
        {
            *this = *this * k;
            return *this;
        }

        ATTR_HOST_DEV_INL Vec4 operator/=(const T k)
        {
            *this = *this / k;
            return *this;
        }

        ATTR_HOST_DEV_INL T Len() const { return std::sqrt(x * x + y * y + z * z + w * w); }

        ATTR_HOST_DEV_INL T LenSq() const { return x * x + y * y + z * z + w * w; }

        ATTR_HOST_DEV_INL T Dist(Vec4 const &v0) const
        {
            const T dx = this->x - v0.x;
            const T dy = this->y - v0.y;
            const T dz = this->z - v0.z;
            const T dw = this->w - v0.w;
            return std::sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
        }

        ATTR_HOST_DEV_INL T Dot(const Vec4 &v0) const
        {
            return this->x * v0.x + this->y * v0.y + this->z * v0.z + this->w * v0.w;
        }

        static ATTR_HOST_DEV_INL constexpr Vec4 Zero() { return {0, 0, 0, 0}; }

        static ATTR_HOST_DEV_INL T Len(Vec4 const &v0)
        {
            return std::sqrt(v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w);
        }

        static ATTR_HOST_DEV_INL T LenSq(const Vec4 &v0) { return v0.LenSq(); }

        ATTR_HOST_DEV_INL void Normalize()
        {
            *this = *this * (T(1) / std::sqrt(x * x + y * y + z * z + w * w));
        }

        static ATTR_HOST_DEV_INL T Dist(Vec4 const &v0, Vec4 const &v1)
        {
            const T dx = v0.x - v1.x;
            const T dy = v0.y - v1.y;
            const T dz = v0.z - v1.z;
            const T dw = v0.w - v1.w;
            return std::sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
        }

        static ATTR_HOST_DEV_INL Vec4 Normalize(Vec4 const &v0)
        {
            return Vec4(v0)
                   * (T(1) / std::sqrt(v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w));
        }

        static ATTR_HOST_DEV_INL T Dot(Vec4 const &v0, Vec4 const &v1)
        {
            return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
        }
    };

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<T> operator+(const Vec4<T> &v0, const Vec4<T> &v1)
    {
        Vec4<T> ret;
        ret.x = v0.x + v1.x;
        ret.y = v0.y + v1.y;
        ret.z = v0.z + v1.z;
        ret.w = v0.w + v1.w;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<T> operator-(const Vec4<T> &v0, const Vec4<T> &v1)
    {
        Vec4<T> ret;
        ret.x = v0.x - v1.x;
        ret.y = v0.y - v1.y;
        ret.z = v0.z - v1.z;
        ret.w = v0.w - v1.w;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<T> operator*(const Vec4<T> &v0, const T k)
    {
        Vec4<T> ret;
        ret.x = k * v0.x;
        ret.y = k * v0.y;
        ret.z = k * v0.z;
        ret.w = k * v0.w;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<T> operator*(const T k, const Vec4<T> &v0)
    {
        Vec4<T> ret;
        ret.x = k * v0.x;
        ret.y = k * v0.y;
        ret.z = k * v0.z;
        ret.w = k * v0.w;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<T> operator/(const Vec4<T> &v0, const T k)
    {
        Vec4<T> ret;
        ret.x = v0.x / k;
        ret.y = v0.y / k;
        ret.z = v0.z / k;
        ret.w = v0.w / k;
        return ret;
    }

    // -----------------------------------------------------------------------------

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<bool> lessThan(const Vec4<T> &v0, const Vec4<T> &v1)
    {
        return Vec4<bool>{v0.x < v1.x, v0.y < v1.y, v0.z < v1.z, v0.w < v1.w};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<bool> lessThanEqual(const Vec4<T> &v0, const Vec4<T> &v1)
    {
        return Vec4<bool>{v0.x <= v1.x, v0.y <= v1.y, v0.z <= v1.z, v0.w <= v1.w};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<bool> greaterThan(const Vec4<T> &v0, const Vec4<T> &v1)
    {
        return Vec4<bool>{v0.x > v1.x, v0.y > v1.y, v0.z > v1.z, v0.w > v1.w};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<bool> greaterThanEqual(const Vec4<T> &v0, const Vec4<T> &v1)
    {
        return Vec4<bool>{v0.x >= v1.x, v0.y >= v1.y, v0.z >= v1.z, v0.w >= v1.w};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<bool> equal(const Vec4<T> &v0, const Vec4<T> &v1)
    {
        return Vec4<bool>{v0.x == v1.x, v0.y == v1.y, v0.z == v1.z, v0.w == v1.w};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec4<bool> notEqual(const Vec4<T> &v0, const Vec4<T> &v1)
    {
        return Vec4<bool>{v0.x != v1.x, v0.y != v1.y, v0.z != v1.z, v0.w != v1.w};
    }

    static ATTR_HOST_DEV_INL bool all(const Vec4<bool> &v) { return v.x && v.y && v.z && v.w; }

    static ATTR_HOST_DEV_INL bool any(const Vec4<bool> &v) { return v.x || v.y || v.z || v.w; }

    // -----------------------------------------------------------------------------

    template <typename T>
    static inline std::ostream &operator<<(std::ostream &os, const Vec4<T> &v)
    {
        os << "|" << v.x << " " << v.y << " " << v.z << " " << v.w << "|";
        return os;
    }

    template <typename T>
    static inline std::istream &operator>>(std::istream &is, Vec4<T> &v)
    {
        is >> v.x >> v.y >> v.z >> v.w;
        return is;
    }

    using Vec4b = Vec4<uint8_t>;
    using Vec4u = Vec4<uint32_t>;
    using Vec4i = Vec4<int32_t>;
    using Vec4f = Vec4<float>;
    using Vec4d = Vec4<double>;
} // namespace math
} // namespace fusion
