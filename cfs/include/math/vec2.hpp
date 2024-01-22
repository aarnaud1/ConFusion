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

#include "attributes.hpp"
#include "common.hpp"

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
    struct Vec2
    {
        static_assert(std::is_arithmetic_v<T>, "Only arithmetic types can be used for Vec2");

        using value_type = T;

        T x;
        T y;

        // Constructors
        ATTR_HOST_DEV_INL constexpr Vec2() : x{0}, y{0} {}
        ATTR_HOST_DEV_INL constexpr Vec2(const T x, const T y) noexcept : x{x}, y{y} {}
        ATTR_HOST_DEV_INL constexpr Vec2(const T val) noexcept : x{val}, y{val} {}

        Vec2(const Vec2 &cp) noexcept = default;
        Vec2(Vec2 &&cp) noexcept = default;
        Vec2 &operator=(const Vec2 &cp) noexcept = default;
        Vec2 &operator=(Vec2 &&cp) noexcept = default;

        template <typename U>
        ATTR_HOST_DEV_INL constexpr operator Vec2<U>() const noexcept
        {
            return Vec2<U>(U{x}, U{y});
        }

        ATTR_HOST_DEV_INL bool operator==(const Vec2 &v0) const { return x == v0.x && y == v0.y; }

        ATTR_HOST_DEV_INL bool operator!=(const Vec2 &v0) const { return x != v0.x || y != v0.y; }

        ATTR_HOST_DEV_INL bool operator<(const Vec2 &v0) const
        {
            return (x < v0.x) || ((x == v0.x) && (y < v0.y));
        }

        ATTR_HOST_DEV_INL bool operator<=(const Vec2 &v0) const
        {
            return (x < v0.x) || ((x == v0.x) && (y <= v0.y));
        }

        ATTR_HOST_DEV_INL bool operator>(const Vec2 &v0) const { return v0 >= *this; }

        ATTR_HOST_DEV_INL bool operator>=(const Vec2 &v0) const { return v0 > *this; }

        ATTR_HOST_DEV_INL Vec2 &operator+=(const Vec2 &v0)
        {
            *this = *this + v0;
            return *this;
        }

        ATTR_HOST_DEV_INL Vec2 operator-() const
        {
            Vec2 ret;
            ret.x = -this->x;
            ret.y = -this->y;
            return ret;
        }

        ATTR_HOST_DEV_INL Vec2 &operator-=(const Vec2 &v0)
        {
            *this = *this - v0;
            return *this;
        }

        ATTR_HOST_DEV_INL Vec2 &operator*=(const T k)
        {
            *this = *this * k;
            return *this;
        }

        ATTR_HOST_DEV_INL Vec2 operator/=(const T k)
        {
            *this = *this / k;
            return *this;
        }

        ATTR_HOST_DEV_INL T Len() const { return std::sqrt(x * x + y * y); }

        ATTR_HOST_DEV_INL T LenSq() const { return x * x + y * y; }

        ATTR_HOST_DEV_INL T Dist(const Vec2 &v0) const { return Len(*this - v0); }

        ATTR_HOST_DEV_INL T Dot(const Vec2 &v0) const { return x * v0.x + y * v0.y; }

        ATTR_HOST_DEV_INL void Normalize()
        {
            const auto len = this->Len();
            *this /= len;
        }

        static ATTR_HOST_DEV_INL constexpr Vec2 Zero() { return {0, 0}; }

        static ATTR_HOST_DEV_INL T Len(const Vec2 &v0) { return v0.Len(); }

        static ATTR_HOST_DEV_INL T LenSq(const Vec2 &v0) { return v0.LenSq(); }

        static ATTR_HOST_DEV_INL T Dist(const Vec2 &v0, const Vec2 &v1) { return v0.Dist(v1); }

        static ATTR_HOST_DEV_INL T Dot(const Vec2 &v0, const Vec2 &v1) { return v0.Dot(v1); }

        static ATTR_HOST_DEV_INL Vec2 Normalize(const Vec2 &v0)
        {
            Vec2 ret{v0};
            ret.Normalize();
            return ret;
        }
    };

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<T> operator+(const Vec2<T> &v0, const Vec2<T> &v1)
    {
        Vec2<T> ret;
        ret.x = v0.x + v1.x;
        ret.y = v0.y + v1.y;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<T> operator-(const Vec2<T> &v0, const Vec2<T> &v1)
    {
        Vec2<T> ret;
        ret.x = v0.x - v1.x;
        ret.y = v0.y - v1.y;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<T> operator*(const Vec2<T> &v0, const T k)
    {
        Vec2<T> ret;
        ret.x = k * v0.x;
        ret.y = k * v0.y;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<T> operator*(const T k, const Vec2<T> &v0)
    {
        Vec2<T> ret;
        ret.x = k * v0.x;
        ret.y = k * v0.y;
        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<T> operator/(const Vec2<T> &v0, const T k)
    {
        Vec2<T> ret;
        ret.x = v0.x / k;
        ret.y = v0.y / k;
        return ret;
    }

    // -----------------------------------------------------------------------------

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<bool> lessThan(const Vec2<T> &v0, const Vec2<T> &v1)
    {
        return Vec2<bool>{v0.x < v1.x, v0.y < v1.y};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<bool> lessThanEqual(const Vec2<T> &v0, const Vec2<T> &v1)
    {
        return Vec2<bool>{v0.x <= v1.x, v0.y <= v1.y};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<bool> greaterThan(const Vec2<T> &v0, const Vec2<T> &v1)
    {
        return Vec2<bool>{v0.x > v1.x, v0.y > v1.y};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<bool> greaterThanEqual(const Vec2<T> &v0, const Vec2<T> &v1)
    {
        return Vec2<bool>{v0.x >= v1.x, v0.y >= v1.y};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<bool> equal(const Vec2<T> &v0, const Vec2<T> &v1)
    {
        return Vec2<bool>{v0.x == v1.x, v0.y == v1.y};
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec2<bool> notEqual(const Vec2<T> &v0, const Vec2<T> &v1)
    {
        return Vec2<bool>{v0.x != v1.x, v0.y != v1.y};
    }

    static ATTR_HOST_DEV_INL bool all(const Vec2<bool> &v) { return v.x && v.y; }

    static ATTR_HOST_DEV_INL bool any(const Vec2<bool> &v) { return v.x || v.y; }

    // -----------------------------------------------------------------------------

    template <typename T>
    static inline std::ostream &operator<<(std::ostream &os, const Vec2<T> &v)
    {
        os << "|" << v.x << " " << v.y << "|";
        return os;
    }

    template <typename T>
    static inline std::istream &operator>>(std::istream &is, Vec2<T> &v)
    {
        is >> v.x >> v.y;
        return is;
    }

    using Vec2b = Vec2<uint8_t>;
    using Vec2u = Vec2<uint32_t>;
    using Vec2i = Vec2<int32_t>;
    using Vec2f = Vec2<float>;
    using Vec2d = Vec2<double>;
} // namespace math
} // namespace cfs