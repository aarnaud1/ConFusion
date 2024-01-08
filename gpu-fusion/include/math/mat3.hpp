/*
 * Copyright (C) 2020  Adrien ARNAUD
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
#include "math/vec3.hpp"
#include "math/vec4.hpp"

#include <iostream>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <type_traits>

namespace fusion
{
namespace math
{
    template <typename T>
    struct Mat3
    {
        static_assert(std::is_arithmetic_v<T>, "Only arithmetic types can be used for Mat3");

        using DataType = T;

        T c00, c01, c02;
        T c10, c11, c12;
        T c20, c21, c22;

        ATTR_HOST_DEV_INL constexpr Mat3() noexcept
        {
            c00 = T(0);
            c01 = T(0);
            c02 = T(0);

            c10 = T(0);
            c11 = T(0);
            c12 = T(0);

            c20 = T(0);
            c21 = T(0);
            c22 = T(0);
        }

        template <typename U>
        ATTR_HOST_DEV_INL constexpr Mat3(U coeffs[9]) noexcept
        {
            c00 = T(coeffs[0]);
            c01 = T(coeffs[1]);
            c02 = T(coeffs[2]);

            c10 = T(coeffs[3]);
            c11 = T(coeffs[4]);
            c12 = T(coeffs[5]);

            c20 = T(coeffs[6]);
            c21 = T(coeffs[7]);
            c22 = T(coeffs[8]);
        }

        ATTR_HOST_DEV_INL Mat3(const Vec3<T> &v0, const Vec3<T> &v1, const Vec3<T> &v2) noexcept
        {
            this->c00 = v0.x;
            this->c01 = v0.y;
            this->c02 = v0.z;

            this->c10 = v1.x;
            this->c11 = v1.y;
            this->c12 = v1.z;

            this->c20 = v2.x;
            this->c21 = v2.y;
            this->c22 = v2.z;
        }

        Mat3(const Mat3 &cp) noexcept = default;
        Mat3(Mat3 &&cp) noexcept = default;
        Mat3 &operator=(const Mat3 &cp) noexcept = default;
        Mat3 &operator=(Mat3 &&cp) noexcept = default;

        ATTR_HOST_DEV_INL Mat3 &operator+=(const Mat3 &m)
        {
            this->c00 += m.c00;
            this->c01 += m.c01;
            this->c02 += m.c02;

            this->c10 += m.c10;
            this->c11 += m.c11;
            this->c12 += m.c12;

            this->c20 += m.c20;
            this->c21 += m.c21;
            this->c22 += m.c22;
            return *this;
        }

        ATTR_HOST_DEV_INL Mat3 &operator-=(const Mat3 &m)
        {
            this->c00 -= m.c00;
            this->c01 -= m.c01;
            this->c02 -= m.c02;

            this->c10 -= m.c10;
            this->c11 -= m.c11;
            this->c12 -= m.c12;

            this->c20 -= m.c20;
            this->c21 -= m.c21;
            this->c22 -= m.c22;
            return *this;
        }

        ATTR_HOST_DEV_INL Mat3 &operator*=(const Mat3 &m)
        {
            this->c00 = m.c00 * this->c00 + m.c01 * this->c10 + m.c02 * this->c20;
            this->c01 = m.c00 * this->c01 + m.c01 * this->c11 + m.c02 * this->c21;
            this->c02 = m.c00 * this->c02 + m.c01 * this->c12 + m.c02 * this->c22;

            this->c10 = m.c10 * this->c00 + m.c11 * this->c10 + m.c12 * this->c20;
            this->c11 = m.c10 * this->c01 + m.c11 * this->c11 + m.c12 * this->c21;
            this->c12 = m.c10 * this->c02 + m.c11 * this->c12 + m.c12 * this->c22;

            this->c20 = m.c20 * this->c00 + m.c21 * this->c10 + m.c22 * this->c20;
            this->c21 = m.c20 * this->c01 + m.c21 * this->c11 + m.c22 * this->c21;
            this->c22 = m.c20 * this->c02 + m.c21 * this->c12 + m.c22 * this->c22;

            return *this;
        }

        ATTR_HOST_DEV_INL Mat3 &operator*=(const T k)
        {
            this->c00 *= k;
            this->c01 *= k;
            this->c02 *= k;

            this->c10 *= k;
            this->c11 *= k;
            this->c12 *= k;

            this->c20 *= k;
            this->c21 *= k;
            this->c22 *= k;

            return *this;
        }

        ATTR_HOST_DEV_INL Mat3 &operator/=(const T k)
        {
            this->c00 /= k;
            this->c01 /= k;
            this->c02 /= k;

            this->c10 /= k;
            this->c11 /= k;
            this->c12 /= k;

            this->c20 /= k;
            this->c21 /= k;
            this->c22 /= k;

            return *this;
        }

        ATTR_HOST_DEV_INL Mat3 &operator-()
        {
            this->c00 = -this->c00;
            this->c01 = -this->c01;
            this->c02 = -this->c02;

            this->c10 = -this->c10;
            this->c11 = -this->c11;
            this->c12 = -this->c12;

            this->c20 = -this->c20;
            this->c21 = -this->c21;
            this->c22 = -this->c22;
            return *this;
        }

        ATTR_HOST_DEV_INL Mat3 operator/(const T k) const
        {
            Mat3 ret;

            ret.c00 = this->c00 / k;
            ret.c01 = this->c01 / k;
            ret.c02 = this->c02 / k;

            ret.c10 = this->c10 / k;
            ret.c11 = this->c11 / k;
            ret.c12 = this->c12 / k;

            ret.c20 = this->c20 / k;
            ret.c21 = this->c21 / k;
            ret.c22 = this->c22 / k;

            return ret;
        }

        ATTR_HOST_DEV_INL void Transpose()
        {
            const T swap10 = this->c10;
            this->c10 = this->c01;
            this->c01 = swap10;

            const T swap20 = this->c20;
            this->c20 = this->c02;
            this->c02 = swap20;

            const T swap21 = this->c21;
            this->c21 = this->c12;
            this->c12 = swap21;
        }

        ATTR_HOST_DEV_INL T Det()
        {
            return c00 * (c11 * c22 - c12 * c21) - c01 * (c10 * c22 - c12 * c20)
                   + c02 * (c10 * c21 - c11 * c20);
        }

        ATTR_HOST_DEV_INL void Inverse()
        {
            const T det = c00 * (c11 * c22 - c12 * c21) - c01 * (c10 * c22 - c12 * c20)
                          + c02 * (c10 * c21 - c11 * c20);

            const T tmpc00 = +(c11 * c22 - c21 * c12);
            const T tmpc01 = -(c10 * c22 - c20 * c12);
            const T tmpc02 = +(c10 * c21 - c20 * c11);
            const T tmpc10 = -(c01 * c22 - c21 * c02);
            const T tmpc11 = +(c00 * c22 - c20 * c02);
            const T tmpc12 = -(c00 * c21 - c20 * c01);
            const T tmpc20 = +(c01 * c12 - c11 * c02);
            const T tmpc21 = -(c00 * c12 - c10 * c02);
            const T tmpc22 = +(c00 * c11 - c10 * c01);

            c00 = tmpc00 / det;
            c10 = tmpc01 / det;
            c20 = tmpc02 / det;
            c01 = tmpc10 / det;
            c11 = tmpc11 / det;
            c21 = tmpc12 / det;
            c02 = tmpc20 / det;
            c12 = tmpc21 / det;
            c22 = tmpc22 / det;
        }

        static ATTR_HOST_DEV_INL Mat3 Identity()
        {
            Mat3 ret;
            ret.c00 = T(1);
            ret.c01 = T(0);
            ret.c02 = T(0);

            ret.c10 = T(0);
            ret.c11 = T(1);
            ret.c12 = T(0);

            ret.c20 = T(0);
            ret.c21 = T(0);
            ret.c22 = T(1);

            return ret;
        }

        static ATTR_HOST_DEV_INL Mat3 Transpose(const Mat3 &m)
        {
            Mat3 ret(m);
            ret.Transpose();
            return ret;
        }

        static ATTR_HOST_DEV_INL T Det(const Mat3 &m) { return m.Det(); }

        static ATTR_HOST_DEV_INL Mat3 Inverse(const Mat3 &m)
        {
            Mat3 ret(m);
            ret.Inverse();
            return ret;
        }
    };

    // -----------------------------------------------------------------------------

    template <typename T>
    static ATTR_HOST_DEV_INL Mat3<T> operator+(const Mat3<T> &m0, const Mat3<T> &m1)
    {
        Mat3<T> ret;

        ret.c00 = m0.c00 + m1.c00;
        ret.c01 = m0.c01 + m1.c01;
        ret.c02 = m0.c02 + m1.c02;

        ret.c10 = m0.c10 + m1.c10;
        ret.c11 = m0.c11 + m1.c11;
        ret.c12 = m0.c12 + m1.c12;

        ret.c20 = m0.c20 + m1.c20;
        ret.c21 = m0.c21 + m1.c21;
        ret.c22 = m0.c22 + m1.c22;

        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Mat3<T> operator-(const Mat3<T> &m0, const Mat3<T> &m1)
    {
        Mat3<T> ret;

        ret.c00 = m0.c00 - m1.c00;
        ret.c01 = m0.c01 - m1.c01;
        ret.c02 = m0.c02 - m1.c02;

        ret.c10 = m0.c10 - m1.c10;
        ret.c11 = m0.c11 - m1.c11;
        ret.c12 = m0.c12 - m1.c12;

        ret.c20 = m0.c20 - m1.c20;
        ret.c21 = m0.c21 - m1.c21;
        ret.c22 = m0.c22 - m1.c22;

        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<T> operator*(const Mat3<T> &m, const Vec3<T> &v)
    {
        Vec3<T> ret;

        ret.x = m.c00 * v.x + m.c01 * v.y + m.c02 * v.z;
        ret.y = m.c10 * v.x + m.c11 * v.y + m.c12 * v.z;
        ret.z = m.c20 * v.x + m.c21 * v.y + m.c22 * v.z;

        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Mat3<T> operator*(const T k, const Mat3<T> &m)
    {
        Mat3<T> ret;

        ret.c00 = m.c00 * k;
        ret.c01 = m.c01 * k;
        ret.c02 = m.c02 * k;

        ret.c10 = m.c10 * k;
        ret.c11 = m.c11 * k;
        ret.c12 = m.c12 * k;

        ret.c20 = m.c20 * k;
        ret.c21 = m.c21 * k;
        ret.c22 = m.c22 * k;

        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Mat3<T> operator*(const Mat3<T> &m, const T k)
    {
        return k * m;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Mat3<T> operator*(const Mat3<T> &m0, const Mat3<T> &m1)
    {
        Mat3<T> ret;

        ret.c00 = m0.c00 * m1.c00 + m0.c01 * m1.c10 + m0.c02 * m1.c20;
        ret.c01 = m0.c00 * m1.c01 + m0.c01 * m1.c11 + m0.c02 * m1.c21;
        ret.c02 = m0.c00 * m1.c02 + m0.c01 * m1.c12 + m0.c02 * m1.c22;

        ret.c10 = m0.c10 * m1.c00 + m0.c11 * m1.c10 + m0.c12 * m1.c20;
        ret.c11 = m0.c10 * m1.c01 + m0.c11 * m1.c11 + m0.c12 * m1.c21;
        ret.c12 = m0.c10 * m1.c02 + m0.c11 * m1.c12 + m0.c12 * m1.c22;

        ret.c20 = m0.c20 * m1.c00 + m0.c21 * m1.c10 + m0.c22 * m1.c20;
        ret.c21 = m0.c20 * m1.c01 + m0.c21 * m1.c11 + m0.c22 * m1.c21;
        ret.c22 = m0.c20 * m1.c02 + m0.c21 * m1.c12 + m0.c22 * m1.c22;

        return ret;
    }

    template <typename T>
    static ATTR_HOST_DEV_INL Vec3<T> operator*=(const Vec3<T> &v, const Mat3<T> &m)
    {
        return Vec3<T>(m * v);
    }

    template <typename T>
    static inline std::ostream &operator<<(std::ostream &os, const Mat3<T> &m)
    {
        os << "|" << m.c00 << " " << m.c01 << " " << m.c02 << "|\n";
        os << "|" << m.c10 << " " << m.c11 << " " << m.c12 << "|\n";
        os << "|" << m.c20 << " " << m.c21 << " " << m.c22 << "|";
        return os;
    }

    template <typename T>
    static inline std::istream &operator>>(std::istream &is, Mat3<T> &m)
    {
        is >> m.c00 >> m.c01 >> m.c02 >> m.c10 >> m.c11 >> m.c12 >> m.c20 >> m.c21 >> m.c22;
        return is;
    }

    using Mat3u = Mat3<uint32_t>;
    using Mat3i = Mat3<int32_t>;
    using Mat3f = Mat3<float>;
    using Mat3d = Mat3<double>;
} // namespace math
} // namespace fusion
