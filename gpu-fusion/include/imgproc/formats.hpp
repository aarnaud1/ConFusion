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

#include "math/geometry.hpp"

#include <cstdint>

namespace fusion
{
enum class ImageFormat : uint32_t
{
    R,
    RGB,
    RGBA
};

template <ImageFormat format, typename T>
struct ImageInfo
{
    static_assert("Unknown image format");
};

template <typename T>
struct ImageInfo<ImageFormat::R, T>
{
    static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic for images");
    using value_type = T;
    static constexpr size_t channelCount = 1;
};
template <typename T>
struct ImageInfo<ImageFormat::RGB, T>
{
    static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic for images");
    using value_type = math::Vec3<T>;
    static constexpr size_t channelCount = 3;
};
template <typename T>
struct ImageInfo<ImageFormat::RGBA, T>
{
    static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic for images");
    using value_type = math::Vec4<T>;
    static constexpr size_t channelCount = 4;
};
} // namespace fusion