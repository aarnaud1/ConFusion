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

#include "imgproc/formats.hpp"
#include "utils/Buffer2D.hpp"
#include "utils/Img.hpp"
#include "utils/Ptr.hpp"

namespace cfs
{
// Same as Buffer2D but the data type is constrained see formats.hpp.
template <ImageFormat format, typename T>
class Image2D final : public Buffer2D<typename ImageInfo<format, T>::value_type>
{
  public:
    using value_type = typename ImageInfo<format, T>::value_type;

    constexpr Image2D() noexcept = default;
    inline Image2D(const size_t width, const size_t height) : Buffer2D<T>{width, height} {}
    inline Image2D(const Image2D&) = default;
    inline Image2D(Image2D&&) = default;
    inline Image2D& operator=(const Image2D&) = default;
    inline Image2D& operator=(Image2D&&) = default;

    constexpr size_t channels() { return ImageInfo<format, T>::channelCount; }
};
} // namespace cfs
