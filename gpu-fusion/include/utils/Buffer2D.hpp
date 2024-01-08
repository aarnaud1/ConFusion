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

#include <utils/Img.hpp>
#include <utils/Ptr.hpp>

namespace fusion
{
// Wrapper class around fusion::BaseImg that allow to use the same data container for images with
// different sizes.
// A Buffer2D instance can be allocated with a max width and height, then resized without any
// allocation as long as the contained image dimensions are less than the max dimensions.
template <typename T>
class Buffer2D
{
  public:
    using value_type = T;

    constexpr Buffer2D() noexcept = default;
    inline Buffer2D(const size_t width, const size_t height)
        : width_{width}
        , height_{height}
        , maxWidth_{width}
        , maxHeight_{height}
        , img_{width_, height_}
    {}
    inline Buffer2D(const Buffer2D&) = default;
    inline Buffer2D(Buffer2D&&) = default;
    inline Buffer2D& operator=(const Buffer2D&) = default;
    inline Buffer2D& operator=(Buffer2D&&) = default;

    inline void resize(const size_t width, const size_t height)
    {
        width_ = width;
        height_ = height;
        if(width_ > maxWidth_ || height_ > maxHeight_)
        {
            maxWidth_ = width_;
            maxHeight_ = height_;
            img_.resize(width_, height_);
        }
    }

    inline void realloc(const size_t width, const size_t height)
    {
        width_ = width;
        height_ = height;
        maxWidth_ = width_;
        maxHeight_ = height_;
        img_.resize(width_, height_);
    }

    template <bool pageLocked>
    inline void upload(const CpuPtr<T, pageLocked>& data, const cudaStream_t& stream)
    {
        if(data.size() < width_ * height_)
        {
            throw std::runtime_error("Buffer not large enough");
        }
        img_.uploadFrom(data, stream);
    }

    template <bool pageLocked>
    inline void download(CpuPtr<T, pageLocked>& data, const cudaStream_t& stream)
    {
        if(data.size() < width_ * height_)
        {
            throw std::runtime_error("Buffer not large enough");
        }
        img_.downloadTo(data, stream);
    }

    inline auto& img() { return img_; }
    inline const auto& img() const { return img_; }

    inline size_t width() const { return width_; }
    inline size_t height() const { return height_; }
    inline size_t maxWidth() const { return maxWidth_; }
    inline size_t maxHeight() const { return maxHeight_; }

  private:
    size_t width_;
    size_t height_;
    size_t maxWidth_;
    size_t maxHeight_;

    GpuImg<value_type> img_{};
};
} // namespace fusion