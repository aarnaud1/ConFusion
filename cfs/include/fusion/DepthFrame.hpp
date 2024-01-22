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
#include "math/geometry.hpp"
#include "utils/DepthUtils.hpp"
#include "utils/Img.hpp"
#include "utils/Ptr.hpp"

namespace cfs
{
struct GpuFrameData
{
    size_t width;
    size_t height;

    uint16_t* depth;
    math::Vec3f* points;
    math::Vec3f* colors;
    math::Vec3f* normals;
};

class GpuFrame
{
  public:
    GpuFrame() = default;
    GpuFrame(const size_t width, const size_t height)
        : width_{width}
        , height_{height}
        , depthData_{width_ * height_}
        , rgbData_{3 * width_ * height_}
        , depth_{width_, height_}
        , rgb_{width_, height_}
        , points_{width_ * height_}
        , colors_{width_ * height_}
        , normals_{width_ * height_}
    {}
    GpuFrame(const GpuFrame&) = default;
    GpuFrame(GpuFrame&&) = default;
    GpuFrame& operator=(const GpuFrame&) = default;
    GpuFrame& operator=(GpuFrame&&) = default;

    void resize(const size_t width, const size_t height)
    {
        width_ = width;
        height_ = height;
        depthData_.resize(width_ * height_);
        rgbData_.resize(3 * width_ * height_);
        depth_.resize(width_, height_);
        rgb_.resize(width_, height_);
        points_.resize(width_ * height_);
        colors_.resize(width_ * height_);
        normals_.resize(width_ * height_);
    }

    void prepare(const math::Mat3f& k, const float depthScale, const cudaStream_t& stream)
    {
        utils::initRGBDFrame(depthData_, rgbData_, depth_, rgb_, width_, height_, stream);
        utils::extractPoints(depth_, points_, k, depthScale, stream);
        utils::extractColors(rgb_, colors_, true, stream);
    }

    GpuFrameData getData() { return {width_, height_, depthData_, points_, colors_, normals_}; }

    auto width() const { return width_; }
    auto height() const { return height_; }

    auto& depthData() { return depthData_; }
    const auto& depthData() const { return depthData_; }

    auto& rgbData() { return rgbData_; }
    const auto& rgbData() const { return rgbData_; }

    auto& depth() { return depth_; }
    const auto& depth() const { return depth_; }

    auto& rgb() { return rgb_; }
    const auto& rgb() const { return rgb_; }

    auto& points() { return points_; }
    const auto& points() const { return points_; }

    auto& colors() { return colors_; }
    const auto& colors() const { return colors_; }

    auto& normals() { return normals_; }
    const auto& normals() const { return normals_; }

    static size_t getMemorySize(const size_t w, const size_t h)
    {
        // TODO : round values because of mem align in images
        const size_t res = w * h;
        return res
               * (sizeof(decltype(depthData_)::value_type) + sizeof(decltype(rgbData_)::value_type)
                  + sizeof(decltype(depth_)::value_type) + sizeof(decltype(rgb_)::value_type)
                  + sizeof(decltype(points_)::value_type) + sizeof(decltype(colors_)::value_type)
                  + sizeof(decltype(normals_)::value_type));
    }

  private:
    size_t width_{0};
    size_t height_{0};

    GpuPtr<uint16_t> depthData_;
    GpuPtr<uint8_t> rgbData_;
    GpuImg<uint16_t> depth_;
    GpuImg<math::Vec3b> rgb_;
    GpuPtr<math::Vec3f> points_;
    GpuPtr<math::Vec3f> colors_;
    GpuPtr<math::Vec3f> normals_;
};

template <bool pageLocked = false>
class CpuFrame final
{
  public:
    CpuFrame() = default;
    CpuFrame(const size_t width, const size_t height)
        : width_{width}, height_{height}, depth_{width_ * height_}, rgb_{3 * width_ * height_}
    {}
    CpuFrame(const CpuFrame&) = default;
    CpuFrame(CpuFrame&&) = default;
    CpuFrame& operator=(const CpuFrame&) = default;
    CpuFrame& operator=(CpuFrame&&) = default;

    void resize(const size_t width, const size_t height)
    {
        width_ = width;
        height_ = height;
        depth_.resize(width_ * height_);
        rgb_.resize(3 * width_ * height_);
    }

    template <bool dstPageLocked>
    void copyTo(CpuFrame<dstPageLocked>& dst, const cudaStream_t& stream) const
    {
        depth_.copyTo(dst.depth(), stream);
        rgb_.copyTo(dst.rgb(), stream);
    }

    void uploadTo(GpuFrame& dst, const cudaStream_t& stream) const
    {
        depth_.uploadTo(dst.depthData(), stream);
        rgb_.uploadTo(dst.rgbData(), stream);
    }

    auto width() const { return width_; }
    auto height() const { return height_; }

    auto& depth() { return depth_; }
    const auto& depth() const { return depth_; }

    auto& rgb() { return rgb_; }
    const auto& rgb() const { return rgb_; }

  private:
    size_t width_{0};
    size_t height_{0};

    CpuPtr<uint16_t, pageLocked> depth_;
    CpuPtr<uint8_t, pageLocked> rgb_;
};

} // namespace cfs