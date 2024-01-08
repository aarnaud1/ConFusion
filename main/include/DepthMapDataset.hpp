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

#include <math/geometry.hpp>
#include <memory>
#include <vector>

namespace tests
{
struct FrameInfo
{
    FrameInfo(const size_t w, const size_t h) : width{w}, height{h} {}
    size_t width;
    size_t height;
};

class DepthMapDataset
{
  public:
    DepthMapDataset(const std::string& dir, const size_t n = 0);

    auto size() const { return size_; }

    auto& intrinsics() { return intrinsics_; }
    const auto& intrinsics() const { return intrinsics_; }

    auto& poses() { return poses_; }
    const auto& poses() const { return poses_; }

    auto& depthImages() { return depthImages_; }
    const auto& depthImages() const { return depthImages_; }

    auto& colorImages() { return colorImages_; }
    const auto& colorImages() const { return colorImages_; }

    auto& getDepthInfo() { return depthFramesInfo_; }
    const auto& getDepthInfo() const { return depthFramesInfo_; }

    auto& getColorInfo() { return colorFramesInfo_; }
    const auto& getColorInfo() const { return colorFramesInfo_; }

  private:
    size_t size_;
    fusion::math::Mat4f intrinsics_;
    std::vector<fusion::math::Mat4f> poses_;
    std::vector<std::unique_ptr<uint16_t>> depthImages_;
    std::vector<std::unique_ptr<uint8_t>> colorImages_;
    std::vector<FrameInfo> depthFramesInfo_;
    std::vector<FrameInfo> colorFramesInfo_;

    void readIntrinsics(const std::string& filename);
    void readDepth(const std::string& filename);
    void readColor(const std::string& filename);
    void readPose(const std::string& filename);
};
} // namespace tests