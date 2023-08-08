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

#include "common.hpp"
#include "utils/Ptr.hpp"
#include "utils/Img.hpp"
#include "geometry/geometry.hpp"
#include "fusion/OrderedPointCloud.hpp"
#include "fusion/Surface.hpp"

#include <cuda_runtime.h>

namespace fusion
{
struct AABB
{
  geometry::Vec3f min;
  geometry::Vec3f max;
};

class RGBDFrame
{
public:
  RGBDFrame() noexcept = default;
  RGBDFrame(const size_t width, const size_t height);
  RGBDFrame(const RGBDFrame& cp);
  RGBDFrame(RGBDFrame&& cp) = default;
  RGBDFrame& operator=(const RGBDFrame& cp);
  RGBDFrame& operator=(RGBDFrame&& cp) = default;

  virtual ~RGBDFrame() = default;

  virtual void resize(const size_t w, const size_t h);

  void init(const GpuPtr<uint8_t>& rgb, const GpuPtr<uint16_t>& depth, const cudaStream_t& stream);

  void filterDepth(
      const float sigmaS, const float sigmaC, GpuImg<uint16_t>& tmp, const cudaStream_t& stream);

  void copy(RGBDFrame& dst, const cudaStream_t& stream) const;

  void extractPoints(
      OrderedPointCloud& dst,
      const double scale,
      const geometry::Mat3d& intrinsics,
      const geometry::Mat4d& extrinsics,
      const double near,
      const double far,
      const ColorFormat format,
      const cudaStream_t& stream);
  void render(Surface& surf, const cudaStream_t& stream);

  void exportPNG(const std::string& filename);

  size_t width() const { return width_; }
  size_t height() const { return height_; }

  auto& rgbData() { return rgbData_; }
  const auto& rgbData() const { return rgbData_; }

  auto& depthData() { return depthData_; }
  const auto& depthData() const { return depthData_; }

protected:
  size_t width_{0};
  size_t height_{0};

  GpuImg<geometry::Vec3<uint8_t>> rgbData_;
  GpuImg<uint16_t> depthData_;
};
} // namespace fusion