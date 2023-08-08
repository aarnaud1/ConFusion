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
#include "geometry/geometry.hpp"
#include "fusion/Surface.hpp"

namespace fusion
{
class OrderedPointCloud
{
public:
  OrderedPointCloud() noexcept = default;
  OrderedPointCloud(const size_t width, const size_t height);
  OrderedPointCloud(const OrderedPointCloud&) = delete;
  OrderedPointCloud(OrderedPointCloud&&) = delete;
  OrderedPointCloud& operator=(const OrderedPointCloud&) = delete;
  OrderedPointCloud& operator=(OrderedPointCloud&&) = delete;

  void resize(const size_t width, const size_t height);

  void copy(OrderedPointCloud& dst, const cudaStream_t& stream);
  void transform(const geometry::Mat4d& m, const cudaStream_t& stream);
  void estimateNormals(const cudaStream_t& stream);
  void render(Surface& surf, const cudaStream_t& stream);

  void exportPLY(const std::string& filename, const cudaStream_t& stream);

  virtual ~OrderedPointCloud() = default;

  size_t width() const { return width_; }
  size_t height() const { return height_; }

  auto& points() { return points_; }
  const auto& points() const { return points_; }

  auto& normals() { return normals_; }
  const auto& normals() const { return normals_; }

  auto& colors() { return colors_; }
  const auto& colors() const { return colors_; }

  auto& masks() { return masks_; }
  const auto& masks() const { return masks_; }

private:
  size_t width_{0};
  size_t height_{0};

  GpuPtr<geometry::Vec3f> points_{};
  GpuPtr<geometry::Vec3f> normals_{};
  GpuPtr<geometry::Vec3f> colors_{};
  GpuPtr<uint8_t> masks_{};
};
} // namespace fusion