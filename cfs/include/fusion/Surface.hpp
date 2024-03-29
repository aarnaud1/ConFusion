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
#include "utils/Ptr.hpp"

namespace cfs
{
class Surface
{
  public:
    Surface() = default;
    Surface(const size_t maxPointCount, const size_t maxTriangleCount);
    Surface(const Surface&) = default;
    Surface(Surface&&) = default;
    Surface& operator=(const Surface&) = default;
    Surface& operator=(Surface&&) = default;

    void resize(const size_t maxPointCount, const size_t maxTriangleCount);

    void transform(const math::Mat4d& m, const cudaStream_t& stream);
    void exportPLY(const std::string& filename, const cudaStream_t& stream);

    size_t& pointCount() { return pointCount_; }
    const size_t& pointCount() const { return pointCount_; }
    size_t& triangleCount() { return triangleCount_; }
    const size_t& triangleCount() const { return triangleCount_; }

    size_t maxPointCount() const { return maxPointCount_; }
    size_t maxTriangleCount() const { return maxTriangleCount_; }

    auto& points() { return points_; }
    const auto& points() const { return points_; }

    auto& colors() { return colors_; }
    const auto& colors() const { return colors_; }

    auto& normals() { return normals_; }
    const auto& normals() const { return normals_; }

    auto& triangles() { return triangles_; }
    const auto& triangles() const { return triangles_; }

  private:
    size_t maxPointCount_;
    size_t maxTriangleCount_;
    size_t pointCount_;
    size_t triangleCount_;

    GpuPtr<math::Vec3f> points_;
    GpuPtr<math::Vec3f> colors_;
    GpuPtr<math::Vec3f> normals_;
    GpuPtr<math::Vec3i> triangles_;
};
} // namespace cfs