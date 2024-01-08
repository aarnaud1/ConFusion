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

#include "common.inl"
#include "fusion/Surface.hpp"
#include "io/Ply.hpp"

namespace fusion
{
Surface::Surface(const size_t maxPointCount, const size_t maxTriangleCount)
    : maxPointCount_{maxPointCount}
    , maxTriangleCount_{maxTriangleCount}
    , pointCount_{0}
    , triangleCount_{0}
    , points_{maxPointCount_}
    , colors_{maxPointCount_}
    , triangles_{maxPointCount_}
{}

void Surface::resize(const size_t maxPointCount, const size_t maxTriangleCount)
{
    maxPointCount_ = maxPointCount;
    maxTriangleCount_ = maxTriangleCount;
    pointCount_ = 0;
    triangleCount_ = 0;
    points_.resize(maxPointCount_);
    colors_.resize(maxPointCount_);
    triangles_.resize(maxPointCount_);
}

void Surface::transform(const math::Mat4d& m, const cudaStream_t& stream)
{
    transformPointsKernel<<<1024, 32, 0, stream>>>(points_, normals_, m, pointCount_);
}

void Surface::exportPLY(const std::string& filename, const cudaStream_t& stream)
{
    CpuPtr<math::Vec3f, true> pointsCpu(points_.size());
    CpuPtr<math::Vec3f, true> colorsCpu(colors_.size());
    CpuPtr<math::Vec3f, true> normalsCpu(normals_.size());
    CpuPtr<math::Vec3i, true> trianglesCpu(triangles_.size());

    // TODO : make it more async
    pointsCpu.downloadFrom(points_, stream);
    colorsCpu.downloadFrom(colors_, stream);
    normalsCpu.downloadFrom(normals_, stream);
    trianglesCpu.downloadFrom(triangles_, stream);
    cudaStreamSynchronize(stream);

    std::vector<math::Vec3f> xyz;
    std::vector<math::Vec3<uint8_t>> rgb;
    std::vector<math::Vec3f> norm;
    std::vector<math::Vec3i> tri;
    xyz.reserve(pointCount_);
    rgb.reserve(pointCount_);
    norm.reserve(pointCount_);
    tri.reserve(triangleCount_);

    for(size_t i = 0; i < pointCount_; ++i)
    {
        xyz.emplace_back(pointsCpu[i]);
        norm.emplace_back(normalsCpu[i]);
        rgb.emplace_back(255.0f * colorsCpu[i]);
    }
    for(size_t i = 0; i < triangleCount_; ++i)
    {
        tri.emplace_back(trianglesCpu[i]);
    }

    io::Ply::saveSurface(filename, xyz, rgb, norm, tri);
}
} // namespace fusion