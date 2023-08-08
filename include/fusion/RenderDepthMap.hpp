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
#include "fusion/DepthMap.hpp"

namespace fusion
{
class RenderDepthMap : public RGBDFrame
{
public:
  RenderDepthMap() noexcept = default;
  RenderDepthMap(const size_t width, const size_t height);
  RenderDepthMap(const RenderDepthMap& cp) = default;
  RenderDepthMap(RenderDepthMap&& cp) = default;
  RenderDepthMap& operator=(const RenderDepthMap& cp) = default;
  RenderDepthMap& operator=(RenderDepthMap&& cp) = default;

  void resize(const size_t w, const size_t h) override;

  void render(const float scale, const geometry::Mat3d& intrinsics, const cudaStream_t& stream);
  void estimateNormals(const cudaStream_t& stream);
  void exportPLY(const std::string& flename, const cudaStream_t& stream);

  auto& size() { return size_; }
  const auto& size() const { return size_; }

private:
  GpuPtr<int32_t> size_{1};
  GpuPtr<geometry::Vec3i> triangles_;
  GpuPtr<geometry::Vec3f> points_;
  GpuPtr<geometry::Vec3f> normals_;
  GpuPtr<float> footprints_;
};
}; // namespace fusion