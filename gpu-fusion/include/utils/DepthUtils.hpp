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
#include "utils/Ptr.hpp"
#include "utils/Img.hpp"
#include "utils/Buffer2D.hpp"
#include "utils/DepthUtils.hpp"

#include <cuda.h>

namespace fusion
{
namespace utils
{
    void initRGBDFrame(
        const GpuPtr<uint16_t>& depthData,
        const GpuPtr<uint8_t>& rgbData,
        GpuImg<uint16_t>& depth,
        GpuImg<math::Vec3<uint8_t>>& rgb,
        const size_t w,
        const size_t h,
        const cudaStream_t& stream);

    void extractPoints(
        const GpuImg<uint16_t>& depth,
        GpuPtr<math::Vec3f>& points,
        const math::Mat3d& k,
        const size_t depthScale,
        const cudaStream_t& stream);
    void extractPoints(
        const GpuImg<uint16_t>& depth,
        GpuPtr<math::Vec3f>& points,
        GpuPtr<math::Vec3f>& bbox,
        const math::Mat3d& k,
        const size_t depthScale,
        const cudaStream_t& stream);
    void extractPoints(
        const GpuImg<uint16_t>& depth,
        GpuPtr<math::Vec3f>& points,
        GpuPtr<float>& footprints,
        const math::Mat3d& k,
        const size_t depthScale,
        const cudaStream_t& stream);
    void extractPoints(
        const GpuImg<uint16_t>& depth,
        GpuPtr<math::Vec3f>& points,
        GpuPtr<float>& footprints,
        GpuPtr<math::Vec3f>& bbox,
        const math::Mat3d& k,
        const size_t depthScale,
        const cudaStream_t& stream);

    void extractColors(
        const GpuImg<math::Vec3b>& rgb,
        GpuPtr<math::Vec3f>& color,
        const bool useBgr,
        const cudaStream_t& stream);
} // namespace utils
} // namespace fusion