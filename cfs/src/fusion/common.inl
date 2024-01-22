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

#include "attributes.hpp"
#include "math/geometry.hpp"

#include <cuda.h>

namespace cfs
{
__global__ static void transformPointsKernel(
    math::Vec3f* __restrict__ points,
    math::Vec3f* __restrict__ normals,
    const math::Mat4d& m,
    const size_t n)
{
    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        points[idx] = m * math::Vec4d(points[idx], 1.0);
        normals[idx] = m.GetRotation() * math::Vec3d(normals[idx]);
    }
}
} // namespace cfs