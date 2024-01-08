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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

namespace fusion
{
namespace mc
{
    size_t extractMesh(
        const float *tsdf,
        const float *xx,
        const float *yy,
        const float *zz,
        const float *xy,
        const float *xz,
        const float *yz,
        const float *xyz,
        const float *rgba,
        const float *cxx,
        const float *cyy,
        const float *czz,
        const float *cxy,
        const float *cxz,
        const float *cyz,
        const float *cxyz,
        const float *grad,
        const float *gxx,
        const float *gyy,
        const float *gzz,
        const float *gxy,
        const float *gxz,
        const float *gyz,
        const float *gxyz,
        const float *w,
        const float *wxx,
        const float *wyy,
        const float *wzz,
        const float *wxy,
        const float *wxz,
        const float *wyz,
        const float *wxyz,
        float *triangles,
        float *colors,
        float *normals,
        const float wThr,
        const size_t blockSize,
        const float voxelRes,
        const float *blockPos);

} // namespace mc
} // namespace fusion
