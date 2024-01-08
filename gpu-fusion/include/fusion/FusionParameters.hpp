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

namespace fusion
{
struct FusionParameters
{
    float voxelRes;
    float tau;
    float near;
    float far;
    float depthScale;
    size_t maxWidth;
    size_t maxHeight;
    math::Mat3f intrinsics;
    math::Mat4f camToSensor; // Converts world camera coordinates to sensor coordinates
};
} // namespace fusion