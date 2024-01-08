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

#include "math/geometry.hpp"

#include <string>
#include <vector>

namespace fusion
{
namespace io
{
    class Ply
    {
      public:
        static void savePoints(
            const std::string& filename,
            const std::vector<math::Vec3f>& xyz,
            const std::vector<math::Vec3<uint8_t>>& rgb = {},
            const std::vector<math::Vec3f>& normals = {});
        static void saveSurface(
            const std::string& filename,
            const std::vector<math::Vec3f>& xyz,
            const std::vector<math::Vec3<uint8_t>>& rgb,
            const std::vector<math::Vec3i>& triangles);
        static void saveSurface(
            const std::string& filename,
            const std::vector<math::Vec3f>& xyz,
            const std::vector<math::Vec3<uint8_t>>& rgb,
            const std::vector<math::Vec3f>& normals,
            const std::vector<math::Vec3i>& triangles);

      private:
        Ply() = delete;
    };
} // namespace io
} // namespace fusion