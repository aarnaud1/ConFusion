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

#include "io/Ply.hpp"

#include "common.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tinyply.h>

namespace fusion
{
namespace io
{
    namespace ply = tinyply;

    void Ply::savePoints(
        const std::string& filename,
        const std::vector<math::Vec3f>& xyz,
        const std::vector<math::Vec3<uint8_t>>& rgb,
        const std::vector<math::Vec3f>& normals)
    {
        utils::Log::info("Ply", "Exporting %s...", filename.c_str());

        std::filebuf fb;
        fb.open(filename, std::ios::out | std::ios::binary);
        std::ostream os(&fb);
        if(os.fail())
        {
            throw std::runtime_error("Error exporting .ply file");
        }

        ply::PlyFile outFile;

        outFile.add_properties_to_element(
            "vertex",
            {"x", "y", "z"},
            ply::Type::FLOAT32,
            xyz.size(),
            reinterpret_cast<const uint8_t*>(xyz.data()),
            ply::Type::INVALID,
            0);

        if(rgb.size() > 0)
        {
            if(rgb.size() != xyz.size())
            {
                throw std::runtime_error("Error exporting .ply fie : xyz and rgb sizes mismatch");
            }
            outFile.add_properties_to_element(
                "vertex",
                {"red", "green", "blue"},
                ply::Type::UINT8,
                rgb.size(),
                reinterpret_cast<const uint8_t*>(rgb.data()),
                ply::Type::INVALID,
                0);
        }
        if(normals.size() > 0)
        {
            if(normals.size() != xyz.size())
            {
                throw std::runtime_error(
                    "Error exporting .ply fie : xyz and normal sizes mismatch");
            }
            outFile.add_properties_to_element(
                "vertex",
                {"nx", "ny", "nz"},
                ply::Type::FLOAT32,
                normals.size(),
                reinterpret_cast<const uint8_t*>(normals.data()),
                ply::Type::INVALID,
                0);
        }

        outFile.get_comments().push_back("GPU Fusion V1.0");
        outFile.write(os, true);
        utils::Log::info("Ply", "Exporting %s : done", filename.c_str());
    }

    void Ply::saveSurface(
        const std::string& filename,
        const std::vector<math::Vec3f>& xyz,
        const std::vector<math::Vec3<uint8_t>>& rgb,
        const std::vector<math::Vec3i>& triangles)
    {
        utils::Log::info("Ply", "Exporting %s...", filename.c_str());
        if(rgb.size() != xyz.size())
        {
            throw std::runtime_error("Error exporting .ply fie : xyz and rgb sizes mismatch");
        }

        std::filebuf fb;
        fb.open(filename, std::ios::out | std::ios::binary);
        std::ostream os(&fb);
        if(os.fail())
        {
            throw std::runtime_error("Error exporting .ply file");
        }

        ply::PlyFile outFile;
        outFile.add_properties_to_element(
            "vertex",
            {"x", "y", "z"},
            ply::Type::FLOAT32,
            xyz.size(),
            reinterpret_cast<const uint8_t*>(xyz.data()),
            ply::Type::INVALID,
            0);
        outFile.add_properties_to_element(
            "vertex",
            {"red", "green", "blue"},
            ply::Type::UINT8,
            rgb.size(),
            reinterpret_cast<const uint8_t*>(rgb.data()),
            ply::Type::INVALID,
            0);
        outFile.add_properties_to_element(
            "face",
            {"vertex_indices"},
            ply::Type::INT32,
            triangles.size(),
            reinterpret_cast<const uint8_t*>(triangles.data()),
            ply::Type::UINT8,
            3);

        outFile.get_comments().push_back("GPU Fusion V1.0");
        outFile.write(os, true);

        utils::Log::info("Ply", "Exporting %s : done", filename.c_str());
    }

    void Ply::saveSurface(
        const std::string& filename,
        const std::vector<math::Vec3f>& xyz,
        const std::vector<math::Vec3<uint8_t>>& rgb,
        const std::vector<math::Vec3f>& normals,
        const std::vector<math::Vec3i>& triangles)
    {
        utils::Log::info("Ply", "Exporting %s...", filename.c_str());
        if(rgb.size() != xyz.size())
        {
            throw std::runtime_error("Error exporting .ply fie : xyz and rgb sizes mismatch");
        }
        if(normals.size() != xyz.size())
        {
            throw std::runtime_error("Error exporting .ply fie : xyz and normal sizes mismatch");
        }

        std::filebuf fb;
        fb.open(filename, std::ios::out | std::ios::binary);
        std::ostream os(&fb);
        if(os.fail())
        {
            throw std::runtime_error("Error exporting .ply file");
        }

        ply::PlyFile outFile;
        outFile.add_properties_to_element(
            "vertex",
            {"x", "y", "z"},
            ply::Type::FLOAT32,
            xyz.size(),
            reinterpret_cast<const uint8_t*>(xyz.data()),
            ply::Type::INVALID,
            0);
        outFile.add_properties_to_element(
            "vertex",
            {"red", "green", "blue"},
            ply::Type::UINT8,
            rgb.size(),
            reinterpret_cast<const uint8_t*>(rgb.data()),
            ply::Type::INVALID,
            0);
        outFile.add_properties_to_element(
            "vertex",
            {"nx", "ny", "nz"},
            ply::Type::FLOAT32,
            normals.size(),
            reinterpret_cast<const uint8_t*>(normals.data()),
            ply::Type::INVALID,
            0);
        outFile.add_properties_to_element(
            "face",
            {"vertex_indices"},
            ply::Type::INT32,
            triangles.size(),
            reinterpret_cast<const uint8_t*>(triangles.data()),
            ply::Type::UINT8,
            3);

        outFile.get_comments().push_back("GPU Fusion V1.0");
        outFile.write(os, true);

        utils::Log::info("Ply", "Exporting %s : done", filename.c_str());
    }
} // namespace io
} // namespace fusion