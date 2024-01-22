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

#include <DepthMapDataset.hpp>
#include <chrono>
#include <common.hpp>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fusion/Fusion.hpp>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <string>
#include <utils/Ptr.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

using Vec3 = fusion::math::Vec3<float>;
using Vec4 = fusion::math::Vec4<float>;

DEFINE_string(dataset, "", "Dataset path");
DEFINE_uint64(frameCount, 0, "Number of frames to load");
DEFINE_double(maxDepth, 10.0, "Max depth");
DEFINE_double(voxelRes, 0.1, "Voxel resolution");
DEFINE_double(tau, 0.1, "Truncation distance (in meters)");

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::SetUsageMessage("GPU fusion");

    tests::DepthMapDataset dataset(FLAGS_dataset, FLAGS_frameCount);
    static constexpr float depthScale = 5000.0f;

    std::vector<fusion::CpuFrameType> frames;
    frames.resize(dataset.size());

    std::vector<fusion::math::Mat4f> poses;
    poses.resize(dataset.size());

    size_t maxWidth = 0;
    size_t maxHeight = 0;
    fusion::utils::Log::message("Reading %zu frames from the input dataset\n", dataset.size());
#pragma omp parallel for
    for(size_t idx = 0; idx < dataset.size(); idx += 10)
    {
        fusion::utils::Log::message(
            "Loading frames %zu to %zu", idx, std::min(idx + 10, dataset.size() - 1));
        for(size_t id = 0; (id < 10) && (idx + id < dataset.size()); ++id)
        {
            const size_t i = idx + id;
            const auto& depthInfo = dataset.getDepthInfo()[i];
            const auto& colorInfo = dataset.getColorInfo()[i];
            const auto width = colorInfo.width;
            const auto height = colorInfo.height;
            if(depthInfo.width != colorInfo.width || depthInfo.height != colorInfo.height)
            {
                fusion::utils::Log::warning("Dataset reading", "Depth and color sizes mismatch\n");
                continue;
            }
            maxWidth = std::max(width, maxWidth);
            maxHeight = std::max(height, maxHeight);
            frames[i].resize(width, height);
            poses[i] = dataset.poses()[i];

            // Copy data
            memcpy(
                frames[i].depth().data(),
                dataset.depthImages()[i].get(),
                width * height * sizeof(uint16_t));
            memcpy(
                frames[i].rgb().data(),
                dataset.colorImages()[i].get(),
                3 * width * height * sizeof(uint8_t));
        }
    }
    fusion::utils::Log::message("Reading frames : done");

    fusion::utils::Log::message("Running fusion...");
    try
    {
        const float maxDepth = static_cast<float>(FLAGS_maxDepth);
        const float voxelRes = static_cast<float>(FLAGS_voxelRes);
        const float tau = static_cast<float>(FLAGS_tau);

        fusion::FusionParameters params;
        params.voxelRes = voxelRes;
        params.tau = tau;
        params.near = 0.0f;
        params.far = maxDepth;
        params.depthScale = depthScale;
        params.maxWidth = maxWidth;
        params.maxHeight = maxHeight;
        params.intrinsics = dataset.intrinsics().GetRotation();
        params.camToSensor = fusion::math::Mat4f{
            fusion::math::Vec4f{0.0f, -1.0f, 0.0f, 0.0f},
            fusion::math::Vec4f{1.0f, 0.0f, 0.0f, 0.0f},
            fusion::math::Vec4f{0.0f, 0.0f, -1.0f, 0.0f},
            fusion::math::Vec4f{0.0f, 0.0f, 0.0f, 1.0f}};

        fusion::Fusion fusion{params};
        fusion.integrateFrames(frames, poses);
        fusion.exportFinalMesh("output.ply");

        // Export frames
        // int frameCount = 0;
        // for(size_t frameId = 0; frameId < frames.size(); ++frameId)
        // {
        //     char filename[512];
        //     snprintf(filename, 512, "frame_%d.ply", frameCount);
        //     fusion.exportFrame(frames[frameId], poses[frameId], filename);
        //     snprintf(filename, 512, "frame_untransformed_%d.ply", frameCount);
        //     fusion.exportFrame(
        //         frames[frameId], fusion::math::Mat4f::Inverse(params.camToSensor), filename);
        //     frameCount++;
        // }
    }
    catch(const std::exception& e)
    {
        fusion::utils::Log::critical("Running fusion", "%s", e.what());
        return EXIT_FAILURE;
    }
    fusion::utils::Log::message("Running fusion : done");

    return EXIT_SUCCESS;
}
#pragma GCC diagnostic pop
