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

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <string>
#include <memory>
#include <gflags/gflags.h>

#include <Parameters.hpp>
#include <DepthMapDataset.hpp>

#include <common.hpp>
#include <utils/Ptr.hpp>
#include <fusion/Fusion.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

using Vec3 = fusion::math::Vec3<float>;
using Vec4 = fusion::math::Vec4<float>;

DEFINE_string(dataset, "", "Dataset path");
DEFINE_uint64(frameCount, 0, "Number of frames to load");

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::SetUsageMessage("GPU fusion");

    tests::DepthMapDataset dataset(FLAGS_dataset, FLAGS_frameCount);
    static constexpr float depthScale = 5000.0f;

    std::vector<fusion::CpuFrameType> frames;
    frames.resize(dataset.size());

    std::vector<fusion::math::Mat4d> poses;
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
        const fusion::FusionParameters params{
            0.01f, 0.2f, 0.0f, 5.0f, maxWidth, maxHeight, dataset.intrinsics().GetRotation()};
        fusion::Fusion fusion{params};
        fusion.integrateFrames(frames, poses, depthScale);
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
