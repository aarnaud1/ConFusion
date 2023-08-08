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
#include <geometry/geometry.hpp>
#include <fusion/VoxelBlock.hpp>
#include <fusion/RenderDepthMap.hpp>
#include <fusion/OrderedPointCloud.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

using Vec3 = fusion::geometry::Vec3<float>;
using Vec4 = fusion::geometry::Vec4<float>;

DEFINE_string(dataset, "", "Dataset path");
DEFINE_uint64(frameCount, 0, "Number of frames to load");

int main(int argc, char** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage("GPU fusion");

  tests::DepthMapDataset dataset(FLAGS_dataset, FLAGS_frameCount);
  static constexpr float depthScale = 5000.0f;

  try
  {
#pragma omp parallel num_threads(4)
    {
      cudaStream_t stream;
      gpuErrcheck(cudaStreamCreate(&stream));

      fusion::CpuPtr<uint16_t, true> depthDataCpu;
      fusion::CpuPtr<uint8_t, true> colorDataCpu;
      fusion::GpuPtr<uint16_t> depthData;
      fusion::GpuPtr<uint8_t> colorData;
      fusion::RenderDepthMap frame;
      fusion::OrderedPointCloud cloud;

#pragma omp for
      for(size_t i = 0; i < dataset.size(); ++i)
      {
        fprintf(stdout, "Initializing frame %zu\n", i);
        const auto& depthInfo = dataset.getDepthInfo()[i];
        const auto& colorInfo = dataset.getColorInfo()[i];

        if(depthInfo.width != colorInfo.width || depthInfo.height != colorInfo.height)
        {
          fprintf(stderr, "Depth and color dims mismatch");
          continue;
        }

        const auto width = colorInfo.width;
        const auto height = colorInfo.height;
        if(width * height > depthDataCpu.size())
        {
          depthDataCpu.resize(width * height);
          depthData.resize(width * height);
          frame.resize(width, height);
          cloud.resize(width, height);
        }
        if(3 * width * height > colorDataCpu.size())
        {
          colorDataCpu.resize(3 * width * height);
          colorData.resize(3 * width * height);
        }

        depthDataCpu.copyFrom(dataset.depthImages()[i].get(), width * height, stream);
        colorDataCpu.copyFrom(dataset.colorImages()[i].get(), 3 * width * height, stream);
        depthDataCpu.uploadTo(depthData, stream);
        colorDataCpu.uploadTo(colorData, stream);

        frame.init(colorData, depthData, stream);
        frame.extractPoints(
            cloud, depthScale, dataset.intrinsics().GetRotation(),
            fusion::geometry::Mat4d::Identity(), 0.1, 5.0, fusion::ColorFormat::BGR, stream);
        cloud.estimateNormals(stream);

        std::string filename = "cloud_" + std::to_string(i) + ".ply";
        cloud.exportPLY(filename, stream);

        // Test rendering
        frame.render(depthScale, dataset.intrinsics().GetRotation(), stream);
        frame.estimateNormals(stream);
        std::string renderedFilename = "mesh_" + std::to_string(i) + ".ply";
        frame.exportPLY(renderedFilename, stream);
      }

      gpuErrcheck(cudaStreamDestroy(stream));
    }
  }
  catch(std::exception& e)
  {
    fprintf(stderr, "%s\n", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
#pragma GCC diagnostic pop
