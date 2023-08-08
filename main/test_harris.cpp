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

#include <imgproc/Image2D.hpp>
#include <imgproc/Gauss2D.hpp>
#include <imgproc/Harris2D.hpp>
#include <imgproc/Image2D.hpp>
#include <imgproc/ImgCommon.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter" *

DEFINE_string(input, "", "Input image path");
DEFINE_double(k, 0.03, "Detector threshold");
DEFINE_double(eps, 0.001, "Epsilon threshold");

int main(int argc, char** argv)
{
  // Check arguments
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage("Harris detection");

  const auto path = FLAGS_input;
  if(path.empty())
  {
    throw std::runtime_error("Empty input file");
  }
  auto img = cv::imread(path, cv::IMREAD_ANYCOLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  const size_t width = img.cols;
  const size_t height = img.rows;

  auto inputImg = fusion::Image2D<fusion::ImageFormat::R, float>{width, height};
  auto outputImg = fusion::Image2D<fusion::ImageFormat::R, float>{width, height};
  auto gradientsCpu = fusion::CpuPtr<fusion::geometry::Vec2f, true>{width * height};
  auto edgesCpu = fusion::CpuPtr<float, true>{width * height};
  auto harrisImg = fusion::Buffer2D<float>{width, height};

  fusion::Gauss2D<fusion::ImageFormat::R, float> gauss(0.25, 2);
  fusion::Harris2D<float> harris(float(FLAGS_k), width, height);

  std::vector<uchar> outputData;
  outputData.resize(width * height);
  cv::Mat output(img.rows, img.cols, img.type(), outputData.data());

  std::vector<float> gradientsData;
  gradientsData.resize(3 * width * height);
  cv::Mat gradients(height, width, CV_32FC3, gradientsData.data());

  std::vector<float> edgesData;
  edgesData.resize(width * height);
  cv::Mat edges(height, width, CV_32FC1, edgesData.data());

  std::vector<float> responsesData;
  responsesData.resize(width * height);
  cv::Mat responses(height, width, CV_32FC1, responsesData.data());

  std::vector<uint8_t> harrisPoints;
  harrisPoints.resize(width * height);
  cv::Mat harrisOutput(height, width, CV_8UC1, harrisPoints.data());

  fusion::CpuPtr<float, true> outputImgData{width * height};
  fusion::CpuPtr<float, true> responsesImgData{width * height};
  fusion::CpuPtr<float, true> harrisImgData{width * height};

  try
  {
    cudaStream_t stream;
    gpuErrcheck(cudaStreamCreate(&stream));

    fusion::CpuPtr<float, true> imgData{width * height};
    for(size_t i = 0; i < width * height; ++i)
    {
      imgData[i] = float(img.data[i]) / 255.0f;
    }

    inputImg.upload(imgData, stream);
    gauss.filter(inputImg, outputImg, stream);
    harris.compute(outputImg, stream);
    fusion::performNmsSuppression(harris.responses(), harrisImg, stream);

    // Download GPU data
    outputImg.download(outputImgData, stream);
    harris.gradients().download(gradientsCpu, stream);
    harris.edges().download(edgesCpu, stream);
    harris.responses().download(responsesImgData, stream);
    harrisImg.download(harrisImgData, stream);
    cudaStreamSynchronize(stream);

    float gMax = 0.0f;
    float rMax = 0.0;
    for(size_t i = 0; i < width * height; ++i)
    {
      auto grad = gradientsCpu[i];
      gMax = std::max(gMax, grad.Len());
      gradientsData[3 * i] = abs(grad.x);
      gradientsData[3 * i + 1] = abs(grad.y);
      gradientsData[3 * i + 2] = 0.0f;

      auto r = responsesImgData[i];
      rMax = std::max(r, rMax);
      responsesData[i] = r;

      auto e = edgesCpu[i];
      edgesData[i] = e > 0.0f ? 255.0f : 0.0f;
    }
    cv::cvtColor(output, output, cv::COLOR_GRAY2RGB);
    for(size_t i = 0; i < width * height; ++i)
    {
      gradientsData[3 * i] /= gMax;
      gradientsData[3 * i + 1] /= gMax;
      responsesData[i] /= rMax;
      harrisPoints[i] =
          (harrisImgData[i] > FLAGS_eps) ? uint8_t(255.0f * harrisImgData[i] / rMax) : 0;
      output.data[3 * i] = img.data[i];
      output.data[3 * i + 1] = img.data[i];
      output.data[3 * i + 2] = img.data[i];
      if(harrisPoints[i] > 0)
      {
        cv::circle(output, {int(i % width), int(i / width)}, 0, cv::Scalar{0, 0, 255}, 2);
      }
    }
    gpuErrcheck(cudaStreamDestroy(stream));
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "Error : %s\n", e.what());
    return EXIT_FAILURE;
  }

  cv::imshow("input", img);
  cv::imshow("gradients", gradients);
  cv::imshow("edges", edges);
  // cv::imshow("responses", responses);
  // cv::imshow("harris output", harrisOutput);
  // cv::imshow("output", output);
  cv::waitKey();
  return EXIT_SUCCESS;
}

#pragma GCC diagnostic pop