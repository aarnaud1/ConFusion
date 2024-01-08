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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <gflags/gflags.h>
#include <imgproc/Canny2D.hpp>
#include <imgproc/Gauss2D.hpp>
#include <imgproc/Image2D.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

int main(int argc, char** argv)
{
    auto img = cv::imread("img/img.png", cv::IMREAD_ANYCOLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    const size_t width = img.cols;
    const size_t height = img.rows;

    auto inputImg = fusion::Image2D<fusion::ImageFormat::R, float>{width, height};
    auto outputImg = fusion::Image2D<fusion::ImageFormat::R, float>{width, height};
    auto gradientsImg = fusion::Image2D<fusion::ImageFormat::R, float>{width, height};
    auto edgesImg = fusion::Image2D<fusion::ImageFormat::R, uint8_t>{width, height};

    fusion::Gauss2D<fusion::ImageFormat::R, float> gauss(0.7, 3);

    const double maxThr = 0.005;
    fusion::Canny2D<float> canny(0.5 * maxThr, maxThr);

    std::vector<uchar> outputData;
    outputData.resize(width * height);
    cv::Mat output(img.rows, img.cols, img.type(), outputData.data());

    std::vector<uchar> edgesData;
    edgesData.resize(width * height);
    cv::Mat edges(img.rows, img.cols, CV_8UC1, edgesData.data());

    std::vector<float> gradientsData;
    gradientsData.resize(width * height);
    cv::Mat gradients(img.rows, img.cols, CV_32FC1, gradientsData.data());

    try
    {
        cudaStream_t stream;
        gpuErrcheck(cudaStreamCreate(&stream));
        fusion::CpuPtr<float, true> imgData{width * height};
        fusion::CpuPtr<float, true> outputImgData{width * height};
        fusion::CpuPtr<uint8_t, true> edgesImgData{width * height};
        fusion::CpuPtr<float, true> gradientsImgData{width * height};

        for(size_t i = 0; i < width * height; ++i)
        {
            imgData[i] = float(img.data[i]) / 255.0f;
        }
        inputImg.upload(imgData, stream);
        gauss.filter(inputImg, outputImg, stream);
        canny.getGradients(outputImg, gradientsImg, stream);
        canny.extractEdges(gradientsImg, edgesImg, stream);
        outputImg.download(outputImgData, stream);
        edgesImg.download(edgesImgData, stream);
        gradientsImg.download(gradientsImgData, stream);
        cudaStreamSynchronize(stream);

        float gMax = 0.0;
        for(size_t i = 0; i < width * height; ++i)
        {
            output.data[i] = 255.0f * reinterpret_cast<float*>(outputImgData.data())[i];
            edges.data[i] = edgesImgData.data()[i];
            reinterpret_cast<float*>(gradients.data)[i] = gradientsImgData.data()[i];
            gMax = std::max(gMax, reinterpret_cast<float*>(gradients.data)[i]);
        }

        for(size_t i = 0; i < width * height; ++i)
        {
            reinterpret_cast<float*>(gradients.data)[i] /= gMax;
        }

        gpuErrcheck(cudaStreamDestroy(stream));
    }
    catch(std::exception& e)
    {
        fprintf(stderr, "Error : %s\n", e.what());
        return EXIT_FAILURE;
    }

    cv::imshow("input", img);
    cv::imshow("output", output);
    cv::imshow("edges", edges);
    cv::imshow("gradients", gradients);
    cv::waitKey();
    return EXIT_SUCCESS;
}

#pragma GCC diagnostic pop