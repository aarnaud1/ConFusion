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

#include "DepthMapDataset.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sstream>

namespace fs = std::filesystem;

namespace tests
{
DepthMapDataset::DepthMapDataset(const std::string& dir, const size_t n) : size_{0}
{
    fs::path path(dir);
    std::vector<std::string> depthNames;
    std::vector<std::string> colorNames;
    std::vector<std::string> poseNames;
    for(auto& entry : fs::directory_iterator(path))
    {
        if(entry.is_directory())
        {
            continue;
        }
        const auto file = fs::path(entry);
        const auto ext = file.extension();
        if(file.filename() == std::string("intrinsics.txt"))
        {}
        else if(ext == std::string(".txt"))
        {
            size_++;
        }
    }

    // Load files
    if(n > 0)
    {
        size_ = std::min(n, size_);
    }

    const auto root = dir + std::string("/");
    readIntrinsics(root + "intrinsics.txt");
    for(size_t i = 0; i < size_; ++i)
    {
        std::string depthName = root + "depth_" + std::to_string(i) + ".png";
        std::string colorName = root + "color_" + std::to_string(i) + ".png";
        std::string poseName = root + "pose_" + std::to_string(i) + ".txt";

        readDepth(depthName);
        readColor(colorName);
        readPose(poseName);
    }
}

void DepthMapDataset::readIntrinsics(const std::string& filename)
{
    auto M = fusion::math::Mat4f::Identity();
    std::string line;
    std::ifstream ifs(filename.c_str());
    if(!ifs)
    {
        char msg[512];
        sprintf(msg, "Error reading : %s", filename.c_str());
        throw std::runtime_error(msg);
    }
    {
        std::getline(ifs, line);
        std::istringstream iss(line);
        iss >> M.c00 >> M.c01 >> M.c02;
    }
    {
        std::getline(ifs, line);
        std::istringstream iss(line);
        iss >> M.c10 >> M.c11 >> M.c12;
    }
    {
        std::getline(ifs, line);
        std::istringstream iss(line);
        iss >> M.c20 >> M.c21 >> M.c22;
    }
    intrinsics_ = M;
}

void DepthMapDataset::readDepth(const std::string& filename)
{
    try
    {
        auto img = cv::imread(filename, cv::IMREAD_ANYDEPTH);
        const auto width = img.cols;
        const auto height = img.rows;
        depthImages_.emplace_back(new uint16_t[width * height]);
        memcpy(depthImages_.back().get(), img.data, width * height * sizeof(uint16_t));
        depthFramesInfo_.emplace_back(width, height);
    }
    catch(std::exception& e)
    {
        char msg[512];
        sprintf(msg, "Error reading : %s", filename.c_str());
        throw std::runtime_error(msg);
    }
}

void DepthMapDataset::readColor(const std::string& filename)
{
    try
    {
        auto img = cv::imread(filename, cv::IMREAD_ANYCOLOR);
        const auto width = img.cols;
        const auto height = img.rows;
        colorImages_.emplace_back(new uint8_t[3 * width * height]);
        memcpy(colorImages_.back().get(), img.data, 3 * width * height * sizeof(uint8_t));
        colorFramesInfo_.emplace_back(width, height);
    }
    catch(std::exception& e)
    {
        char msg[512];
        sprintf(msg, "Error reading : %s", filename.c_str());
        throw std::runtime_error(msg);
    }
}

void DepthMapDataset::readPose(const std::string& filename)
{
    fusion::math::Vec3f T;
    fusion::math::Vec4f Q;
    std::string line;
    std::ifstream ifs(filename.c_str());
    if(!ifs)
    {
        char msg[512];
        sprintf(msg, "Error reading : %s", filename.c_str());
        throw std::runtime_error(msg);
    }
    std::getline(ifs, line);
    std::istringstream iss(line);
    iss >> T.x >> T.y >> T.z >> Q.x >> Q.y >> Q.z >> Q.w;

    const auto M = fusion::math::Mat4f::Affine(fusion::math::Vec4f{Q.x, Q.y, Q.z, -Q.w}, T);
    const fusion::math::Mat4f axis{
        fusion::math::Vec4f(1.0f, 0.0f, 0.0f, 0.0f),
        fusion::math::Vec4f(0.0f, 1.0f, 0.0f, 0.0f),
        fusion::math::Vec4f(0.0f, 0.0f, -1.0f, 0.0f),
        fusion::math::Vec4f(0.0f, 0.0f, 0.0f, 1.0f)};
    poses_.emplace_back(axis * M);
}

} // namespace tests