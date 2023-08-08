/*
 * Copyright (C) 2022 Adrien ARNAUD
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

#include "DataStreamer.hpp"

DataStreamer::DataStreamer(const char *dataset) : IDataStreamer(dataset) {}

void DataStreamer::RegisterRGBDFrameCallback(const RGBDFrameCallback pCallback)
{
  this->rgbdFrameCallback_ = pCallback;
}

int DataStreamer::StreamNextData()
{
  if(frames_.size() > 0)
  {
    const DataFrameInfo frame = this->frames_.front();
    const std::string depthMapPath = this->dataset_ + "/" + frame.depthName;
    const std::string imgPath = this->dataset_ + "/" + frame.imgName;
    const Pose pose = frame.pose;

    cv::Mat inputDepthMap_ = cv::imread(depthMapPath, cv::IMREAD_ANYDEPTH);
    cv::Mat inputImg_ = cv::imread(imgPath, cv::IMREAD_ANYCOLOR);
    this->width_ = inputImg_.cols;
    this->height_ = inputImg_.rows;

    rgbdFrameCallback_(
        (uint16_t *) inputDepthMap_.data, inputImg_.data,
        Vec3(pose.tx, pose.ty, pose.tz),
        Vec4(pose.qw, pose.qx, pose.qy, pose.qz), this->width_, this->height_);

    this->frames_.pop();
    return this->frames_.size();
  }
  else
  {
    return 0;
  }
}

void DataStreamer::PrepareStreamingData()
{
  fprintf(stdout, "Preparing streaming data...\n");
  ReadData(this->dataset_ + "/output.txt");
  fprintf(stdout, "Preparing streaming data : done\n");
}

void DataStreamer::ReadData(const std::string &filename)
{
  std::string line;
  std::ifstream ifs(filename.c_str());
  if(!ifs)
  {
    char msg[512];
    sprintf(msg, "error while opening %s", filename.c_str());
    throw std::runtime_error(msg);
  }

  while(std::getline(ifs, line))
  {
    std::istringstream iss(line);
    std::string depthName;
    std::string imgName;
    Pose pose;
    if(!(iss >> depthName >> imgName >> pose.tx >> pose.ty >> pose.tz >> pose.qx
         >> pose.qy >> pose.qz >> pose.qw))
    {
      char msg[512];
      sprintf(
          msg, "error while reading %s: %s\n", filename.c_str(),
          strerror(errno));
      exitError(msg);
    }

    pose.qw = -pose.qw; // Invert angle for freiburg dataset
    DataFrameInfo frame = {depthName, imgName, pose};
    frames_.push(frame);
  }
}

// -----------------------------------------------------------------------------

SyntheticDataStreamer::SyntheticDataStreamer(const char *dataset) :
    IDataStreamer(dataset)
{}

void SyntheticDataStreamer::RegisterRGBDFrameCallback(
    const RGBDFrameCallback pCallback)
{
  this->rgbdFrameCallback_ = pCallback;
}

int SyntheticDataStreamer::StreamNextData()
{
  if(frames_.size() > 0)
  {
    const DataFrameInfo frame = this->frames_.front();
    const std::string depthMapPath = this->dataset_ + "/" + frame.depthName;
    // const std::string imgPath = this->dataset_ + "/" + frame.imgName;
    const Pose pose = frame.pose;

    cv::Mat inputDepthMap_ = cv::imread(depthMapPath, cv::IMREAD_ANYDEPTH);
    // cv::Mat inputImg_ = cv::imread(imgPath, cv::IMREAD_ANYCOLOR);
    this->width_ = inputDepthMap_.cols;
    this->height_ = inputDepthMap_.rows;

    rgbdFrameCallback_(
        (uint16_t *) inputDepthMap_.data, nullptr /*inputImg_.data*/,
        Vec3(pose.tx, pose.ty, pose.tz),
        Vec4(pose.qw, pose.qx, pose.qy, pose.qz), this->width_, this->height_);

    this->frames_.pop();
    return this->frames_.size();
  }
  else
  {
    return 0;
  }
}

void SyntheticDataStreamer::PrepareStreamingData()
{
  fprintf(stdout, "Preparing streaming data...\n");
  ReadData(this->dataset_ + "/output.txt");
  fprintf(stdout, "Preparing streaming data : done\n");
}

void SyntheticDataStreamer::ReadData(const std::string &filename)
{
  std::string line;
  std::ifstream ifs(filename.c_str());
  if(!ifs)
  {
    char msg[512];
    sprintf(msg, "error while opening %s", filename.c_str());
    throw std::runtime_error(msg);
  }

  while(std::getline(ifs, line))
  {
    std::istringstream iss(line);
    std::string depthName;
    // std::string imgName;
    Pose pose;
    if(!(iss >> depthName >> /*imgName >>*/ pose.tx >> pose.ty >> pose.tz
         >> pose.qx >> pose.qy >> pose.qz >> pose.qw))
    {
      char msg[512];
      sprintf(
          msg, "error while reading %s: %s\n", filename.c_str(),
          strerror(errno));
      throw std::runtime_error(msg);
    }

    pose.qw = pose.qw; // Invert angle for freiburg dataset
    DataFrameInfo frame = {depthName, "" /*imgName*/, pose};
    frames_.push(frame);
  }
}
