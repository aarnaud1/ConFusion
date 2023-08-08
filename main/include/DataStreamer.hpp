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

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "global.hpp"

#include <geometry/geometry.hpp>

using Vec3 = fusion::geometry::Vec3<float>;
using Vec4 = fusion::geometry::Vec4<float>;

typedef void (*RGBDFrameCallback)(
    const uint16_t *,
    const uint8_t *,
    const Vec3 &,
    const Vec4 &,
    const size_t,
    const size_t);

struct Pose
{
  float tx, ty, tz, qx, qy, qz, qw;
};

struct DataFrameInfo
{
  std::string depthName;
  std::string imgName;
  Pose pose;

  DataFrameInfo() = default;

  DataFrameInfo(const DataFrameInfo &cp) = default;

  DataFrameInfo &operator=(const DataFrameInfo &cp) = default;
};

// -----------------------------------------------------------------------------

class IDataStreamer
{
public:
  IDataStreamer(const char *dataset) : dataset_(dataset) {}

  virtual ~IDataStreamer() {}

  virtual void RegisterRGBDFrameCallback(const RGBDFrameCallback pCallback) = 0;
  virtual int StreamNextData() = 0;
  virtual void PrepareStreamingData() = 0;

protected:
  const std::string dataset_;
  int width_;
  int height_;

  RGBDFrameCallback rgbdFrameCallback_;
  std::queue<DataFrameInfo> frames_;

  virtual void ReadData(const std::string &filename) = 0;
};

class DataStreamer : public IDataStreamer
{
public:
  DataStreamer(const char *dataset);

  void RegisterRGBDFrameCallback(const RGBDFrameCallback pCallback) override;
  int StreamNextData() override;
  void PrepareStreamingData() override;

private:
  void ReadData(const std::string &filename) override;
};

class SyntheticDataStreamer : public IDataStreamer
{
public:
  SyntheticDataStreamer(const char *dataset);

  void RegisterRGBDFrameCallback(const RGBDFrameCallback pCallback) override;
  int StreamNextData() override;
  void PrepareStreamingData() override;

private:
  void ReadData(const std::string &filename) override;
};
