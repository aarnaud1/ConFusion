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

#pragma once

#include "imgproc/Image2D.hpp"
#include "imgproc/formats.hpp"
#include "utils/Img.hpp"
#include "utils/Ptr.hpp"

namespace fusion
{
template <ImageFormat format, typename T>
class Gauss2D
{
  public:
    Gauss2D() = delete;
    Gauss2D(const double sigma, const size_t k = 1);

    Gauss2D(const Gauss2D&) = default;
    Gauss2D(Gauss2D&&) = default;
    Gauss2D& operator=(const Gauss2D&) = default;
    Gauss2D& operator=(Gauss2D&&) = default;

    void filter(
        const Image2D<format, T>& input, Image2D<format, T>& output, const cudaStream_t& stream);

  private:
    size_t k_;
    size_t kernelStride_;
    size_t kernelSize_;
    CpuPtr<double, true> kernelCpu_;
    GpuPtr<double> kernelGpu_;
};
} // namespace fusion