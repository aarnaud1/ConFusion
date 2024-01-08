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

#include <type_traits>

namespace fusion
{
template <typename T>
class Canny2D final
{
    static_assert(std::is_arithmetic_v<T>, "Canny detector works only with arithmetic types");

  public:
    Canny2D() = delete;
    Canny2D(const double minThr, const double maxThr) : minThr_{minThr}, maxThr_{maxThr} {}

    Canny2D(const Canny2D&) = delete;
    Canny2D(Canny2D&&) = default;
    Canny2D& operator=(const Canny2D&) = delete;
    Canny2D& operator=(Canny2D&&) = default;

    void getGradients(
        const Image2D<ImageFormat::R, T>& input,
        Image2D<ImageFormat::R, float>& gradients,
        const cudaStream_t& stream);
    void extractEdges(
        const Image2D<ImageFormat::R, float>& gradients,
        Image2D<ImageFormat::R, uint8_t>& edges,
        const cudaStream_t& stream);

  private:
    double minThr_;
    double maxThr_;
};
} // namespace fusion