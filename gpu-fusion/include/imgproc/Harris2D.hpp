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

#pragma once

#include "imgproc/Image2D.hpp"
#include "imgproc/formats.hpp"
#include "utils/Img.hpp"
#include "utils/Ptr.hpp"

namespace fusion
{
template <typename T>
class Harris2D
{
  public:
    Harris2D() = delete;
    Harris2D(const float k, const size_t width, const size_t height);
    Harris2D(const Harris2D&);
    Harris2D(Harris2D&&) = default;
    Harris2D& operator=(const Harris2D&);
    Harris2D& operator=(Harris2D&&) = default;

    void resize(const size_t width, const size_t height);

    void compute(const Image2D<ImageFormat::R, T>& input, const cudaStream_t& stream);

    size_t width() const { return width_; }
    size_t height() const { return height_; }

    auto& gradients() { return gradients_; }
    const auto& gradients() const { return gradients_; }

    auto& edges() { return edges_; }
    const auto& edges() const { return edges_; }

    auto& responses() { return responses_; }
    const auto& responses() const { return responses_; }

  private:
    float kThr_;
    size_t width_;
    size_t height_;

    Buffer2D<math::Vec2f> gradients_;
    Buffer2D<float> edges_;
    Buffer2D<float> responses_;
};
} // namespace fusion