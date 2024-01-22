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
#include "utils/Buffer2D.hpp"
#include "utils/Img.hpp"
#include "utils/Ptr.hpp"

namespace cfs
{
template <typename T>
void performNmsSuppression(
    const Buffer2D<T>& inputImg, Buffer2D<T>& outputImg, const cudaStream_t& stream);
} // namespace cfs