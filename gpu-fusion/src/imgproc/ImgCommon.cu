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

#include "imgProcCommon.inl"
#include "imgproc/ImgCommon.hpp"

namespace fusion
{
template <typename T>
void performNmsSuppression(
    const Buffer2D<T>& inputImg, Buffer2D<T>& outputImg, const cudaStream_t& stream)
{
    if(inputImg.width() != outputImg.width())
    {
        throw std::runtime_error("performNmsSuppression : input and output width mismatch");
    }
    if(inputImg.height() != outputImg.height())
    {
        throw std::runtime_error("performNmsSuppression : input and output height mismatch");
    }
    performNmsSuppression<<<dim3(16, 16), dim3(32, 32), 0, stream>>>(
        inputImg.img().data(),
        outputImg.img().data(),
        inputImg.width(),
        inputImg.height(),
        inputImg.img().stride(),
        outputImg.img().stride());
}

#define INSTANTIATE_FUNCTION(T)                                                                    \
    template void performNmsSuppression(                                                           \
        const Buffer2D<T>& inputImg, Buffer2D<T>& outputImg, const cudaStream_t& stream);
INSTANTIATE_FUNCTION(uint8_t);
INSTANTIATE_FUNCTION(uint16_t);
INSTANTIATE_FUNCTION(uint32_t);
INSTANTIATE_FUNCTION(int8_t);
INSTANTIATE_FUNCTION(int16_t);
INSTANTIATE_FUNCTION(int32_t);
INSTANTIATE_FUNCTION(float);
INSTANTIATE_FUNCTION(double);
#undef INSTANTIATE_FUNCTION

} // namespace fusion