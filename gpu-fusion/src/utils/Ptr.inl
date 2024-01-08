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

#include "common.hpp"
#include "utils/Ptr.hpp"

#include <cuda.h>

namespace fusion
{
template <typename T>
__global__ __launch_bounds__(256) static void setValueKernel(
    T* __restrict__ ptr, const T val, const size_t n)
{
    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        ptr[idx] = val;
    }
}

template <typename T, typename AllocType>
void BasePtr<T, AllocType>::set(const T val, const cudaStream_t& stream)
{
    static_assert(
        BasePtr<T, AllocType>::memory_kind == MemType::DEVICE, "Ptr must be of device type");
    const size_t n = utils::div_up(this->size_, 256);
    setValueKernel<<<n, 256, 0, stream>>>(this->data_, val, this->size_);
}
} // namespace fusion
