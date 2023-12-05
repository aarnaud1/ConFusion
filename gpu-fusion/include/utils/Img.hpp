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

#include "common.hpp"
#include "utils/Ptr.hpp"

#include <array>

namespace fusion
{
// TODO : get rid of it
enum class ColorFormat : uint32_t
{
    RGB,
    BGR
};

template <typename T, bool pageLocked>
struct HostImgAllocator
{};
template <typename T>
struct HostImgAllocator<T, true>
{
    static constexpr MemType memory_kind = MemType::DEVICE;
    static inline size_t allocate(T** data, const size_t w, const size_t h)
    {
        gpuErrcheck(cudaMallocHost(data, w * h * sizeof(T)));
        return 0;
    }
    static inline void deallocate(T* data){gpuErrcheck(cudaFreeHost(data))};
};
template <typename T>
struct HostImgAllocator<T, false>
{
    static constexpr MemType memory_kind = MemType::HOST;
    static inline size_t allocate(T** data, const size_t w, const size_t h)
    {
        *data = new T[w * h];
        return 0;
    }
    static inline void deallocate(T* data) { delete[] data; }
};

template <typename T>
struct DeviceImgAllocator
{
    static constexpr MemType memory_kind = MemType::DEVICE;
    static inline size_t allocate(T** data, const size_t w, const size_t h)
    {
        size_t pitch;
        gpuErrcheck(cudaMallocPitch(data, &pitch, w * sizeof(T), h));
        return pitch;
    }
    static inline void deallocate(T* data){gpuErrcheck(cudaFree(data))};
};

template <typename T, typename AllocType>
class BaseImg final
{
  public:
    using value_type = T;
    static constexpr MemType memory_kind = AllocType::memory_kind;

    constexpr BaseImg() noexcept = default;
    BaseImg(const size_t width, const size_t height) : width_{width}, height_{height}
    {
        pitch_ = AllocType::allocate(&data_, width_, height_);
    }
    BaseImg(const BaseImg& cp) : width_{cp.width_}, height_{cp.height_}
    {
        pitch_ = AllocType::allocate(&data_, width_, height_);
        gpuErrcheck(cudaMemcpy2D(
            data_,
            pitch_,
            cp.data_,
            cp.pitch_,
            width_ * sizeof(T),
            height_,
            cudaMemcpyDeviceToDevice));
    }
    BaseImg(BaseImg&&) = default;
    BaseImg& operator=(const BaseImg& cp)
    {
        width_ = cp.width_;
        height_ = cp.height_;
        pitch_ = AllocType::allocate(&data_, width_, height_);
        gpuErrcheck(cudaMemcpy2D(
            data_,
            pitch_,
            cp.data_,
            cp.pitch_,
            width_ * sizeof(T),
            height_,
            cudaMemcpyDeviceToDevice));
        return *this;
    }
    BaseImg& operator=(BaseImg&&) = default;

    ~BaseImg()
    {
        if(data_ != nullptr)
        {
            clear();
        }
    }

    inline T* data() noexcept { return data_; }
    inline const T* data() const noexcept { return data_; }

    inline operator T*() noexcept { return data_; }
    inline operator const T*() const noexcept { return data_; }

    inline size_t width() const { return width_; }
    inline size_t height() const { return height_; }
    inline size_t stride() const { return pitch_ / sizeof(T); }

    void resize(const size_t width, const size_t height)
    {
        clear();
        width_ = width;
        height_ = height;
        pitch_ = AllocType::allocate(&data_, width_, height_);
    }
    void clear()
    {
        AllocType::deallocate(data_);
        data_ = nullptr;
        width_ = 0;
        height_ = 0;
        pitch_ = 0;
    }

    template <typename DstType, typename DstAlloc>
    void uploadTo(BaseImg<DstType, DstAlloc>& dst, const cudaStream_t& stream) const
    {
        using DstPtrType = BaseImg<DstType, AllocType>;
        static_assert(
            DstPtrType::memory_kind == MemType::DEVICE, "Destination for upload must be device");
        static_assert(memory_kind == MemType::HOST, "Source for upload must be host");
        copyImages(dst, *this, stream);
    }

    template <typename SrcType, typename SrcAlloc>
    void uploadFrom(const BaseImg<SrcType, SrcAlloc>& src, const cudaStream_t& stream)
    {
        using SrcPtrType = BaseImg<SrcType, SrcAlloc>;
        static_assert(
            memory_kind == MemType::DEVICE, "Destination for uploadFrom() must be device");
        static_assert(
            SrcPtrType::memory_kind == MemType::HOST, "Source for uploadFrom() must be host");
        copyImages(*this, src, stream);
    }

    template <typename SrcType, typename SrcAlloc>
    void uploadFrom(const BasePtr<SrcType, SrcAlloc>& src, const cudaStream_t& stream)
    {
        using SrcPtrType = BasePtr<SrcType, SrcAlloc>;
        static_assert(
            memory_kind == MemType::DEVICE, "Destination for uploadFrom() must be device");
        static_assert(
            SrcPtrType::memory_kind == MemType::HOST, "Source for uploadFrom() must be host");
        const size_t sizeBytes = width_ * height_ * sizeof(T);
        if(src.sizeBytes() < sizeBytes)
        {
            throw std::runtime_error("Src buffer not large enough");
        }
        gpuErrcheck(cudaMemcpy2DAsync(
            this->data(),
            this->pitch_,
            src.data(),
            width_ * sizeof(SrcType),
            width_ * sizeof(SrcType),
            height_,
            cudaMemcpyHostToDevice,
            stream));
    }

    template <typename DstType, typename DstAlloc>
    void downloadTo(BaseImg<DstType, DstAlloc>& dst, const cudaStream_t& stream) const
    {
        using DstPtrType = BaseImg<DstType, DstAlloc>;
        static_assert(memory_kind == MemType::DEVICE, "Source for downloadTo() must be device");
        static_assert(
            DstPtrType::memory_kind == MemType::HOST, "Destination for downloadTo() must be host");
        copyImages(dst, *this, stream);
    }

    template <typename DstType, typename DstAlloc>
    void downloadTo(BasePtr<DstType, DstAlloc>& dst, const cudaStream_t& stream) const
    {
        using DstPtrType = BasePtr<DstType, DstAlloc>;
        static_assert(memory_kind == MemType::DEVICE, "Source for downloadTo() must be device");
        static_assert(
            DstPtrType::memory_kind == MemType::HOST, "Destination for downloadTo() must be host");
        const size_t sizeBytes = width_ * height_ * sizeof(T);
        if(dst.sizeBytes() < sizeBytes)
        {
            throw std::runtime_error("Dst buffer not large enough");
        }
        gpuErrcheck(cudaMemcpy2DAsync(
            dst.data(),
            width_ * sizeof(DstType),
            this->data(),
            this->pitch_,
            width_ * sizeof(DstType),
            height_,
            cudaMemcpyDeviceToHost,
            stream));
    }

    template <typename SrcType, typename SrcAlloc>
    void downloadFrom(const BaseImg<SrcType, SrcAlloc>& src, const cudaStream_t& stream)
    {
        using SrcPtrType = BaseImg<SrcType, SrcAlloc>;
        static_assert(memory_kind == MemType::HOST, "Destination for download must be host");
        static_assert(
            SrcPtrType::memory_kind == MemType::DEVICE, "Source for download must be device");
        copyImages(*this, src, stream);
    }

    template <typename DstType, typename DstAlloc>
    void copyTo(BaseImg<DstType, DstAlloc>& dst, const cudaStream_t& stream) const
    {
        using DstPtrType = BaseImg<DstType, DstAlloc>;
        static_assert(
            memory_kind == DstPtrType::memory_kind,
            "Destination and source must be of the same type for copyTo()");
        copyImages(dst, *this, stream);
    }

    template <typename SrcType, typename SrcAlloc>
    void copyFrom(const BaseImg<SrcType, SrcAlloc>& src, const cudaStream_t& stream)
    {
        using SrcPtrType = BaseImg<SrcType, SrcAlloc>;
        static_assert(
            memory_kind == SrcPtrType::memory_kind,
            "Destination and source must be of the same type for copyFrom()");
        copyImages(src, *this, stream);
    }

  private:
    size_t width_{0};
    size_t height_{0};
    size_t pitch_{0};

    T* data_{nullptr};

    template <typename DstType, typename SrcType>
    static void copyImages(DstType& dst, const SrcType& src, const cudaStream_t& stream)
    {
        if(dst.width_ != src.width_ && dst.height_ != src.height_)
        {
            throw std::runtime_error("Images must have the same dimentions for copy");
        }
        if(sizeof(typename DstType::value_type) != sizeof(typename SrcType::value_type))
        {
            throw std::runtime_error("Src and dst images must have a similar type size");
        }
        gpuErrcheck(cudaMemcpy2DAsync(
            dst.data(),
            dst.pitch_,
            src.data(),
            src.pitch_,
            dst.width_ * sizeof(typename DstType::value_type),
            dst.height_,
            TransferType<DstType::memory_kind, SrcType::memory_kind>::value,
            stream));
    }
};

// Types definitions
template <typename T, bool pageLocked>
using CpuImg = BaseImg<T, HostImgAllocator<T, pageLocked>>;

template <typename T>
using GpuImg = BaseImg<T, DeviceImgAllocator<T>>;

} // namespace fusion