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

#include <cuda.h>
#include <cstdint>
#include <type_traits>

namespace fusion
{
enum class MemType : uint32_t
{
  HOST,
  DEVICE
};

template <MemType DstMemType, MemType SrcMemType>
struct TransferType
{
  static constexpr cudaMemcpyKind value = cudaMemcpyHostToHost;
};
template <>
struct TransferType<MemType::HOST, MemType::DEVICE>
{
  static constexpr cudaMemcpyKind value = cudaMemcpyDeviceToHost;
};
template <>
struct TransferType<MemType::DEVICE, MemType::DEVICE>
{
  static constexpr cudaMemcpyKind value = cudaMemcpyDeviceToDevice;
};
template <>
struct TransferType<MemType::DEVICE, MemType::HOST>
{
  static constexpr cudaMemcpyKind value = cudaMemcpyHostToDevice;
};

template <typename T, bool pageLocked>
struct HostAllocator
{};
template <typename T>
struct HostAllocator<T, true>
{
  static constexpr MemType memory_kind = MemType::HOST;
  static inline T* allocate(const size_t n)
  {
    T* ret;
    gpuErrcheck(cudaMallocHost(&ret, n * sizeof(T)));
    return ret;
  }
  static inline void deallocate(T* data) { gpuErrcheck(cudaFreeHost(data)); }
};
template <typename T>
struct HostAllocator<T, false>
{
  static constexpr MemType memory_kind = MemType::HOST;
  static inline T* allocate(const size_t n) { return new T[n]; }
  static inline void deallocate(T* data) { delete[] data; }
};

template <typename T>
struct DeviceAllocator
{
  static constexpr MemType memory_kind = MemType::DEVICE;
  static inline T* allocate(const size_t n)
  {
    T* ret;
    gpuErrcheck(cudaMalloc(&ret, n * sizeof(T)));
    return ret;
  }
  static inline void deallocate(T* data) { gpuErrcheck(cudaFree(data)); }
};

template <typename T, typename AllocType>
class BasePtr final
{
public:
  using value_type = T;
  static constexpr MemType memory_kind = AllocType::memory_kind;

  constexpr BasePtr() noexcept : size_{0}, data_{nullptr} {}
  BasePtr(const size_t size) : size_{size}, data_{AllocType::allocate(size_)} {}
  BasePtr(const BasePtr& cp) : size_{cp.size_}, data_{AllocType::allocate(size_)}
  {
    gpuErrcheck(cudaMemcpy(data_, cp.data_, size_ * sizeof(T), cudaMemcpyHostToHost));
  }
  BasePtr(BasePtr&& cp) noexcept = default;
  BasePtr& operator=(const BasePtr& cp)
  {
    size_ = cp.size_;
    data_ = AllocType::allocate(size_);
    gpuErrcheck(cudaMemcpy(data_, cp.data_, size_ * sizeof(T), cudaMemcpyHostToHost));
  }
  BasePtr& operator=(BasePtr&& cp) noexcept = default;

  ~BasePtr() { clear(); }

  inline void clear()
  {
    size_ = 0;
    if(data_ != nullptr)
    {
      AllocType::deallocate(data_);
      data_ = nullptr;
    }
  }
  inline void resize(const size_t n)
  {
    clear();
    size_ = n;
    data_ = AllocType::allocate(size_);
  }

  void set(const T val, const cudaStream_t& stream);

  inline operator T*() noexcept { return data_; }
  inline operator const T*() const noexcept { return data_; }

  inline size_t size() const { return size_; }
  inline size_t sizeBytes() const { return size_ * sizeof(T); }

  inline T* data() noexcept { return data_; }
  inline const T* data() const noexcept { return data_; }

  inline T& operator[](const size_t i) noexcept
  {
    static_assert(memory_kind == MemType::HOST, "Operator[] implemented only for host memory");
    return data_[i];
  }
  inline const T& operator[](const size_t i) const noexcept
  {
    static_assert(memory_kind == MemType::HOST, "Operator[] implemented only for host memory");
    return data_[i];
  }

  inline T* begin() noexcept
  {
    static_assert(memory_kind == MemType::HOST, "begin() implemented only for host memory");
    return data_;
  }
  inline const T* begin() const noexcept
  {
    static_assert(memory_kind == MemType::HOST, "begin() implemented only for host memory");
    return data_;
  }
  inline T* end() noexcept
  {
    static_assert(memory_kind == MemType::HOST, "end() implemented only for host memory");
    return data_ + size_;
  }
  inline const T* end() const noexcept
  {
    static_assert(memory_kind == MemType::HOST, "end() implemented only for host memory");
    return data_ + size_;
  }

  template <typename SrcType>
  void copyFrom(const SrcType* src, const size_t n, const cudaStream_t& stream)
  {
    gpuErrcheck(cudaMemcpyAsync(
        this->data(), src, n * sizeof(T), TransferType<memory_kind, memory_kind>::value, stream));
  }
  template <typename DstType>
  void copyTo(DstType* src, const size_t n, const cudaStream_t& stream) const
  {
    gpuErrcheck(cudaMemcpyAsync(
        src, this->data(), n * sizeof(T), TransferType<memory_kind, memory_kind>::value, stream));
  }
  template <typename DstType>
  void downloadTo(DstType* dst, const size_t n, const cudaStream_t& stream) const
  {
    gpuErrcheck(cudaMemcpyAsync(dst, this->data(), n * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }

  template <typename DstType, typename DstAlloc>
  void uploadTo(BasePtr<DstType, DstAlloc>& dst, const cudaStream_t& stream) const
  {
    using DstPtrType = BasePtr<DstType, DstAlloc>;
    static_assert(
        DstPtrType::memory_kind == MemType::DEVICE, "Destination for upload must be device");
    static_assert(memory_kind == MemType::HOST, "Source for upload must be host");
    copyBuffers(dst, *this, size_, stream);
  }
  template <typename DstType, typename DstAlloc>
  void uploadTo(BasePtr<DstType, DstAlloc>& dst, const size_t n, const cudaStream_t& stream) const
  {
    using DstPtrType = BasePtr<DstType, DstAlloc>;
    static_assert(
        DstPtrType::memory_kind == MemType::DEVICE, "Destination for upload must be device");
    static_assert(memory_kind == MemType::HOST, "Source for upload must be host");
    copyBuffers(dst, *this, n, stream);
  }

  template <typename SrcType, typename SrcAlloc>
  void uploadFrom(const BasePtr<SrcType, SrcAlloc>& src, const cudaStream_t& stream)
  {
    using SrcPtrType = BasePtr<SrcType, SrcAlloc>;
    static_assert(memory_kind == MemType::DEVICE, "Destination for uploadFrom() must be device");
    static_assert(SrcPtrType::memory_kind == MemType::HOST, "Source for uploadFrom() must be host");
    copyBuffers(*this, src, this->size(), stream);
  }
  template <typename SrcType, typename SrcAlloc>
  void uploadFrom(const BasePtr<SrcType, SrcAlloc>& src, const size_t n, const cudaStream_t& stream)
  {
    using SrcPtrType = BasePtr<SrcType, SrcAlloc>;
    static_assert(memory_kind == MemType::DEVICE, "Destination for uploadFrom() must be device");
    static_assert(SrcPtrType::memory_kind == MemType::HOST, "Source for uploadFrom() must be host");
    copyBuffers(*this, src, n, stream);
  }

  template <typename DstType, typename DstAlloc>
  void downloadTo(BasePtr<DstType, DstAlloc>& src, const cudaStream_t& stream) const
  {
    using DstPtrType = BasePtr<DstType, DstAlloc>;
    static_assert(memory_kind == MemType::DEVICE, "Source for downloadTo() must be device");
    static_assert(
        DstPtrType::memory_kind == MemType::HOST, "Destination for downloadTo() must be host");
    copyBuffers(*this, src, this->size(), stream);
  }
  template <typename DstType, typename DstAlloc>
  void downloadTo(BasePtr<DstType, DstAlloc>& src, const size_t n, const cudaStream_t& stream) const
  {
    using DstPtrType = BasePtr<DstType, DstAlloc>;
    static_assert(memory_kind == MemType::DEVICE, "Source for downloadTo() must be device");
    static_assert(
        DstPtrType::memory_kind == MemType::HOST, "Destination for downloadTo() must be host");
    copyBuffers(*this, src, n, stream);
  }

  template <typename SrcType, typename SrcAlloc>
  void downloadFrom(const BasePtr<SrcType, SrcAlloc>& src, const cudaStream_t& stream)
  {
    using SrcPtrType = BasePtr<SrcType, SrcAlloc>;
    static_assert(memory_kind == MemType::HOST, "Destination for download must be host");
    static_assert(SrcPtrType::memory_kind == MemType::DEVICE, "Source for download must be device");
    copyBuffers(*this, src, src.size(), stream);
  }
  template <typename SrcType, typename SrcAlloc>
  void
  downloadFrom(const BasePtr<SrcType, SrcAlloc>& src, const size_t n, const cudaStream_t& stream)
  {
    using SrcPtrType = BasePtr<SrcType, SrcAlloc>;
    static_assert(memory_kind == MemType::HOST, "Destination for download must be host");
    static_assert(SrcPtrType::memory_kind == MemType::DEVICE, "Source for download must be device");
    copyBuffers(*this, src, n, stream);
  }

  template <typename DstType, typename DstAlloc>
  void copyTo(BasePtr<DstType, DstAlloc>& dst, const cudaStream_t& stream) const
  {
    using DstPtrType = BasePtr<DstType, DstAlloc>;
    static_assert(
        memory_kind == DstPtrType::memory_kind,
        "Destination and source must be of the same type for copyTo()");
    copyBuffers(dst, *this, size(), stream);
  }
  template <typename DstType, typename DstAlloc>
  void copyTo(BasePtr<DstType, DstAlloc>& dst, const size_t n, const cudaStream_t& stream) const
  {
    using DstPtrType = BasePtr<DstType, DstAlloc>;
    static_assert(
        memory_kind == DstPtrType::memory_kind,
        "Destination and source must be of the same type for copyTo()");
    copyBuffers(dst, *this, n, stream);
  }

  template <typename SrcType, typename SrcAlloc>
  void copyFrom(const BasePtr<SrcType, SrcAlloc>& dst, const cudaStream_t& stream)
  {
    using SrcPtrType = BasePtr<SrcType, SrcAlloc>;
    static_assert(
        memory_kind == SrcPtrType::memory_kind,
        "Destination and source must be of the same type for copyFrom()");
    copyBuffers(*this, dst, size(), stream);
  }
  template <typename SrcType, typename SrcAlloc>
  void copyFrom(const BasePtr<SrcType, SrcAlloc>& dst, const size_t n, const cudaStream_t& stream)
  {
    using SrcPtrType = BasePtr<SrcType, SrcAlloc>;
    static_assert(
        memory_kind == SrcPtrType::memory_kind,
        "Destination and source must be of the same type for copyFrom()");
    copyBuffers(*this, dst, n, stream);
  }

  template <typename ImgType>
  void uploadTo(ImgType& img, const cudaStream_t& stream) const
  {
    img.uploadFrom(*this, stream);
  }
  template <typename ImgType>
  void downloadFrom(const ImgType& img, const cudaStream_t& stream)
  {
    img.downloadTo(*this, stream);
  }

private:
  size_t size_ = 0;
  T* data_ = nullptr;

  template <typename DstType, typename SrcType>
  static void
  copyBuffers(DstType& dst, const SrcType& src, const size_t n, const cudaStream_t& stream)
  {
    const size_t sizeBytes = n * sizeof(typename SrcType::value_type);
    if(dst.sizeBytes() < sizeBytes)
    {
      throw std::runtime_error("Dst buffer not large enough for copy");
    }
    if(src.sizeBytes() < sizeBytes)
    {
      throw std::runtime_error("Src buffer not large enough for copy");
    }
    gpuErrcheck(cudaMemcpyAsync(
        dst.data(), src.data(), sizeBytes,
        TransferType<DstType::memory_kind, SrcType::memory_kind>::value, stream));
  }
};

// Types definitions
template <typename T, bool pageLocked>
using CpuPtr = BasePtr<T, HostAllocator<T, pageLocked>>;

template <typename T>
using GpuPtr = BasePtr<T, DeviceAllocator<T>>;
} // namespace fusion