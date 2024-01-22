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

#include <chrono>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "attributes.hpp"

#define gpuErrcheck(f)                                                                             \
    {                                                                                              \
        gpuAssert((f), __FILE__, __LINE__);                                                        \
    }

static inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        char msg[512];
        sprintf(msg, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort)
        {
            throw std::runtime_error(std::string(msg));
        }
    }
}

// -------------------------------------------------------------------------------------------------

#define LOG_LEVEL_VERBOSE 0
#define LOG_LEVEL_WARNING 1
#define LOG_LEVEL_ERROR 2
#define LOG_LEVEL_CRITICAL 3

#ifndef NO_DEBUG
#    ifndef LOG_LEVEL
#        define LOG_LEVEL LOG_LEVEL_VERBOSE
#    endif
#else
#    define LOG_LEVEL -1
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wformat-security"
namespace cfs
{
namespace utils
{
    static ATTR_HOST_DEV_INL uint32_t div_up(const uint32_t x, const uint32_t n)
    {
        return (x + n - 1) / n;
    }

    class Log
    {
      public:
        template <typename... Args>
        static inline void message(const char* format, Args... args)
        {
            if constexpr(LogLevel >= 0)
            {
                char buf[lineSize];
                snprintf(buf, lineSize, format, args...);
                fprintf(stdout, "%s\n", buf);
            }
        }
        template <typename... Args>
        static inline void time(const char* tag, const char* format, Args... args)
        {
            if constexpr(LogLevel <= LOG_LEVEL_VERBOSE)
            {
                char buf[lineSize];
                snprintf(buf, lineSize, format, args...);
                fprintf(stdout, "\033[0;32m[%s]: %s\n\033[0m", tag, buf);
            }
        }
        template <typename... Args>
        static inline void info(const char* tag, const char* format, Args... args)
        {
            if constexpr(LogLevel <= LOG_LEVEL_VERBOSE)
            {
                char buf[lineSize];
                snprintf(buf, lineSize, format, args...);
                fprintf(stdout, "\033[0;34m[%s]: %s\n\033[0m", tag, buf);
            }
        }
        template <typename... Args>
        static inline void warning(const char* tag, const char* format, Args... args)
        {
            if constexpr(LogLevel <= LOG_LEVEL_WARNING)
            {
                char buf[lineSize];
                snprintf(buf, lineSize, format, args...);
                fprintf(stdout, "\033[0;33m[%s]: %s\n\033[0m", tag, buf);
                fflush(stdout);
            }
        }
        template <typename... Args>
        static inline void error(const char* tag, const char* format, Args... args)
        {
            if constexpr(LogLevel <= LOG_LEVEL_ERROR)
            {
                char buf[lineSize];
                snprintf(buf, lineSize, format, args...);
                fprintf(stdout, "\033[0;31m[%s]: %s\n\033[0m", tag, buf);
                fflush(stdout);
            }
        }
        template <typename... Args>
        static inline void critical(const char* tag, const char* format, Args... args)
        {
            char buf[lineSize];
            snprintf(buf, lineSize, format, args...);
            fprintf(stdout, "\033[0;31m[%s]: %s\n\033[0m", tag, buf);
            fflush(stderr);
        }

      private:
        static constexpr size_t lineSize = 1024;
        static constexpr int LogLevel = LOG_LEVEL;
        Log() = default;
    };
} // namespace utils
} // namespace cfs

#pragma GCC diagnostic pop

namespace cfs
{
namespace utils
{
    class Timer
    {
      public:
        inline Timer() = default;
        inline Timer(const char* name) : name_{name}
        {
            start_ = std::chrono::high_resolution_clock::now();
            started_ = true;
        }

        ~Timer() { stop(); }

        inline void start(const char* name)
        {
            stop();
            name_ = std::string(name);
            start_ = std::chrono::high_resolution_clock::now();
            started_ = true;
        }

        inline void stop()
        {
            if(started_)
            {
                stop_ = std::chrono::high_resolution_clock::now();
                started_ = false;
                const auto duration
                    = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
                utils::Log::time(
                    "Timer", "%s took %f [ms]", name_.c_str(), double(duration.count()) / 1000.0);
            }
        }

      private:
        bool started_ = false;
        std::string name_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop_;
    };
} // namespace utils
} // namespace cfs
