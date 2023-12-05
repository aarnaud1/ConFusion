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

#include <cstdio>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda.h>

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
namespace fusion
{
namespace utils
{
    static inline uint32_t div_up(const uint32_t x, const uint32_t n) { return (x + n - 1) / n; }

    class Log
    {
      public:
        template <typename... Args>
        static inline void message(const char* format, Args... args)
        {
            if constexpr(LogLevel >= 0)
            {
                char buf[1024];
                snprintf(buf, 1024, format, args...);
                fprintf(stdout, "%s\n", buf);
            }
        }
        template <typename... Args>
        static inline void info(const char* tag, const char* format, Args... args)
        {
            if constexpr(LogLevel <= LOG_LEVEL_VERBOSE)
            {
                char buf[1024];
                snprintf(buf, 1024, format, args...);
                fprintf(stdout, "\033[0;34m[%s]: %s\n\033[0m", tag, buf);
            }
        }
        template <typename... Args>
        static inline void warning(const char* tag, const char* format, Args... args)
        {
            if constexpr(LogLevel <= LOG_LEVEL_WARNING)
            {
                char buf[1024];
                snprintf(buf, 1024, format, args...);
                fprintf(stdout, "\033[0;33m[%s]: %s\n\033[0m", tag, buf);
                fflush(stdout);
            }
        }
        template <typename... Args>
        static inline void error(const char* tag, const char* format, Args... args)
        {
            if constexpr(LogLevel <= LOG_LEVEL_ERROR)
            {
                char buf[1024];
                snprintf(buf, 1024, format, args...);
                fprintf(stdout, "\033[0;31m[%s]: %s\n\033[0m", tag, buf);
                fflush(stdout);
            }
        }
        template <typename... Args>
        static inline void critical(const char* tag, const char* format, Args... args)
        {
            char buf[1024];
            snprintf(buf, 1024, format, args...);
            fprintf(stdout, "\033[0;31m[%s]: %s\n\033[0m", tag, buf);
            fflush(stderr);
        }

      private:
        static constexpr int LogLevel = LOG_LEVEL;
        Log() = default;
    };
} // namespace utils
} // namespace fusion

#pragma GCC diagnostic pop

#define CHRONO(f)                                                                                  \
    {                                                                                              \
        auto start = std::chrono::steady_clock::now();                                             \
        f;                                                                                         \
        auto stop = std::chrono::steady_clock::now();                                              \
        const double t                                                                             \
            = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();         \
    }

#define START_CHRONO(M)                                                                            \
    {                                                                                              \
        const char* msg = M;                                                                       \
        auto start = std::chrono::steady_clock::now();

#define STOP_CHRONO()                                                                              \
    auto stop = std::chrono::steady_clock::now();                                                  \
    const double t = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();  \
    utils::Log::info("Timing", "%s took : %f ms\n", msg, t);                                       \
    }
