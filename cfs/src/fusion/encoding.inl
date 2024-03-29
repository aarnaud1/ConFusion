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

#include "common.hpp"

#include <cstdint>

// See : https://stackoverflow.com/questions/49748864/morton-reverse-encoding-for-a-3d-grid

/* Morton encoding in binary (components 21-bit: 0..2097151)
                0zyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyx */
#define BITMASK_0000000001000001000001000001000001000001000001000001000001000001                   \
    UINT64_C(18300341342965825)
#define BITMASK_0000001000001000001000001000001000001000001000001000001000001000                   \
    UINT64_C(146402730743726600)
#define BITMASK_0001000000000000000000000000000000000000000000000000000000000000                   \
    UINT64_C(1152921504606846976)
/*              0000000ccc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc */
#define BITMASK_0000000000000011000000000011000000000011000000000011000000000011                   \
    UINT64_C(844631138906115)
#define BITMASK_0000000111000000000011000000000011000000000011000000000011000000                   \
    UINT64_C(126113986927919296)
/*              00000000000ccccc00000000cccc00000000cccc00000000cccc00000000cccc */
#define BITMASK_0000000000000000000000000000000000001111000000000000000000001111 UINT64_C(251658255)
#define BITMASK_0000000000000000000000001111000000000000000000001111000000000000                   \
    UINT64_C(1030792212480)
#define BITMASK_0000000000011111000000000000000000000000000000000000000000000000                   \
    UINT64_C(8725724278030336)
/*              000000000000000000000000000ccccccccccccc0000000000000000cccccccc */
#define BITMASK_0000000000000000000000000000000000000000000000000000000011111111 UINT64_C(255)
#define BITMASK_0000000000000000000000000001111111111111000000000000000000000000                   \
    UINT64_C(137422176256)
/*                                                         ccccccccccccccccccccc */
#define BITMASK_21BITS UINT64_C(2097151)

__host__ __device__ __forceinline__ static void morton_decode(
    uint64_t m, uint32_t *xto, uint32_t *yto, uint32_t *zto)
{
    static constexpr uint64_t mask0
        = BITMASK_0000000001000001000001000001000001000001000001000001000001000001;
    static constexpr uint64_t mask1
        = BITMASK_0000001000001000001000001000001000001000001000001000001000001000;
    static constexpr uint64_t mask2
        = BITMASK_0001000000000000000000000000000000000000000000000000000000000000;
    static constexpr uint64_t mask3
        = BITMASK_0000000000000011000000000011000000000011000000000011000000000011;
    static constexpr uint64_t mask4
        = BITMASK_0000000111000000000011000000000011000000000011000000000011000000;
    static constexpr uint64_t mask5
        = BITMASK_0000000000000000000000000000000000001111000000000000000000001111;
    static constexpr uint64_t mask6
        = BITMASK_0000000000000000000000001111000000000000000000001111000000000000;
    static constexpr uint64_t mask7
        = BITMASK_0000000000011111000000000000000000000000000000000000000000000000;
    static constexpr uint64_t mask8
        = BITMASK_0000000000000000000000000000000000000000000000000000000011111111;
    static constexpr uint64_t mask9
        = BITMASK_0000000000000000000000000001111111111111000000000000000000000000;
    uint64_t x = m, y = m >> 1, z = m >> 2;

    /* 000c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c */
    x = (x & mask0) | ((x & mask1) >> 2) | ((x & mask2) >> 4);
    y = (y & mask0) | ((y & mask1) >> 2) | ((y & mask2) >> 4);
    z = (z & mask0) | ((z & mask1) >> 2) | ((z & mask2) >> 4);
    /* 0000000ccc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc */
    x = (x & mask3) | ((x & mask4) >> 4);
    y = (y & mask3) | ((y & mask4) >> 4);
    z = (z & mask3) | ((z & mask4) >> 4);
    /* 00000000000ccccc00000000cccc00000000cccc00000000cccc00000000cccc */
    x = (x & mask5) | ((x & mask6) >> 8) | ((x & mask7) >> 16);
    y = (y & mask5) | ((y & mask6) >> 8) | ((y & mask7) >> 16);
    z = (z & mask5) | ((z & mask6) >> 8) | ((z & mask7) >> 16);
    /* 000000000000000000000000000ccccccccccccc0000000000000000cccccccc */
    x = (x & mask8) | ((x & mask9) >> 16);
    y = (y & mask8) | ((y & mask9) >> 16);
    z = (z & mask8) | ((z & mask9) >> 16);
    /* 0000000000000000000000000000000000000000000ccccccccccccccccccccc */
    if(xto)
        *xto = x;
    if(yto)
        *yto = y;
    if(zto)
        *zto = z;
}

__host__ __device__ __forceinline__ static uint64_t morton_encode(
    const uint32_t xsrc, const uint32_t ysrc, const uint32_t zsrc)
{
    static constexpr uint64_t mask0
        = BITMASK_0000000001000001000001000001000001000001000001000001000001000001;
    static constexpr uint64_t mask1
        = BITMASK_0000001000001000001000001000001000001000001000001000001000001000;
    static constexpr uint64_t mask2
        = BITMASK_0001000000000000000000000000000000000000000000000000000000000000;
    static constexpr uint64_t mask3
        = BITMASK_0000000000000011000000000011000000000011000000000011000000000011;
    static constexpr uint64_t mask4
        = BITMASK_0000000111000000000011000000000011000000000011000000000011000000;
    static constexpr uint64_t mask5
        = BITMASK_0000000000000000000000000000000000001111000000000000000000001111;
    static constexpr uint64_t mask6
        = BITMASK_0000000000000000000000001111000000000000000000001111000000000000;
    static constexpr uint64_t mask7
        = BITMASK_0000000000011111000000000000000000000000000000000000000000000000;
    static constexpr uint64_t mask8
        = BITMASK_0000000000000000000000000000000000000000000000000000000011111111;
    static constexpr uint64_t mask9
        = BITMASK_0000000000000000000000000001111111111111000000000000000000000000;
    uint64_t x = xsrc, y = ysrc, z = zsrc;
    /* 0000000000000000000000000000000000000000000ccccccccccccccccccccc */
    x = (x & mask8) | ((x << 16) & mask9);
    y = (y & mask8) | ((y << 16) & mask9);
    z = (z & mask8) | ((z << 16) & mask9);
    /* 000000000000000000000000000ccccccccccccc0000000000000000cccccccc */
    x = (x & mask5) | ((x << 8) & mask6) | ((x << 16) & mask7);
    y = (y & mask5) | ((y << 8) & mask6) | ((y << 16) & mask7);
    z = (z & mask5) | ((z << 8) & mask6) | ((z << 16) & mask7);
    /* 00000000000ccccc00000000cccc00000000cccc00000000cccc00000000cccc */
    x = (x & mask3) | ((x << 4) & mask4);
    y = (y & mask3) | ((y << 4) & mask4);
    z = (z & mask3) | ((z << 4) & mask4);
    /* 0000000ccc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc */
    x = (x & mask0) | ((x << 2) & mask1) | ((x << 4) & mask2);
    y = (y & mask0) | ((y << 2) & mask1) | ((y << 4) & mask2);
    z = (z & mask0) | ((z << 2) & mask1) | ((z << 4) & mask2);
    /* 000c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c */
    return x | (y << 1) | (z << 2);
}