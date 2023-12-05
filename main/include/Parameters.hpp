/*
 * Copyright (C) 2022 Adrien ARNAUD
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

#include <string>

#include <common.hpp>
#include <math/geometry.hpp>

#define KINECT_IMG_H 480
#define KINECT_IMG_W 640
#define KINECT_IMG_RES (KINECT_IMG_H * KINECT_IMG_W)

#define INTRINSICS 525.0f, 525.0f, 319.5f, 239.5f
#define ICL_INTRINSICS_1 481.20f, -480.0f, 319.5f, 239.5f

#define SYNTH_W 640
#define SYNTH_H 480
#define SYNTH_TAN_FOV tanf((50.0f * M_PI) / 180.0f)
#define SYNTH_FX (float(SYNTH_W / 2) * SYNTH_TAN_FOV)
#define SYNTH_FY (float(SYNTH_H / 2) * SYNTH_TAN_FOV)
#define SYNTH_CX (float) (SYNTH_W / 2)
#define SYNTH_CY (float) (SYNTH_H / 2)
#define SYNTHETIC_0 SYNTH_FX, SYNTH_FY, SYNTH_CX, SYNTH_CY

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

struct CameraParameters
{
    size_t cameraWidth;
    size_t cameraHeight;
    // fusion::CameraIntrinsics<float> depthIntrinsics;
    fusion::math::Mat4<float> OPENGL_TO_CAMERA;
    fusion::math::Mat4<float> AXIS_PERMUT;
    fusion::math::Mat4<float> MODEL_TO_OPENGL;

    CameraParameters() = default;

    CameraParameters& operator=(CameraParameters const& cp) = default;
};

struct FusionParameters
{
    float tau;
    float voxelRes;
};

static const CameraParameters FR1_PARAMS
    = {KINECT_IMG_W,
       KINECT_IMG_H,
       // {INTRINSICS},
       fusion::math::Mat4<float>(
           fusion::math::Vec4<float>(1.0f, 0.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, -1.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, -1.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f)),
       fusion::math::Mat4<float>(
           fusion::math::Vec4<float>(1.0f, 0.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 1.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 1.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f)),
       fusion::math::Mat4<float>(
           fusion::math::Vec4<float>(1.0f, 0.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, -1.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, -1.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f))};

static const CameraParameters ICL1_PARAMS
    = {KINECT_IMG_W,
       KINECT_IMG_H,
       // {ICL_INTRINSICS_1},
       fusion::math::Mat4<float>(
           fusion::math::Vec4<float>(1.0f, 0.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 1.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, -1.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f)),
       fusion::math::Mat4<float>(
           fusion::math::Vec4<float>(1.0f, 0.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 1.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, -1.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f)),
       fusion::math::Mat4<float>(
           fusion::math::Vec4<float>(1.0f, 0.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 1.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, -1.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f))};

static const CameraParameters SYNTHETIC_0_PARAMS
    = {KINECT_IMG_W,
       KINECT_IMG_H,
       // {SYNTHETIC_0},
       fusion::math::Mat4<float>(
           fusion::math::Vec4<float>(1.0f, 0.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 1.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, -1.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f)),
       fusion::math::Mat4<float>(
           fusion::math::Vec4<float>(1.0f, 0.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 1.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, -1.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f)),
       fusion::math::Mat4<float>(
           fusion::math::Vec4<float>(1.0f, 0.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 1.0f, 0.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, -1.0f, 0.0f),
           fusion::math::Vec4<float>(0.0f, 0.0f, 0.0f, 1.0f))};

#pragma GCC diagnostic pop
