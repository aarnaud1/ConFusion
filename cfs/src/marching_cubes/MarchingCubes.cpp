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

#include "marching_cubes/MarchingCubes.hpp"
#include "marching_cubes/tables.h"

#define ISOVALUE_MAX std::numeric_limits<float>::max()

#define COLOR_ID(i, j, k, blockSize) (((i) + (j) *blockSize + (k) *blockSize * blockSize))

#define GRAD_ID(i, j, k, blockSize) (((i) + (j) *blockSize + (k) *blockSize * blockSize))

#define GRID_OFFSET(i, j, k, blockSize) ((i) + (j) * (blockSize) + (k) * (blockSize) * (blockSize))

#define INVALID_CUBE(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7)                       \
    (tsdf0 == ISOVALUE_MAX || tsdf1 == ISOVALUE_MAX || tsdf2 == ISOVALUE_MAX                       \
     || tsdf3 == ISOVALUE_MAX || tsdf4 == ISOVALUE_MAX || tsdf5 == ISOVALUE_MAX                    \
     || tsdf6 == ISOVALUE_MAX || tsdf7 == ISOVALUE_MAX)

#define CUBE_INDEX(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7, isoValue)               \
    ((tsdf0 < isoValue) | ((tsdf1 < isoValue) << 1) | ((tsdf2 < isoValue) << 2)                    \
     | ((tsdf3 < isoValue) << 3) | ((tsdf4 < isoValue) << 4) | ((tsdf5 < isoValue) << 5)           \
     | ((tsdf6 < isoValue) << 6) | ((tsdf7 < isoValue) << 7))

#define INVALID_WEIGHT(w0, w1, w2, w3, w4, w5, w6, w7)                                             \
    (w0 < wThr) || (w1 < wThr) || (w2 < wThr) || (w3 < wThr) || (w4 < wThr) || (w5 < wThr)         \
        || (w6 < wThr) || (w7 < wThr)

#define INTERPOLATE_POINTS(v, c, g)                                                                \
    if(edgeTable[cubeIndex] & 1)                                                                   \
    {                                                                                              \
        v[0] = interpolate(isoValue, p0, p1, tsdf0, tsdf1);                                        \
        c[0] = interpolate(isoValue, *c0, *c1, tsdf0, tsdf1);                                      \
        g[0] = normalize(interpolate(isoValue, *g0, *g1, tsdf0, tsdf1));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 2)                                                                   \
    {                                                                                              \
        v[1] = interpolate(isoValue, p1, p2, tsdf1, tsdf2);                                        \
        c[1] = interpolate(isoValue, *c1, *c2, tsdf1, tsdf2);                                      \
        g[1] = normalize(interpolate(isoValue, *g1, *g2, tsdf1, tsdf2));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 4)                                                                   \
    {                                                                                              \
        v[2] = interpolate(isoValue, p2, p3, tsdf2, tsdf3);                                        \
        c[2] = interpolate(isoValue, *c2, *c3, tsdf2, tsdf3);                                      \
        g[2] = normalize(interpolate(isoValue, *g2, *g3, tsdf2, tsdf3));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 8)                                                                   \
    {                                                                                              \
        v[3] = interpolate(isoValue, p3, p0, tsdf3, tsdf0);                                        \
        c[3] = interpolate(isoValue, *c3, *c0, tsdf3, tsdf0);                                      \
        g[3] = normalize(interpolate(isoValue, *g3, *g0, tsdf3, tsdf0));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 16)                                                                  \
    {                                                                                              \
        v[4] = interpolate(isoValue, p4, p5, tsdf4, tsdf5);                                        \
        c[4] = interpolate(isoValue, *c4, *c5, tsdf4, tsdf5);                                      \
        g[4] = normalize(interpolate(isoValue, *g4, *g5, tsdf4, tsdf5));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 32)                                                                  \
    {                                                                                              \
        v[5] = interpolate(isoValue, p5, p6, tsdf5, tsdf6);                                        \
        c[5] = interpolate(isoValue, *c5, *c6, tsdf5, tsdf6);                                      \
        g[5] = normalize(interpolate(isoValue, *g5, *g6, tsdf5, tsdf6));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 64)                                                                  \
    {                                                                                              \
        v[6] = interpolate(isoValue, p6, p7, tsdf6, tsdf7);                                        \
        c[6] = interpolate(isoValue, *c6, *c7, tsdf6, tsdf7);                                      \
        g[6] = normalize(interpolate(isoValue, *g6, *g7, tsdf6, tsdf7));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 128)                                                                 \
    {                                                                                              \
        v[7] = interpolate(isoValue, p7, p4, tsdf7, tsdf4);                                        \
        c[7] = interpolate(isoValue, *c7, *c4, tsdf7, tsdf4);                                      \
        g[7] = normalize(interpolate(isoValue, *g7, *g4, tsdf7, tsdf4));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 256)                                                                 \
    {                                                                                              \
        v[8] = interpolate(isoValue, p0, p4, tsdf0, tsdf4);                                        \
        c[8] = interpolate(isoValue, *c0, *c4, tsdf0, tsdf4);                                      \
        g[8] = normalize(interpolate(isoValue, *g0, *g4, tsdf0, tsdf4));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 512)                                                                 \
    {                                                                                              \
        v[9] = interpolate(isoValue, p1, p5, tsdf1, tsdf5);                                        \
        c[9] = interpolate(isoValue, *c1, *c5, tsdf1, tsdf5);                                      \
        g[9] = normalize(interpolate(isoValue, *g1, *g5, tsdf1, tsdf5));                           \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 1024)                                                                \
    {                                                                                              \
        v[10] = interpolate(isoValue, p2, p6, tsdf2, tsdf6);                                       \
        c[10] = interpolate(isoValue, *c2, *c6, tsdf2, tsdf6);                                     \
        g[10] = normalize(interpolate(isoValue, *g2, *g6, tsdf2, tsdf6));                          \
    }                                                                                              \
    if(edgeTable[cubeIndex] & 2048)                                                                \
    {                                                                                              \
        v[11] = interpolate(isoValue, p3, p7, tsdf3, tsdf7);                                       \
        c[11] = interpolate(isoValue, *c3, *c7, tsdf3, tsdf7);                                     \
        g[11] = normalize(interpolate(isoValue, *g3, *g7, tsdf3, tsdf7));                          \
    }

#define EXPORT_TRIANGLES(v, c, g)                                                                  \
    for(int id = 0; triTable[cubeIndex][id] != -1; id += 3)                                        \
    {                                                                                              \
        triangles[9 * numTriangles + 0] = v[triTable[cubeIndex][id + 0]].x;                        \
        colors[9 * numTriangles + 0] = c[triTable[cubeIndex][id + 0]].x;                           \
        normals[9 * numTriangles + 0] = g[triTable[cubeIndex][id + 0]].x;                          \
        triangles[9 * numTriangles + 1] = v[triTable[cubeIndex][id + 0]].y;                        \
        colors[9 * numTriangles + 1] = c[triTable[cubeIndex][id + 0]].y;                           \
        normals[9 * numTriangles + 1] = g[triTable[cubeIndex][id + 0]].y;                          \
        triangles[9 * numTriangles + 2] = v[triTable[cubeIndex][id + 0]].z;                        \
        colors[9 * numTriangles + 2] = c[triTable[cubeIndex][id + 0]].z;                           \
        normals[9 * numTriangles + 2] = g[triTable[cubeIndex][id + 0]].z;                          \
                                                                                                   \
        triangles[9 * numTriangles + 3] = v[triTable[cubeIndex][id + 1]].x;                        \
        colors[9 * numTriangles + 3] = c[triTable[cubeIndex][id + 1]].x;                           \
        normals[9 * numTriangles + 3] = g[triTable[cubeIndex][id + 1]].x;                          \
        triangles[9 * numTriangles + 4] = v[triTable[cubeIndex][id + 1]].y;                        \
        colors[9 * numTriangles + 4] = c[triTable[cubeIndex][id + 1]].y;                           \
        normals[9 * numTriangles + 4] = g[triTable[cubeIndex][id + 1]].y;                          \
        triangles[9 * numTriangles + 5] = v[triTable[cubeIndex][id + 1]].z;                        \
        colors[9 * numTriangles + 5] = c[triTable[cubeIndex][id + 1]].z;                           \
        normals[9 * numTriangles + 5] = g[triTable[cubeIndex][id + 1]].z;                          \
                                                                                                   \
        triangles[9 * numTriangles + 6] = v[triTable[cubeIndex][id + 2]].x;                        \
        colors[9 * numTriangles + 6] = c[triTable[cubeIndex][id + 2]].x;                           \
        normals[9 * numTriangles + 6] = g[triTable[cubeIndex][id + 2]].x;                          \
        triangles[9 * numTriangles + 7] = v[triTable[cubeIndex][id + 2]].y;                        \
        colors[9 * numTriangles + 7] = c[triTable[cubeIndex][id + 2]].y;                           \
        normals[9 * numTriangles + 7] = g[triTable[cubeIndex][id + 2]].y;                          \
        triangles[9 * numTriangles + 8] = v[triTable[cubeIndex][id + 2]].z;                        \
        colors[9 * numTriangles + 8] = c[triTable[cubeIndex][id + 2]].z;                           \
        normals[9 * numTriangles + 8] = g[triTable[cubeIndex][id + 2]].z;                          \
                                                                                                   \
        numTriangles++;                                                                            \
    }

typedef struct __attribute__((packed))
{
    float x, y, z;
} vertex_t;

static inline vertex_t interpolate(
    const float isoValue,
    const vertex_t v0,
    const vertex_t v1,
    const float tsdf0,
    const float tsdf1)
{
    vertex_t ret;

    if(fabsf(isoValue - tsdf0) < 0.00001f)
    {
        return v0;
    }

    if(fabsf(isoValue - tsdf1) < 0.00001f)
    {
        return v1;
    }

    if(fabsf(tsdf0 - tsdf1) < 0.00001f)
    {
        return v0;
    }

    const float mu = (isoValue - tsdf0) / (tsdf1 - tsdf0);
    ret.x = v0.x + mu * (v1.x - v0.x);
    ret.y = v0.y + mu * (v1.y - v0.y);
    ret.z = v0.z + mu * (v1.z - v0.z);

    return ret;
}

static inline vertex_t normalize(const vertex_t v)
{
    const float eps = 0.00001f;
    const float n = eps + sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    vertex_t ret;
    ret.x = v.x / n;
    ret.y = v.y / n;
    ret.z = v.z / n;
    return ret;
}

static inline vertex_t get3DPos(
    const size_t i, const size_t j, const size_t k, const vertex_t org, const float voxelRes)
{
    vertex_t ret;
    ret.x = (float) i * voxelRes + org.x;
    ret.y = (float) j * voxelRes + org.y;
    ret.z = (float) k * voxelRes + org.z;
    return ret;
}

static size_t extractInnerMesh(
    const float *tsdf,
    const float *rgba,
    const float *grad,
    const float *weights,
    float *triangles,
    float *colors,
    float *normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue);

static size_t extractXFaceMesh(
    const float *iTsdf,
    const float *fTsdf,
    const float *irgba,
    const float *frgba,
    const float *grad,
    const float *fgrad,
    const float *iWeights,
    const float *fWeights,
    float *triangles,
    float *colors,
    float *normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue);

static size_t extractYFaceMesh(
    const float *iTsdf,
    const float *fTsdf,
    const float *irgba,
    const float *frgba,
    const float *igrad,
    const float *fgrad,
    const float *iWeights,
    const float *fWeights,
    float *triangles,
    float *colors,
    float *normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue);

static size_t extractZFaceMesh(
    const float *iTsdf,
    const float *fTsdf,
    const float *irgba,
    const float *frgba,
    const float *igrad,
    const float *fgrad,
    const float *iWeights,
    const float *fWeights,
    float *triangles,
    float *colors,
    float *normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue);

static size_t extractXYEdgeMesh(
    const float *iTsdf,
    const float *xTsdf,
    const float *yTsdf,
    const float *xyTsdf,
    const float *irgba,
    const float *xrgba,
    const float *yrgba,
    const float *xyrgba,
    const float *igrad,
    const float *xgrad,
    const float *ygrad,
    const float *xygrad,
    const float *iWeights,
    const float *xWeights,
    const float *yWeights,
    const float *xyWeights,
    float *triangles,
    float *colors,
    float *normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue);

static size_t extractXZEdgeMesh(
    const float *iTsdf,
    const float *xTsdf,
    const float *zTsdf,
    const float *xzTsdf,
    const float *irgba,
    const float *xrgba,
    const float *zrgba,
    const float *xzrgba,
    const float *igrad,
    const float *xgrad,
    const float *zgrad,
    const float *xzgrad,
    const float *iWeights,
    const float *xWeights,
    const float *zWeights,
    const float *xzWeights,
    float *triangles,
    float *colors,
    float *normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue);

static size_t extractYZEdgeMesh(
    const float *iTsdf,
    const float *yTsdf,
    const float *zTsdf,
    const float *yzTsdf,
    const float *irgba,
    const float *yrgba,
    const float *zrgba,
    const float *yzrgba,
    const float *igrad,
    const float *ygrad,
    const float *zgrad,
    const float *yzgrad,
    const float *iWeights,
    const float *yWeights,
    const float *zWeights,
    const float *yzWeights,
    float *triangles,
    float *colors,
    float *normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue);

static size_t extractXYZCornerMesh(
    const float *iTsdf,
    const float *xTsdf,
    const float *yTsdf,
    const float *zTsdf,
    const float *xyTsdf,
    const float *xzTsdf,
    const float *yzTsdf,
    const float *xyzTsdf,
    const float *irgba,
    const float *xrgba,
    const float *yrgba,
    const float *zrgba,
    const float *xyrgba,
    const float *xzrgba,
    const float *yzrgba,
    const float *xyzrgba,
    const float *igrad,
    const float *xgrad,
    const float *ygrad,
    const float *zgrad,
    const float *xygrad,
    const float *xzgrad,
    const float *yzgrad,
    const float *xyzgrad,
    const float *iWeights,
    const float *xWeights,
    const float *yWeights,
    const float *zWeights,
    const float *xyWeights,
    const float *xzWeights,
    const float *yzWeights,
    const float *xyzWeights,
    float *triangles,
    float *colors,
    float *normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue);

namespace cfs
{
namespace mc
{
    size_t extractMesh(
        const float *tsdf,
        const float *xx,
        const float *yy,
        const float *zz,
        const float *xy,
        const float *xz,
        const float *yz,
        const float *xyz,
        const float *rgba,
        const float *cxx,
        const float *cyy,
        const float *czz,
        const float *cxy,
        const float *cxz,
        const float *cyz,
        const float *cxyz,
        const float *grad,
        const float *gxx,
        const float *gyy,
        const float *gzz,
        const float *gxy,
        const float *gxz,
        const float *gyz,
        const float *gxyz,
        const float *w,
        const float *wxx,
        const float *wyy,
        const float *wzz,
        const float *wxy,
        const float *wxz,
        const float *wyz,
        const float *wxyz,
        float *triangles,
        float *colors,
        float *normals,
        const float wThr,
        const size_t blockSize,
        const float voxelRes,
        const float *blockPos)
    {
        const vertex_t org = {blockPos[0], blockPos[1], blockPos[2]};
        size_t numTriangles = 0;

        if(tsdf == NULL)
        {
            return 0;
        }

        numTriangles += extractInnerMesh(
            tsdf, rgba, grad, w, triangles, colors, normals, voxelRes, org, blockSize, wThr, 0.0f);

        if(xx != NULL)
        {
            numTriangles += extractXFaceMesh(
                tsdf,
                xx,
                rgba,
                cxx,
                grad,
                gxx,
                w,
                wxx,
                triangles + 9 * numTriangles,
                colors + 9 * numTriangles,
                normals + 9 * numTriangles,
                voxelRes,
                org,
                blockSize,
                wThr,
                0.0f);
        }

        if(yy != NULL)
        {
            numTriangles += extractYFaceMesh(
                tsdf,
                yy,
                rgba,
                cyy,
                grad,
                gyy,
                w,
                wyy,
                triangles + 9 * numTriangles,
                colors + 9 * numTriangles,
                normals + 9 * numTriangles,
                voxelRes,
                org,
                blockSize,
                wThr,
                0.0f);
        }

        if(zz != NULL)
        {
            numTriangles += extractZFaceMesh(
                tsdf,
                zz,
                rgba,
                czz,
                grad,
                gzz,
                w,
                wzz,
                triangles + 9 * numTriangles,
                colors + 9 * numTriangles,
                normals + 9 * numTriangles,
                voxelRes,
                org,
                blockSize,
                wThr,
                0.0f);
        }

        if(xx != NULL && yy != NULL && xy != NULL)
        {
            numTriangles += extractXYEdgeMesh(
                tsdf,
                xx,
                yy,
                xy,
                rgba,
                cxx,
                cyy,
                cxy,
                grad,
                gxx,
                gyy,
                gxy,
                w,
                wxx,
                wyy,
                wxy,
                triangles + 9 * numTriangles,
                colors + 9 * numTriangles,
                normals + 9 * numTriangles,
                voxelRes,
                org,
                blockSize,
                wThr,
                0.0f);
        }

        if(xx != NULL && zz != NULL && xz != NULL)
        {
            numTriangles += extractXZEdgeMesh(
                tsdf,
                xx,
                zz,
                xz,
                rgba,
                cxx,
                czz,
                cxz,
                grad,
                gxx,
                gzz,
                gxz,
                w,
                wxx,
                wzz,
                wxz,
                triangles + 9 * numTriangles,
                colors + 9 * numTriangles,
                normals + 9 * numTriangles,
                voxelRes,
                org,
                blockSize,
                wThr,
                0.0f);
        }

        if(yy != NULL && zz != NULL && yz != NULL)
        {
            numTriangles += extractYZEdgeMesh(
                tsdf,
                yy,
                zz,
                yz,
                rgba,
                cyy,
                czz,
                cyz,
                grad,
                gyy,
                gzz,
                gyz,
                w,
                wyy,
                wzz,
                wyz,
                triangles + 9 * numTriangles,
                colors + 9 * numTriangles,
                normals + 9 * numTriangles,
                voxelRes,
                org,
                blockSize,
                wThr,
                0.0f);
        }

        if(xx != NULL && yy != NULL && zz != NULL && xy != NULL && xz != NULL && yz != NULL
           && xyz != NULL)
        {
            numTriangles += extractXYZCornerMesh(
                tsdf,
                xx,
                yy,
                zz,
                xy,
                xz,
                yz,
                xyz,
                rgba,
                cxx,
                cyy,
                czz,
                cxy,
                cxz,
                cyz,
                cxyz,
                grad,
                gxx,
                gyy,
                gzz,
                gxy,
                gxz,
                gyz,
                gxyz,
                w,
                wxx,
                wyy,
                wzz,
                wxy,
                wxz,
                wyz,
                wxyz,
                triangles + 9 * numTriangles,
                colors + 9 * numTriangles,
                normals + 9 * numTriangles,
                voxelRes,
                org,
                blockSize,
                wThr,
                0.0f);
        }
        return numTriangles;
    }
} // namespace mc
} // namespace cfs

static size_t extractInnerMesh(
    const float *__restrict__ tsdf,
    const float *__restrict__ rgba,
    const float *__restrict__ grad,
    const float *__restrict__ weights,
    float *__restrict__ triangles,
    float *__restrict__ colors,
    float *__restrict__ normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue)
{
    size_t numTriangles = 0;
    for(size_t k = 0; k < blockSize - 1; k++)
    {
        for(size_t j = 0; j < blockSize - 1; j++)
        {
            for(size_t i = 0; i < blockSize - 1; i++)
            {
                const float tsdf0 = tsdf[GRID_OFFSET(i, j, k, blockSize)];
                const float tsdf1 = tsdf[GRID_OFFSET(i, j + 1, k, blockSize)];
                const float tsdf2 = tsdf[GRID_OFFSET(i + 1, j + 1, k, blockSize)];
                const float tsdf3 = tsdf[GRID_OFFSET(i + 1, j, k, blockSize)];
                const float tsdf4 = tsdf[GRID_OFFSET(i, j, k + 1, blockSize)];
                const float tsdf5 = tsdf[GRID_OFFSET(i, j + 1, k + 1, blockSize)];
                const float tsdf6 = tsdf[GRID_OFFSET(i + 1, j + 1, k + 1, blockSize)];
                const float tsdf7 = tsdf[GRID_OFFSET(i + 1, j, k + 1, blockSize)];

                if(INVALID_CUBE(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7))
                {
                    continue;
                }

                const float w0 = weights[GRID_OFFSET(i, j, k, blockSize)];
                const float w1 = weights[GRID_OFFSET(i, j + 1, k, blockSize)];
                const float w2 = weights[GRID_OFFSET(i + 1, j + 1, k, blockSize)];
                const float w3 = weights[GRID_OFFSET(i + 1, j, k, blockSize)];
                const float w4 = weights[GRID_OFFSET(i, j, k + 1, blockSize)];
                const float w5 = weights[GRID_OFFSET(i, j + 1, k + 1, blockSize)];
                const float w6 = weights[GRID_OFFSET(i + 1, j + 1, k + 1, blockSize)];
                const float w7 = weights[GRID_OFFSET(i + 1, j, k + 1, blockSize)];
                if(INVALID_WEIGHT(w0, w1, w2, w3, w4, w5, w6, w7))
                {
                    continue;
                }

                const uint8_t cubeIndex
                    = CUBE_INDEX(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7, isoValue);

                if(cubeIndex == 0)
                {
                    continue;
                }

                const vertex_t p0 = get3DPos(i, j, k, org, voxelRes);
                const vertex_t p1 = get3DPos(i, j + 1, k, org, voxelRes);
                const vertex_t p2 = get3DPos(i + 1, j + 1, k, org, voxelRes);
                const vertex_t p3 = get3DPos(i + 1, j, k, org, voxelRes);
                const vertex_t p4 = get3DPos(i, j, k + 1, org, voxelRes);
                const vertex_t p5 = get3DPos(i, j + 1, k + 1, org, voxelRes);
                const vertex_t p6 = get3DPos(i + 1, j + 1, k + 1, org, voxelRes);
                const vertex_t p7 = get3DPos(i + 1, j, k + 1, org, voxelRes);

                const vertex_t *c0 = (vertex_t *) rgba + COLOR_ID(i, j, k, blockSize);
                const vertex_t *c1 = (vertex_t *) rgba + COLOR_ID(i, j + 1, k, blockSize);
                const vertex_t *c2 = (vertex_t *) rgba + COLOR_ID(i + 1, j + 1, k, blockSize);
                const vertex_t *c3 = (vertex_t *) rgba + COLOR_ID(i + 1, j, k, blockSize);
                const vertex_t *c4 = (vertex_t *) rgba + COLOR_ID(i, j, k + 1, blockSize);
                const vertex_t *c5 = (vertex_t *) rgba + COLOR_ID(i, j + 1, k + 1, blockSize);
                const vertex_t *c6 = (vertex_t *) rgba + COLOR_ID(i + 1, j + 1, k + 1, blockSize);
                const vertex_t *c7 = (vertex_t *) rgba + COLOR_ID(i + 1, j, k + 1, blockSize);

                const vertex_t *g0 = (vertex_t *) grad + GRAD_ID(i, j, k, blockSize);
                const vertex_t *g1 = (vertex_t *) grad + GRAD_ID(i, j + 1, k, blockSize);
                const vertex_t *g2 = (vertex_t *) grad + GRAD_ID(i + 1, j + 1, k, blockSize);
                const vertex_t *g3 = (vertex_t *) grad + GRAD_ID(i + 1, j, k, blockSize);
                const vertex_t *g4 = (vertex_t *) grad + GRAD_ID(i, j, k + 1, blockSize);
                const vertex_t *g5 = (vertex_t *) grad + GRAD_ID(i, j + 1, k + 1, blockSize);
                const vertex_t *g6 = (vertex_t *) grad + GRAD_ID(i + 1, j + 1, k + 1, blockSize);
                const vertex_t *g7 = (vertex_t *) grad + GRAD_ID(i + 1, j, k + 1, blockSize);

                vertex_t v[12];
                vertex_t c[12];
                vertex_t g[12];

                INTERPOLATE_POINTS(v, c, g);
                EXPORT_TRIANGLES(v, c, g);
            }
        }
    }
    return numTriangles;
}

static size_t extractXFaceMesh(
    const float *__restrict__ iTsdf,
    const float *__restrict__ fTsdf,
    const float *__restrict__ irgba,
    const float *__restrict__ frgba,
    const float *__restrict__ igrad,
    const float *__restrict__ fgrad,
    const float *__restrict__ iWeights,
    const float *__restrict__ fWeights,
    float *__restrict__ triangles,
    float *__restrict__ colors,
    float *__restrict__ normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue)
{
    size_t numTriangles = 0;
    const size_t i0 = blockSize - 1;
    const size_t i1 = 0;
    const vertex_t org0 = org;
    const vertex_t org1 = {org0.x + voxelRes * blockSize, org0.y, org0.z};

    for(size_t k = 0; k < blockSize - 1; k++)
    {
        for(size_t j = 0; j < blockSize - 1; j++)
        {
            const float tsdf0 = iTsdf[GRID_OFFSET(i0, j, k, blockSize)];
            const float tsdf1 = iTsdf[GRID_OFFSET(i0, j + 1, k, blockSize)];
            const float tsdf2 = fTsdf[GRID_OFFSET(i1, j + 1, k, blockSize)];
            const float tsdf3 = fTsdf[GRID_OFFSET(i1, j, k, blockSize)];
            const float tsdf4 = iTsdf[GRID_OFFSET(i0, j, k + 1, blockSize)];
            const float tsdf5 = iTsdf[GRID_OFFSET(i0, j + 1, k + 1, blockSize)];
            const float tsdf6 = fTsdf[GRID_OFFSET(i1, j + 1, k + 1, blockSize)];
            const float tsdf7 = fTsdf[GRID_OFFSET(i1, j, k + 1, blockSize)];

            if(INVALID_CUBE(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7))
            {
                continue;
            }

            const float w0 = iWeights[GRID_OFFSET(i0, j, k, blockSize)];
            const float w1 = iWeights[GRID_OFFSET(i0, j + 1, k, blockSize)];
            const float w2 = fWeights[GRID_OFFSET(i1, j + 1, k, blockSize)];
            const float w3 = fWeights[GRID_OFFSET(i1, j, k, blockSize)];
            const float w4 = iWeights[GRID_OFFSET(i0, j, k + 1, blockSize)];
            const float w5 = iWeights[GRID_OFFSET(i0, j + 1, k + 1, blockSize)];
            const float w6 = fWeights[GRID_OFFSET(i1, j + 1, k + 1, blockSize)];
            const float w7 = fWeights[GRID_OFFSET(i1, j, k + 1, blockSize)];

            if(INVALID_WEIGHT(w0, w1, w2, w3, w4, w5, w6, w7))
            {
                continue;
            }

            const uint8_t cubeIndex
                = CUBE_INDEX(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7, isoValue);

            if(cubeIndex == 0)
            {
                continue;
            }

            const vertex_t p0 = get3DPos(i0, j, k, org0, voxelRes);
            const vertex_t p1 = get3DPos(i0, j + 1, k, org0, voxelRes);
            const vertex_t p2 = get3DPos(i1, j + 1, k, org1, voxelRes);
            const vertex_t p3 = get3DPos(i1, j, k, org1, voxelRes);
            const vertex_t p4 = get3DPos(i0, j, k + 1, org0, voxelRes);
            const vertex_t p5 = get3DPos(i0, j + 1, k + 1, org0, voxelRes);
            const vertex_t p6 = get3DPos(i1, j + 1, k + 1, org1, voxelRes);
            const vertex_t p7 = get3DPos(i1, j, k + 1, org1, voxelRes);

            const vertex_t *c0 = (vertex_t *) irgba + COLOR_ID(i0, j, k, blockSize);
            const vertex_t *c1 = (vertex_t *) irgba + COLOR_ID(i0, j + 1, k, blockSize);
            const vertex_t *c2 = (vertex_t *) frgba + COLOR_ID(i1, j + 1, k, blockSize);
            const vertex_t *c3 = (vertex_t *) frgba + COLOR_ID(i1, j, k, blockSize);
            const vertex_t *c4 = (vertex_t *) irgba + COLOR_ID(i0, j, k + 1, blockSize);
            const vertex_t *c5 = (vertex_t *) irgba + COLOR_ID(i0, j + 1, k + 1, blockSize);
            const vertex_t *c6 = (vertex_t *) frgba + COLOR_ID(i1, j + 1, k + 1, blockSize);
            const vertex_t *c7 = (vertex_t *) frgba + COLOR_ID(i1, j, k + 1, blockSize);

            const vertex_t *g0 = (vertex_t *) igrad + GRAD_ID(i0, j, k, blockSize);
            const vertex_t *g1 = (vertex_t *) igrad + GRAD_ID(i0, j + 1, k, blockSize);
            const vertex_t *g2 = (vertex_t *) fgrad + GRAD_ID(i1, j + 1, k, blockSize);
            const vertex_t *g3 = (vertex_t *) fgrad + GRAD_ID(i1, j, k, blockSize);
            const vertex_t *g4 = (vertex_t *) igrad + GRAD_ID(i0, j, k + 1, blockSize);
            const vertex_t *g5 = (vertex_t *) igrad + GRAD_ID(i0, j + 1, k + 1, blockSize);
            const vertex_t *g6 = (vertex_t *) fgrad + GRAD_ID(i1, j + 1, k + 1, blockSize);
            const vertex_t *g7 = (vertex_t *) fgrad + GRAD_ID(i1, j, k + 1, blockSize);

            vertex_t v[12];
            vertex_t c[12];
            vertex_t g[12];

            INTERPOLATE_POINTS(v, c, g);
            EXPORT_TRIANGLES(v, c, g);
        }
    }
    return numTriangles;
}

static size_t extractYFaceMesh(
    const float *__restrict__ iTsdf,
    const float *__restrict__ fTsdf,
    const float *__restrict__ irgba,
    const float *__restrict__ frgba,
    const float *__restrict__ igrad,
    const float *__restrict__ fgrad,
    const float *__restrict__ iWeights,
    const float *__restrict__ fWeights,
    float *__restrict__ triangles,
    float *__restrict__ colors,
    float *__restrict__ normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue)
{
    size_t numTriangles = 0;
    const size_t j0 = blockSize - 1;
    const size_t j1 = 0;
    const vertex_t org0 = org;
    const vertex_t org1 = {org0.x, org0.y + voxelRes * blockSize, org0.z};

    for(size_t k = 0; k < blockSize - 1; k++)
    {
        for(size_t i = 0; i < blockSize - 1; i++)
        {
            const float tsdf0 = iTsdf[GRID_OFFSET(i, j0, k, blockSize)];
            const float tsdf1 = fTsdf[GRID_OFFSET(i, j1, k, blockSize)];
            const float tsdf2 = fTsdf[GRID_OFFSET(i + 1, j1, k, blockSize)];
            const float tsdf3 = iTsdf[GRID_OFFSET(i + 1, j0, k, blockSize)];
            const float tsdf4 = iTsdf[GRID_OFFSET(i, j0, k + 1, blockSize)];
            const float tsdf5 = fTsdf[GRID_OFFSET(i, j1, k + 1, blockSize)];
            const float tsdf6 = fTsdf[GRID_OFFSET(i + 1, j1, k + 1, blockSize)];
            const float tsdf7 = iTsdf[GRID_OFFSET(i + 1, j0, k + 1, blockSize)];

            if(INVALID_CUBE(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7))
            {
                continue;
            }

            const float w0 = iWeights[GRID_OFFSET(i, j0, k, blockSize)];
            const float w1 = fWeights[GRID_OFFSET(i, j1, k, blockSize)];
            const float w2 = fWeights[GRID_OFFSET(i + 1, j1, k, blockSize)];
            const float w3 = iWeights[GRID_OFFSET(i + 1, j0, k, blockSize)];
            const float w4 = iWeights[GRID_OFFSET(i, j0, k + 1, blockSize)];
            const float w5 = fWeights[GRID_OFFSET(i, j1, k + 1, blockSize)];
            const float w6 = fWeights[GRID_OFFSET(i + 1, j1, k + 1, blockSize)];
            const float w7 = iWeights[GRID_OFFSET(i + 1, j0, k + 1, blockSize)];

            if(INVALID_WEIGHT(w0, w1, w2, w3, w4, w5, w6, w7))
            {
                continue;
            }

            const uint8_t cubeIndex
                = CUBE_INDEX(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7, isoValue);

            if(cubeIndex == 0)
            {
                continue;
            }

            const vertex_t p0 = get3DPos(i, j0, k, org0, voxelRes);
            const vertex_t p1 = get3DPos(i, j1, k, org1, voxelRes);
            const vertex_t p2 = get3DPos(i + 1, j1, k, org1, voxelRes);
            const vertex_t p3 = get3DPos(i + 1, j0, k, org0, voxelRes);
            const vertex_t p4 = get3DPos(i, j0, k + 1, org0, voxelRes);
            const vertex_t p5 = get3DPos(i, j1, k + 1, org1, voxelRes);
            const vertex_t p6 = get3DPos(i + 1, j1, k + 1, org1, voxelRes);
            const vertex_t p7 = get3DPos(i + 1, j0, k + 1, org0, voxelRes);

            const vertex_t *c0 = (vertex_t *) irgba + COLOR_ID(i, j0, k, blockSize);
            const vertex_t *c1 = (vertex_t *) frgba + COLOR_ID(i, j1, k, blockSize);
            const vertex_t *c2 = (vertex_t *) frgba + COLOR_ID(i + 1, j1, k, blockSize);
            const vertex_t *c3 = (vertex_t *) irgba + COLOR_ID(i + 1, j0, k, blockSize);
            const vertex_t *c4 = (vertex_t *) irgba + COLOR_ID(i, j0, k + 1, blockSize);
            const vertex_t *c5 = (vertex_t *) frgba + COLOR_ID(i, j1, k + 1, blockSize);
            const vertex_t *c6 = (vertex_t *) frgba + COLOR_ID(i + 1, j1, k + 1, blockSize);
            const vertex_t *c7 = (vertex_t *) irgba + COLOR_ID(i + 1, j0, k + 1, blockSize);

            const vertex_t *g0 = (vertex_t *) igrad + GRAD_ID(i, j0, k, blockSize);
            const vertex_t *g1 = (vertex_t *) fgrad + GRAD_ID(i, j1, k, blockSize);
            const vertex_t *g2 = (vertex_t *) fgrad + GRAD_ID(i + 1, j1, k, blockSize);
            const vertex_t *g3 = (vertex_t *) igrad + GRAD_ID(i + 1, j0, k, blockSize);
            const vertex_t *g4 = (vertex_t *) igrad + GRAD_ID(i, j0, k + 1, blockSize);
            const vertex_t *g5 = (vertex_t *) fgrad + GRAD_ID(i, j1, k + 1, blockSize);
            const vertex_t *g6 = (vertex_t *) fgrad + GRAD_ID(i + 1, j1, k + 1, blockSize);
            const vertex_t *g7 = (vertex_t *) igrad + GRAD_ID(i + 1, j0, k + 1, blockSize);

            vertex_t v[12];
            vertex_t c[12];
            vertex_t g[12];

            INTERPOLATE_POINTS(v, c, g);
            EXPORT_TRIANGLES(v, c, g);
        }
    }
    return numTriangles;
}

static size_t extractZFaceMesh(
    const float *__restrict__ iTsdf,
    const float *__restrict__ fTsdf,
    const float *__restrict__ irgba,
    const float *__restrict__ frgba,
    const float *__restrict__ igrad,
    const float *__restrict__ fgrad,
    const float *__restrict__ iWeights,
    const float *__restrict__ fWeights,
    float *__restrict__ triangles,
    float *__restrict__ colors,
    float *__restrict__ normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue)
{
    size_t numTriangles = 0;
    const size_t k0 = blockSize - 1;
    const size_t k1 = 0;
    const vertex_t org0 = org;
    const vertex_t org1 = {org0.x, org0.y, org0.z + voxelRes * blockSize};

    for(size_t j = 0; j < blockSize - 1; j++)
    {
        for(size_t i = 0; i < blockSize - 1; i++)
        {
            const float tsdf0 = iTsdf[GRID_OFFSET(i, j, k0, blockSize)];
            const float tsdf1 = iTsdf[GRID_OFFSET(i, j + 1, k0, blockSize)];
            const float tsdf2 = iTsdf[GRID_OFFSET(i + 1, j + 1, k0, blockSize)];
            const float tsdf3 = iTsdf[GRID_OFFSET(i + 1, j, k0, blockSize)];
            const float tsdf4 = fTsdf[GRID_OFFSET(i, j, k1, blockSize)];
            const float tsdf5 = fTsdf[GRID_OFFSET(i, j + 1, k1, blockSize)];
            const float tsdf6 = fTsdf[GRID_OFFSET(i + 1, j + 1, k1, blockSize)];
            const float tsdf7 = fTsdf[GRID_OFFSET(i + 1, j, k1, blockSize)];

            if(INVALID_CUBE(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7))
            {
                continue;
            }

            const float w0 = iWeights[GRID_OFFSET(i, j, k0, blockSize)];
            const float w1 = iWeights[GRID_OFFSET(i, j + 1, k0, blockSize)];
            const float w2 = iWeights[GRID_OFFSET(i + 1, j + 1, k0, blockSize)];
            const float w3 = iWeights[GRID_OFFSET(i + 1, j, k0, blockSize)];
            const float w4 = fWeights[GRID_OFFSET(i, j, k1, blockSize)];
            const float w5 = fWeights[GRID_OFFSET(i, j + 1, k1, blockSize)];
            const float w6 = fWeights[GRID_OFFSET(i + 1, j + 1, k1, blockSize)];
            const float w7 = fWeights[GRID_OFFSET(i + 1, j, k1, blockSize)];

            if(INVALID_WEIGHT(w0, w1, w2, w3, w4, w5, w6, w7))
            {
                continue;
            }

            const uint8_t cubeIndex
                = CUBE_INDEX(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7, isoValue);

            if(cubeIndex == 0)
            {
                continue;
            }

            const vertex_t p0 = get3DPos(i, j, k0, org0, voxelRes);
            const vertex_t p1 = get3DPos(i, j + 1, k0, org0, voxelRes);
            const vertex_t p2 = get3DPos(i + 1, j + 1, k0, org0, voxelRes);
            const vertex_t p3 = get3DPos(i + 1, j, k0, org0, voxelRes);
            const vertex_t p4 = get3DPos(i, j, k1, org1, voxelRes);
            const vertex_t p5 = get3DPos(i, j + 1, k1, org1, voxelRes);
            const vertex_t p6 = get3DPos(i + 1, j + 1, k1, org1, voxelRes);
            const vertex_t p7 = get3DPos(i + 1, j, k1, org1, voxelRes);

            const vertex_t *c0 = (vertex_t *) irgba + COLOR_ID(i, j, k0, blockSize);
            const vertex_t *c1 = (vertex_t *) irgba + COLOR_ID(i, j + 1, k0, blockSize);
            const vertex_t *c2 = (vertex_t *) irgba + COLOR_ID(i + 1, j + 1, k0, blockSize);
            const vertex_t *c3 = (vertex_t *) irgba + COLOR_ID(i + 1, j, k0, blockSize);
            const vertex_t *c4 = (vertex_t *) frgba + COLOR_ID(i, j, k1, blockSize);
            const vertex_t *c5 = (vertex_t *) frgba + COLOR_ID(i, j + 1, k1, blockSize);
            const vertex_t *c6 = (vertex_t *) frgba + COLOR_ID(i + 1, j + 1, k1, blockSize);
            const vertex_t *c7 = (vertex_t *) frgba + COLOR_ID(i + 1, j, k1, blockSize);

            const vertex_t *g0 = (vertex_t *) igrad + GRAD_ID(i, j, k0, blockSize);
            const vertex_t *g1 = (vertex_t *) igrad + GRAD_ID(i, j + 1, k0, blockSize);
            const vertex_t *g2 = (vertex_t *) igrad + GRAD_ID(i + 1, j + 1, k0, blockSize);
            const vertex_t *g3 = (vertex_t *) igrad + GRAD_ID(i + 1, j, k0, blockSize);
            const vertex_t *g4 = (vertex_t *) fgrad + GRAD_ID(i, j, k1, blockSize);
            const vertex_t *g5 = (vertex_t *) fgrad + GRAD_ID(i, j + 1, k1, blockSize);
            const vertex_t *g6 = (vertex_t *) fgrad + GRAD_ID(i + 1, j + 1, k1, blockSize);
            const vertex_t *g7 = (vertex_t *) fgrad + GRAD_ID(i + 1, j, k1, blockSize);

            vertex_t v[12];
            vertex_t c[12];
            vertex_t g[12];

            INTERPOLATE_POINTS(v, c, g);
            EXPORT_TRIANGLES(v, c, g);
        }
    }
    return numTriangles;
}

static size_t extractXYEdgeMesh(
    const float *__restrict__ iTsdf,
    const float *__restrict__ xTsdf,
    const float *__restrict__ yTsdf,
    const float *__restrict__ xyTsdf,
    const float *__restrict__ irgba,
    const float *__restrict__ xrgba,
    const float *__restrict__ yrgba,
    const float *__restrict__ xyrgba,
    const float *__restrict__ igrad,
    const float *__restrict__ xgrad,
    const float *__restrict__ ygrad,
    const float *__restrict__ xygrad,
    const float *__restrict__ iWeights,
    const float *__restrict__ xWeights,
    const float *__restrict__ yWeights,
    const float *__restrict__ xyWeights,
    float *triangles,
    float *colors,
    float *normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue)
{
    size_t numTriangles = 0;

    const vertex_t orgi = org;
    const vertex_t orgx = {orgi.x + voxelRes * blockSize, orgi.y, orgi.z};
    const vertex_t orgy = {orgi.x, orgi.y + voxelRes * blockSize, orgi.z};
    const vertex_t orgxy = {orgi.x + voxelRes * blockSize, orgi.y + voxelRes * blockSize, orgi.z};

    for(size_t k = 0; k < blockSize - 1; k++)
    {
        const float tsdf0 = iTsdf[GRID_OFFSET(blockSize - 1, blockSize - 1, k, blockSize)];
        const float tsdf1 = yTsdf[GRID_OFFSET(blockSize - 1, 0, k, blockSize)];
        const float tsdf2 = xyTsdf[GRID_OFFSET(0, 0, k, blockSize)];
        const float tsdf3 = xTsdf[GRID_OFFSET(0, blockSize - 1, k, blockSize)];
        const float tsdf4 = iTsdf[GRID_OFFSET(blockSize - 1, blockSize - 1, k + 1, blockSize)];
        const float tsdf5 = yTsdf[GRID_OFFSET(blockSize - 1, 0, k + 1, blockSize)];
        const float tsdf6 = xyTsdf[GRID_OFFSET(0, 0, k + 1, blockSize)];
        const float tsdf7 = xTsdf[GRID_OFFSET(0, blockSize - 1, k + 1, blockSize)];

        if(INVALID_CUBE(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7))
        {
            continue;
        }

        const float w0 = iWeights[GRID_OFFSET(blockSize - 1, blockSize - 1, k, blockSize)];
        const float w1 = yWeights[GRID_OFFSET(blockSize - 1, 0, k, blockSize)];
        const float w2 = xyWeights[GRID_OFFSET(0, 0, k, blockSize)];
        const float w3 = xWeights[GRID_OFFSET(0, blockSize - 1, k, blockSize)];
        const float w4 = iWeights[GRID_OFFSET(blockSize - 1, blockSize - 1, k + 1, blockSize)];
        const float w5 = yWeights[GRID_OFFSET(blockSize - 1, 0, k + 1, blockSize)];
        const float w6 = xyWeights[GRID_OFFSET(0, 0, k + 1, blockSize)];
        const float w7 = xWeights[GRID_OFFSET(0, blockSize - 1, k + 1, blockSize)];

        if(INVALID_WEIGHT(w0, w1, w2, w3, w4, w5, w6, w7))
        {
            continue;
        }

        const uint8_t cubeIndex
            = CUBE_INDEX(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7, isoValue);

        if(cubeIndex == 0)
        {
            continue;
        }

        const vertex_t p0 = get3DPos(blockSize - 1, blockSize - 1, k, orgi, voxelRes);
        const vertex_t p1 = get3DPos(blockSize - 1, 0, k, orgy, voxelRes);
        const vertex_t p2 = get3DPos(0, 0, k, orgxy, voxelRes);
        const vertex_t p3 = get3DPos(0, blockSize - 1, k, orgx, voxelRes);
        const vertex_t p4 = get3DPos(blockSize - 1, blockSize - 1, k + 1, orgi, voxelRes);
        const vertex_t p5 = get3DPos(blockSize - 1, 0, k + 1, orgy, voxelRes);
        const vertex_t p6 = get3DPos(0, 0, k + 1, orgxy, voxelRes);
        const vertex_t p7 = get3DPos(0, blockSize - 1, k + 1, orgx, voxelRes);

        const vertex_t *c0
            = (vertex_t *) irgba + COLOR_ID(blockSize - 1, blockSize - 1, k, blockSize);
        const vertex_t *c1 = (vertex_t *) yrgba + COLOR_ID(blockSize - 1, 0, k, blockSize);
        const vertex_t *c2 = (vertex_t *) xyrgba + COLOR_ID(0, 0, k, blockSize);
        const vertex_t *c3 = (vertex_t *) xrgba + COLOR_ID(0, blockSize - 1, k, blockSize);
        const vertex_t *c4
            = (vertex_t *) irgba + COLOR_ID(blockSize - 1, blockSize - 1, k + 1, blockSize);
        const vertex_t *c5 = (vertex_t *) yrgba + COLOR_ID(blockSize - 1, 0, k + 1, blockSize);
        const vertex_t *c6 = (vertex_t *) xyrgba + COLOR_ID(0, 0, k + 1, blockSize);
        const vertex_t *c7 = (vertex_t *) xrgba + COLOR_ID(0, blockSize - 1, k + 1, blockSize);

        const vertex_t *g0
            = (vertex_t *) igrad + GRAD_ID(blockSize - 1, blockSize - 1, k, blockSize);
        const vertex_t *g1 = (vertex_t *) ygrad + GRAD_ID(blockSize - 1, 0, k, blockSize);
        const vertex_t *g2 = (vertex_t *) xygrad + GRAD_ID(0, 0, k, blockSize);
        const vertex_t *g3 = (vertex_t *) xgrad + GRAD_ID(0, blockSize - 1, k, blockSize);
        const vertex_t *g4
            = (vertex_t *) igrad + GRAD_ID(blockSize - 1, blockSize - 1, k + 1, blockSize);
        const vertex_t *g5 = (vertex_t *) ygrad + GRAD_ID(blockSize - 1, 0, k + 1, blockSize);
        const vertex_t *g6 = (vertex_t *) xygrad + GRAD_ID(0, 0, k + 1, blockSize);
        const vertex_t *g7 = (vertex_t *) xgrad + GRAD_ID(0, blockSize - 1, k + 1, blockSize);

        vertex_t v[12];
        vertex_t c[12];
        vertex_t g[12];

        INTERPOLATE_POINTS(v, c, g);
        EXPORT_TRIANGLES(v, c, g);
    }
    return numTriangles;
}

static size_t extractXZEdgeMesh(
    const float *__restrict__ iTsdf,
    const float *__restrict__ xTsdf,
    const float *__restrict__ zTsdf,
    const float *__restrict__ xzTsdf,
    const float *__restrict__ irgba,
    const float *__restrict__ xrgba,
    const float *__restrict__ zrgba,
    const float *__restrict__ xzrgba,
    const float *__restrict__ igrad,
    const float *__restrict__ xgrad,
    const float *__restrict__ zgrad,
    const float *__restrict__ xzgrad,
    const float *__restrict__ iWeights,
    const float *__restrict__ xWeights,
    const float *__restrict__ zWeights,
    const float *__restrict__ xzWeights,
    float *triangles,
    float *__restrict__ colors,
    float *__restrict__ normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue)
{
    size_t numTriangles = 0;

    const vertex_t orgi = org;
    const vertex_t orgx = {orgi.x + voxelRes * blockSize, orgi.y, orgi.z};
    const vertex_t orgz = {orgi.x, orgi.y, orgi.z + voxelRes * blockSize};
    const vertex_t orgxz = {orgi.x + voxelRes * blockSize, orgi.y, orgi.z + voxelRes * blockSize};

    for(size_t j = 0; j < blockSize - 1; j++)
    {
        const float tsdf0 = iTsdf[GRID_OFFSET(blockSize - 1, j, blockSize - 1, blockSize)];
        const float tsdf1 = iTsdf[GRID_OFFSET(blockSize - 1, j + 1, blockSize - 1, blockSize)];
        const float tsdf2 = xTsdf[GRID_OFFSET(0, j + 1, blockSize - 1, blockSize)];
        const float tsdf3 = xTsdf[GRID_OFFSET(0, j, blockSize - 1, blockSize)];
        const float tsdf4 = zTsdf[GRID_OFFSET(blockSize - 1, j, 0, blockSize)];
        const float tsdf5 = zTsdf[GRID_OFFSET(blockSize - 1, j + 1, 0, blockSize)];
        const float tsdf6 = xzTsdf[GRID_OFFSET(0, j + 1, 0, blockSize)];
        const float tsdf7 = xzTsdf[GRID_OFFSET(0, j, 0, blockSize)];

        if(INVALID_CUBE(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7))
        {
            continue;
        }

        const float w0 = iWeights[GRID_OFFSET(blockSize - 1, j, blockSize - 1, blockSize)];
        const float w1 = iWeights[GRID_OFFSET(blockSize - 1, j + 1, blockSize - 1, blockSize)];
        const float w2 = xWeights[GRID_OFFSET(0, j + 1, blockSize - 1, blockSize)];
        const float w3 = xWeights[GRID_OFFSET(0, j, blockSize - 1, blockSize)];
        const float w4 = zWeights[GRID_OFFSET(blockSize - 1, j, 0, blockSize)];
        const float w5 = zWeights[GRID_OFFSET(blockSize - 1, j + 1, 0, blockSize)];
        const float w6 = xzWeights[GRID_OFFSET(0, j + 1, 0, blockSize)];
        const float w7 = xzWeights[GRID_OFFSET(0, j, 0, blockSize)];

        if(INVALID_WEIGHT(w0, w1, w2, w3, w4, w5, w6, w7))
        {
            continue;
        }

        const uint8_t cubeIndex
            = CUBE_INDEX(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7, isoValue);

        if(cubeIndex == 0)
        {
            continue;
        }

        const vertex_t p0 = get3DPos(blockSize - 1, j, blockSize - 1, orgi, voxelRes);
        const vertex_t p1 = get3DPos(blockSize - 1, j + 1, blockSize - 1, orgi, voxelRes);
        const vertex_t p2 = get3DPos(0, j + 1, blockSize - 1, orgx, voxelRes);
        const vertex_t p3 = get3DPos(0, j, blockSize - 1, orgx, voxelRes);
        const vertex_t p4 = get3DPos(blockSize - 1, j, 0, orgz, voxelRes);
        const vertex_t p5 = get3DPos(blockSize - 1, j + 1, 0, orgz, voxelRes);
        const vertex_t p6 = get3DPos(0, j + 1, 0, orgxz, voxelRes);
        const vertex_t p7 = get3DPos(0, j, 0, orgxz, voxelRes);

        const vertex_t *c0
            = (vertex_t *) irgba + COLOR_ID(blockSize - 1, j, blockSize - 1, blockSize);
        const vertex_t *c1
            = (vertex_t *) irgba + COLOR_ID(blockSize - 1, j + 1, blockSize - 1, blockSize);
        const vertex_t *c2 = (vertex_t *) xrgba + COLOR_ID(0, j + 1, blockSize - 1, blockSize);
        const vertex_t *c3 = (vertex_t *) xrgba + COLOR_ID(0, j, blockSize - 1, blockSize);
        const vertex_t *c4 = (vertex_t *) zrgba + COLOR_ID(blockSize - 1, j, 0, blockSize);
        const vertex_t *c5 = (vertex_t *) zrgba + COLOR_ID(blockSize - 1, j + 1, 0, blockSize);
        const vertex_t *c6 = (vertex_t *) xzrgba + COLOR_ID(0, j + 1, 0, blockSize);
        const vertex_t *c7 = (vertex_t *) xzrgba + COLOR_ID(0, j, 0, blockSize);

        const vertex_t *g0
            = (vertex_t *) igrad + GRAD_ID(blockSize - 1, j, blockSize - 1, blockSize);
        const vertex_t *g1
            = (vertex_t *) igrad + GRAD_ID(blockSize - 1, j + 1, blockSize - 1, blockSize);
        const vertex_t *g2 = (vertex_t *) xgrad + GRAD_ID(0, j + 1, blockSize - 1, blockSize);
        const vertex_t *g3 = (vertex_t *) xgrad + GRAD_ID(0, j, blockSize - 1, blockSize);
        const vertex_t *g4 = (vertex_t *) zgrad + GRAD_ID(blockSize - 1, j, 0, blockSize);
        const vertex_t *g5 = (vertex_t *) zgrad + GRAD_ID(blockSize - 1, j + 1, 0, blockSize);
        const vertex_t *g6 = (vertex_t *) xzgrad + GRAD_ID(0, j + 1, 0, blockSize);
        const vertex_t *g7 = (vertex_t *) xzgrad + GRAD_ID(0, j, 0, blockSize);

        vertex_t v[12];
        vertex_t c[12];
        vertex_t g[12];

        INTERPOLATE_POINTS(v, c, g);
        EXPORT_TRIANGLES(v, c, g);
    }
    return numTriangles;
}

static size_t extractYZEdgeMesh(
    const float *__restrict__ iTsdf,
    const float *__restrict__ yTsdf,
    const float *__restrict__ zTsdf,
    const float *__restrict__ yzTsdf,
    const float *__restrict__ irgba,
    const float *__restrict__ yrgba,
    const float *__restrict__ zrgba,
    const float *__restrict__ yzrgba,
    const float *__restrict__ igrad,
    const float *__restrict__ ygrad,
    const float *__restrict__ zgrad,
    const float *__restrict__ yzgrad,
    const float *__restrict__ iWeights,
    const float *__restrict__ yWeights,
    const float *__restrict__ zWeights,
    const float *__restrict__ yzWeights,
    float *__restrict__ triangles,
    float *__restrict__ colors,
    float *__restrict__ normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue)
{
    size_t numTriangles = 0;

    const vertex_t orgi = org;
    const vertex_t orgy = {orgi.x, orgi.y + voxelRes * blockSize, orgi.z};
    const vertex_t orgz = {orgi.x, orgi.y, orgi.z + voxelRes * blockSize};
    const vertex_t orgyz = {orgi.x, orgi.y + voxelRes * blockSize, orgi.z + voxelRes * blockSize};

    for(size_t i = 0; i < blockSize - 1; i++)
    {
        const float tsdf0 = iTsdf[GRID_OFFSET(i, blockSize - 1, blockSize - 1, blockSize)];
        const float tsdf1 = yTsdf[GRID_OFFSET(i, 0, blockSize - 1, blockSize)];
        const float tsdf2 = yTsdf[GRID_OFFSET(i + 1, 0, blockSize - 1, blockSize)];
        const float tsdf3 = iTsdf[GRID_OFFSET(i + 1, blockSize - 1, blockSize - 1, blockSize)];
        const float tsdf4 = zTsdf[GRID_OFFSET(i, blockSize - 1, 0, blockSize)];
        const float tsdf5 = yzTsdf[GRID_OFFSET(i, 0, 0, blockSize)];
        const float tsdf6 = yzTsdf[GRID_OFFSET(i + 1, 0, 0, blockSize)];
        const float tsdf7 = zTsdf[GRID_OFFSET(i + 1, blockSize - 1, 0, blockSize)];

        if(INVALID_CUBE(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7))
        {
            continue;
        }

        const float w0 = iWeights[GRID_OFFSET(i, blockSize - 1, blockSize - 1, blockSize)];
        const float w1 = yWeights[GRID_OFFSET(i, 0, blockSize - 1, blockSize)];
        const float w2 = yWeights[GRID_OFFSET(i + 1, 0, blockSize - 1, blockSize)];
        const float w3 = iWeights[GRID_OFFSET(i + 1, blockSize - 1, blockSize - 1, blockSize)];
        const float w4 = zWeights[GRID_OFFSET(i, blockSize - 1, 0, blockSize)];
        const float w5 = yzWeights[GRID_OFFSET(i, 0, 0, blockSize)];
        const float w6 = yzWeights[GRID_OFFSET(i + 1, 0, 0, blockSize)];
        const float w7 = zWeights[GRID_OFFSET(i + 1, blockSize - 1, 0, blockSize)];

        if(INVALID_WEIGHT(w0, w1, w2, w3, w4, w5, w6, w7))
        {
            continue;
        }

        const uint8_t cubeIndex
            = CUBE_INDEX(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7, isoValue);

        if(cubeIndex == 0)
        {
            continue;
        }

        const vertex_t p0 = get3DPos(i, blockSize - 1, blockSize - 1, orgi, voxelRes);
        const vertex_t p1 = get3DPos(i, 0, blockSize - 1, orgy, voxelRes);
        const vertex_t p2 = get3DPos(i + 1, 0, blockSize - 1, orgy, voxelRes);
        const vertex_t p3 = get3DPos(i + 1, blockSize - 1, blockSize - 1, orgi, voxelRes);
        const vertex_t p4 = get3DPos(i, blockSize - 1, 0, orgz, voxelRes);
        const vertex_t p5 = get3DPos(i, 0, 0, orgyz, voxelRes);
        const vertex_t p6 = get3DPos(i + 1, 0, 0, orgyz, voxelRes);
        const vertex_t p7 = get3DPos(i + 1, blockSize - 1, 0, orgz, voxelRes);

        const vertex_t *c0
            = (vertex_t *) irgba + COLOR_ID(i, blockSize - 1, blockSize - 1, blockSize);
        const vertex_t *c1 = (vertex_t *) yrgba + COLOR_ID(i, 0, blockSize - 1, blockSize);
        const vertex_t *c2 = (vertex_t *) yrgba + COLOR_ID(i + 1, 0, blockSize - 1, blockSize);
        const vertex_t *c3
            = (vertex_t *) irgba + COLOR_ID(i + 1, blockSize - 1, blockSize - 1, blockSize);
        const vertex_t *c4 = (vertex_t *) zrgba + COLOR_ID(i, blockSize - 1, 0, blockSize);
        const vertex_t *c5 = (vertex_t *) yzrgba + COLOR_ID(i, 0, 0, blockSize);
        const vertex_t *c6 = (vertex_t *) yzrgba + COLOR_ID(i + 1, 0, 0, blockSize);
        const vertex_t *c7 = (vertex_t *) zrgba + COLOR_ID(i + 1, blockSize - 1, 0, blockSize);

        const vertex_t *g0
            = (vertex_t *) igrad + GRAD_ID(i, blockSize - 1, blockSize - 1, blockSize);
        const vertex_t *g1 = (vertex_t *) ygrad + GRAD_ID(i, 0, blockSize - 1, blockSize);
        const vertex_t *g2 = (vertex_t *) ygrad + GRAD_ID(i + 1, 0, blockSize - 1, blockSize);
        const vertex_t *g3
            = (vertex_t *) igrad + GRAD_ID(i + 1, blockSize - 1, blockSize - 1, blockSize);
        const vertex_t *g4 = (vertex_t *) zgrad + GRAD_ID(i, blockSize - 1, 0, blockSize);
        const vertex_t *g5 = (vertex_t *) yzgrad + GRAD_ID(i, 0, 0, blockSize);
        const vertex_t *g6 = (vertex_t *) yzgrad + GRAD_ID(i + 1, 0, 0, blockSize);
        const vertex_t *g7 = (vertex_t *) zgrad + GRAD_ID(i + 1, blockSize - 1, 0, blockSize);

        vertex_t v[12];
        vertex_t c[12];
        vertex_t g[12];

        INTERPOLATE_POINTS(v, c, g);
        EXPORT_TRIANGLES(v, c, g);
    }
    return numTriangles;
}

static size_t extractXYZCornerMesh(
    const float *__restrict__ iTsdf,
    const float *__restrict__ xTsdf,
    const float *__restrict__ yTsdf,
    const float *__restrict__ zTsdf,
    const float *__restrict__ xyTsdf,
    const float *__restrict__ xzTsdf,
    const float *__restrict__ yzTsdf,
    const float *__restrict__ xyzTsdf,
    const float *__restrict__ irgba,
    const float *__restrict__ xrgba,
    const float *__restrict__ yrgba,
    const float *__restrict__ zrgba,
    const float *__restrict__ xyrgba,
    const float *__restrict__ xzrgba,
    const float *__restrict__ yzrgba,
    const float *__restrict__ xyzrgba,
    const float *__restrict__ igrad,
    const float *__restrict__ xgrad,
    const float *__restrict__ ygrad,
    const float *__restrict__ zgrad,
    const float *__restrict__ xygrad,
    const float *__restrict__ xzgrad,
    const float *__restrict__ yzgrad,
    const float *__restrict__ xyzgrad,
    const float *__restrict__ iWeights,
    const float *__restrict__ xWeights,
    const float *__restrict__ yWeights,
    const float *__restrict__ zWeights,
    const float *__restrict__ xyWeights,
    const float *__restrict__ xzWeights,
    const float *__restrict__ yzWeights,
    const float *__restrict__ xyzWeights,
    float *__restrict__ triangles,
    float *__restrict__ colors,
    float *__restrict__ normals,
    const float voxelRes,
    const vertex_t org,
    const size_t blockSize,
    const float wThr,
    const float isoValue)

{
    size_t numTriangles = 0;

    const vertex_t orgi = org;
    const vertex_t orgx = {orgi.x + voxelRes * blockSize, orgi.y, orgi.z};
    const vertex_t orgy = {orgi.x, orgi.y + voxelRes * blockSize, orgi.z};
    const vertex_t orgz = {orgi.x, orgi.y, orgi.z + voxelRes * blockSize};
    const vertex_t orgxy = {orgi.x + voxelRes * blockSize, orgi.y + voxelRes * blockSize, orgi.z};
    const vertex_t orgxz = {orgi.x + voxelRes * blockSize, orgi.y, orgi.z + voxelRes * blockSize};
    const vertex_t orgyz = {orgi.x, orgi.y + voxelRes * blockSize, orgi.z + voxelRes * blockSize};
    const vertex_t orgxyz
        = {orgi.x + voxelRes * blockSize,
           orgi.y + voxelRes * blockSize,
           orgi.z + voxelRes * blockSize};

    const float tsdf0 = iTsdf[GRID_OFFSET(blockSize - 1, blockSize - 1, blockSize - 1, blockSize)];
    const float tsdf1 = yTsdf[GRID_OFFSET(blockSize - 1, 0, blockSize - 1, blockSize)];
    const float tsdf2 = xyTsdf[GRID_OFFSET(0, 0, blockSize - 1, blockSize)];
    const float tsdf3 = xTsdf[GRID_OFFSET(0, blockSize - 1, blockSize - 1, blockSize)];
    const float tsdf4 = zTsdf[GRID_OFFSET(blockSize - 1, blockSize - 1, 0, blockSize)];
    const float tsdf5 = yzTsdf[GRID_OFFSET(blockSize - 1, 0, 0, blockSize)];
    const float tsdf6 = xyzTsdf[GRID_OFFSET(0, 0, 0, blockSize)];
    const float tsdf7 = xzTsdf[GRID_OFFSET(0, blockSize - 1, 0, blockSize)];

    if(INVALID_CUBE(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7))
    {
        return 0;
    }

    const float w0 = iWeights[GRID_OFFSET(blockSize - 1, blockSize - 1, blockSize - 1, blockSize)];
    const float w1 = yWeights[GRID_OFFSET(blockSize - 1, 0, blockSize - 1, blockSize)];
    const float w2 = xyWeights[GRID_OFFSET(0, 0, blockSize - 1, blockSize)];
    const float w3 = xWeights[GRID_OFFSET(0, blockSize - 1, blockSize - 1, blockSize)];
    const float w4 = zWeights[GRID_OFFSET(blockSize - 1, blockSize - 1, 0, blockSize)];
    const float w5 = yzWeights[GRID_OFFSET(blockSize - 1, 0, 0, blockSize)];
    const float w6 = xyzWeights[GRID_OFFSET(0, 0, 0, blockSize)];
    const float w7 = xzWeights[GRID_OFFSET(0, blockSize - 1, 0, blockSize)];

    if(INVALID_WEIGHT(w0, w1, w2, w3, w4, w5, w6, w7))
    {
        return 0;
    }

    const uint8_t cubeIndex
        = CUBE_INDEX(tsdf0, tsdf1, tsdf2, tsdf3, tsdf4, tsdf5, tsdf6, tsdf7, isoValue);

    if(cubeIndex == 0)
    {
        return 0;
    }

    const vertex_t p0 = get3DPos(blockSize - 1, blockSize - 1, blockSize - 1, orgi, voxelRes);
    const vertex_t p1 = get3DPos(blockSize - 1, 0, blockSize - 1, orgy, voxelRes);
    const vertex_t p2 = get3DPos(0, 0, blockSize - 1, orgxy, voxelRes);
    const vertex_t p3 = get3DPos(0, blockSize - 1, blockSize - 1, orgx, voxelRes);
    const vertex_t p4 = get3DPos(blockSize - 1, blockSize - 1, 0, orgz, voxelRes);
    const vertex_t p5 = get3DPos(blockSize - 1, 0, 0, orgyz, voxelRes);
    const vertex_t p6 = get3DPos(0, 0, 0, orgxyz, voxelRes);
    const vertex_t p7 = get3DPos(0, blockSize - 1, 0, orgxz, voxelRes);

    const vertex_t *c0
        = (vertex_t *) irgba + COLOR_ID(blockSize - 1, blockSize - 1, blockSize - 1, blockSize);
    const vertex_t *c1 = (vertex_t *) yrgba + COLOR_ID(blockSize - 1, 0, blockSize - 1, blockSize);
    const vertex_t *c2 = (vertex_t *) xyrgba + COLOR_ID(0, 0, blockSize - 1, blockSize);
    const vertex_t *c3 = (vertex_t *) xrgba + COLOR_ID(0, blockSize - 1, blockSize - 1, blockSize);
    const vertex_t *c4 = (vertex_t *) zrgba + COLOR_ID(blockSize - 1, blockSize - 1, 0, blockSize);
    const vertex_t *c5 = (vertex_t *) yzrgba + COLOR_ID(blockSize - 1, 0, 0, blockSize);
    const vertex_t *c6 = (vertex_t *) xyzrgba + COLOR_ID(0, 0, 0, blockSize);
    const vertex_t *c7 = (vertex_t *) xzrgba + COLOR_ID(0, blockSize - 1, 0, blockSize);

    const vertex_t *g0
        = (vertex_t *) igrad + GRAD_ID(blockSize - 1, blockSize - 1, blockSize - 1, blockSize);
    const vertex_t *g1 = (vertex_t *) ygrad + GRAD_ID(blockSize - 1, 0, blockSize - 1, blockSize);
    const vertex_t *g2 = (vertex_t *) xygrad + GRAD_ID(0, 0, blockSize - 1, blockSize);
    const vertex_t *g3 = (vertex_t *) xgrad + GRAD_ID(0, blockSize - 1, blockSize - 1, blockSize);
    const vertex_t *g4 = (vertex_t *) zgrad + GRAD_ID(blockSize - 1, blockSize - 1, 0, blockSize);
    const vertex_t *g5 = (vertex_t *) yzgrad + GRAD_ID(blockSize - 1, 0, 0, blockSize);
    const vertex_t *g6 = (vertex_t *) xyzgrad + GRAD_ID(0, 0, 0, blockSize);
    const vertex_t *g7 = (vertex_t *) xzgrad + GRAD_ID(0, blockSize - 1, 0, blockSize);

    vertex_t v[12];
    vertex_t c[12];
    vertex_t g[12];

    INTERPOLATE_POINTS(v, c, g);
    EXPORT_TRIANGLES(v, c, g);

    return numTriangles;
}
