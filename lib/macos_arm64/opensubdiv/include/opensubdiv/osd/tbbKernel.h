//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_TBB_KERNEL_H
#define OPENSUBDIV3_OSD_TBB_KERNEL_H

#include "../version.h"
#include "../far/patchDescriptor.h"
#include "../far/patchParam.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

struct PatchArray;
struct PatchCoord;
struct PatchParam;
struct BufferDescriptor;

void
TbbEvalStencils(float const * src, BufferDescriptor const &srcDesc,
                float * dst,       BufferDescriptor const &dstDesc,
                int const * sizes,
                int const * offsets,
                int const * indices,
                float const * weights,
                int start, int end);

void
TbbEvalStencils(float const * src, BufferDescriptor const &srcDesc,
                float * dst,       BufferDescriptor const &dstDesc,
                float * dstDu,     BufferDescriptor const &dstDuDesc,
                float * dstDv,     BufferDescriptor const &dstDvDesc,
                int const * sizes,
                int const * offsets,
                int const * indices,
                float const * weights,
                float const * duWeights,
                float const * dvWeights,
                int start, int end);

void
TbbEvalStencils(float const * src, BufferDescriptor const &srcDesc,
                float * dst,       BufferDescriptor const &dstDesc,
                float * dstDu,     BufferDescriptor const &dstDuDesc,
                float * dstDv,     BufferDescriptor const &dstDvDesc,
                float * dstDuu,    BufferDescriptor const &dstDuuDesc,
                float * dstDuv,    BufferDescriptor const &dstDuvDesc,
                float * dstDvv,    BufferDescriptor const &dstDvvDesc,
                int const * sizes,
                int const * offsets,
                int const * indices,
                float const * weights,
                float const * duWeights,
                float const * dvWeights,
                float const * duuWeights,
                float const * duvWeights,
                float const * dvvWeights,
                int start, int end);

void
TbbEvalPatches(float const *src, BufferDescriptor const &srcDesc,
               float *dst,       BufferDescriptor const &dstDesc,
               float *dstDu,     BufferDescriptor const &dstDuDesc,
               float *dstDv,     BufferDescriptor const &dstDvDesc,
               int numPatchCoords,
               const PatchCoord *patchCoords,
               const PatchArray *patchArrayBuffer,
               const int *patchIndexBuffer,
               const PatchParam *patchParamBuffer);

void
TbbEvalPatches(float const *src, BufferDescriptor const &srcDesc,
               float *dst,       BufferDescriptor const &dstDesc,
               float *dstDu,     BufferDescriptor const &dstDuDesc,
               float *dstDv,     BufferDescriptor const &dstDvDesc,
               float *dstDuu,    BufferDescriptor const &dstDuuDesc,
               float *dstDuv,    BufferDescriptor const &dstDuvDesc,
               float *dstDvv,    BufferDescriptor const &dstDvvDesc,
               int numPatchCoords,
               const PatchCoord *patchCoords,
               const PatchArray *patchArrayBuffer,
               const int *patchIndexBuffer,
               const PatchParam *patchParamBuffer);

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_TBB_KERNEL_H
