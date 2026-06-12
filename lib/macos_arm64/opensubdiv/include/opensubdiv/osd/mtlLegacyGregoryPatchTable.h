//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_MTL_LEGACY_GREGORY_PATCH_TABLE_H
#define OPENSUBDIV3_OSD_MTL_LEGACY_GREGORY_PATCH_TABLE_H

#include "../version.h"
#include "../far/patchTable.h"
#include "../osd/nonCopyable.h"
#include "../osd/mtlCommon.h"

@protocol MTLDevice;
@protocol MTLBuffer;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class MTLLegacyGregoryPatchTable
: private NonCopyable<MTLLegacyGregoryPatchTable>
{
public:
    ~MTLLegacyGregoryPatchTable();

    template<typename DEVICE_CONTEXT>
    static MTLLegacyGregoryPatchTable* Create(Far::PatchTable const* farPatchTable, DEVICE_CONTEXT context) {
        return Create(farPatchTable, context);
    }

    static MTLLegacyGregoryPatchTable* Create(Far::PatchTable const* farPatchTable, MTLContext* context);

    void UpdateVertexBuffer(id<MTLBuffer> vbo, int numVertices, int numVertexElements, MTLContext* context);

    id<MTLBuffer> GetVertexBuffer() const
    {
        return _vertexBuffer;
    }

    id<MTLBuffer> GetVertexValenceBuffer() const
    {
        return _vertexValenceBuffer;
    }

    id<MTLBuffer> GetQuadOffsetsBuffer() const
    {
        return _quadOffsetsBuffer;
    }

    int GetQuadOffsetsBase(Far::PatchDescriptor::Type type)
    {
        if(type == Far::PatchDescriptor::GREGORY_BOUNDARY)
            return _quadOffsetsBase[1];
        return _quadOffsetsBase[0];
    }

private:
    id<MTLBuffer> _vertexBuffer;
    id<MTLBuffer> _vertexValenceBuffer;
    id<MTLBuffer> _quadOffsetsBuffer;
    int _quadOffsetsBase[2];
};

} //end namespace Osd

} //end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} //end namespace OpenSuddiv

#endif // OPENSUBDIV3_OSD_MTL_LEGACY_GREGORY_PATCH_TABLE_H
