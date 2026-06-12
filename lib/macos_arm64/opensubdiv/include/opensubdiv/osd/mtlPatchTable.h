//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_MTL_PATCH_TABLE_H
#define OPENSUBDIV3_OSD_MTL_PATCH_TABLE_H

#include "../version.h"
#include "../far/patchDescriptor.h"
#include "../osd/nonCopyable.h"
#include "../osd/types.h"
#include "../osd/mtlCommon.h"

@protocol MTLDevice;
@protocol MTLBuffer;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
    class PatchTable;
};

namespace Osd {

class MTLPatchTable : private NonCopyable<MTLPatchTable> {
public:
    using VertexBufferBinding = id<MTLBuffer>;  // buffer binding type

    MTLPatchTable();
    ~MTLPatchTable();

    template<typename DEVICE_CONTEXT>
    static MTLPatchTable *Create(Far::PatchTable const *farPatchTable, DEVICE_CONTEXT context)
    {
        return Create(farPatchTable, context);
    }

    static MTLPatchTable *Create(Far::PatchTable const *farPatchTable, MTLContext* context);

    PatchArrayVector const &GetPatchArrays() const { return _patchArrays; }
    id<MTLBuffer> GetPatchIndexBuffer() const { return _indexBuffer; }
    id<MTLBuffer> GetPatchParamBuffer() const { return _patchParamBuffer; }

    PatchArrayVector const &GetVaryingPatchArrays() const { return _varyingPatchArrays; }
    id<MTLBuffer> GetVaryingPatchIndexBuffer() const { return _varyingPatchIndexBuffer; }

    int GetNumFVarChannels() const { return (int)_fvarPatchArrays.size(); }
    PatchArrayVector const &GetFVarPatchArrays(int fvarChannel = 0) const { return _fvarPatchArrays[fvarChannel]; }
    id<MTLBuffer> GetFVarPatchIndexBuffer(int fvarChannel = 0) const { return _fvarIndexBuffers[fvarChannel]; }
    id<MTLBuffer> GetFVarPatchParamBuffer(int fvarChannel = 0) const { return _fvarParamBuffers[fvarChannel]; }

protected:
    bool allocate(Far::PatchTable const *farPatchTable, MTLContext* context);

    PatchArrayVector _patchArrays;

    id<MTLBuffer> _indexBuffer;
    id<MTLBuffer> _patchParamBuffer;

    PatchArrayVector _varyingPatchArrays;

    id<MTLBuffer> _varyingPatchIndexBuffer;

    std::vector<PatchArrayVector> _fvarPatchArrays;
    std::vector<id<MTLBuffer>> _fvarIndexBuffers;
    std::vector<id<MTLBuffer>> _fvarParamBuffers;
};

} // end namespace Osd

} //end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} //end namespace OpenSubdiv

#endif //end OPENSUBDIV3_OSD_MTL_PATCH_TABLE_H
