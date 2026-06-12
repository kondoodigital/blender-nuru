//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_CPU_PATCH_TABLE_H
#define OPENSUBDIV3_OSD_CPU_PATCH_TABLE_H

#include "../version.h"

#include <vector>
#include "../far/patchDescriptor.h"
#include "../osd/nonCopyable.h"
#include "../osd/types.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far{
    class PatchTable;
};

namespace Osd {

/// \brief Cpu patch table
///
/// XXX: We can use just Far::PatchTable for typical CpuEval use cases.
///
///  Currently this class exists because of the template resolution
///  for the CpuEvaluator's generic interface functions
///    (glEvalLimit example uses), and
///  device-specific patch tables such as GLPatchTables internally use
///  as a staging buffer to splice patcharray and interleave sharpnesses.
///
///  Ideally Far::PatchTables should have the same data representation
///  and accessors so that we don't have to copy data unnecessarily.
///
class CpuPatchTable {
public:
    static CpuPatchTable *Create(const Far::PatchTable *patchTable,
                                 void *deviceContext = NULL) {
        (void)deviceContext;  // unused
        return new CpuPatchTable(patchTable);
    }

    explicit CpuPatchTable(const Far::PatchTable *patchTable);
    ~CpuPatchTable() {}

    const PatchArray *GetPatchArrayBuffer() const {
        return &_patchArrays[0];
    }
    const int *GetPatchIndexBuffer() const {
        return &_indexBuffer[0];
    }
    const PatchParam *GetPatchParamBuffer() const {
        return &_patchParamBuffer[0];
    }

    size_t GetNumPatchArrays() const {
        return _patchArrays.size();
    }
    size_t GetPatchIndexSize() const {
        return _indexBuffer.size();
    }
    size_t GetPatchParamSize() const {
        return _patchParamBuffer.size();
    }

    const PatchArray *GetVaryingPatchArrayBuffer() const {
        if (_varyingPatchArrays.empty()) {
            return NULL;
        }
        return &_varyingPatchArrays[0];
    }
    const int *GetVaryingPatchIndexBuffer() const {
        if (_varyingIndexBuffer.empty()) {
            return NULL;
        }
        return &_varyingIndexBuffer[0];
    }
    size_t GetVaryingPatchIndexSize() const {
        return _varyingIndexBuffer.size();
    }

    int GetNumFVarChannels() const {
        return (int)_fvarPatchArrays.size();
    }
    const PatchArray *GetFVarPatchArrayBuffer(int fvarChannel = 0) const {
        return &_fvarPatchArrays[fvarChannel][0];
    }
    const int *GetFVarPatchIndexBuffer(int fvarChannel = 0) const {
        return &_fvarIndexBuffers[fvarChannel][0];
    }
    size_t GetFVarPatchIndexSize(int fvarChannel = 0) const {
        return _fvarIndexBuffers[fvarChannel].size();
    }
    const PatchParam *GetFVarPatchParamBuffer(int fvarChannel= 0) const {
        return &_fvarParamBuffers[fvarChannel][0];
    }
    size_t GetFVarPatchParamSize(int fvarChannel = 0) const {
        return _fvarParamBuffers[fvarChannel].size();
    }

protected:
    PatchArrayVector _patchArrays;
    std::vector<int> _indexBuffer;
    PatchParamVector _patchParamBuffer;

    PatchArrayVector _varyingPatchArrays;
    std::vector<int> _varyingIndexBuffer;

    std::vector< PatchArrayVector > _fvarPatchArrays;
    std::vector< std::vector<int> > _fvarIndexBuffers;
    std::vector< PatchParamVector > _fvarParamBuffers;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_CPU_PATCH_TABLE_H
