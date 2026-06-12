//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_GL_LEGACY_GREGORY_PATCH_TABLE_H
#define OPENSUBDIV3_OSD_GL_LEGACY_GREGORY_PATCH_TABLE_H

#include "../version.h"

#include "../far/patchTable.h"
#include "../osd/nonCopyable.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class GLLegacyGregoryPatchTable
    : private NonCopyable<GLLegacyGregoryPatchTable> {
public:
    using ID = unsigned int;    // GLuint resource ID

    ~GLLegacyGregoryPatchTable();

    static GLLegacyGregoryPatchTable *Create(Far::PatchTable const *patchTable);

    void UpdateVertexBuffer(ID vbo);

    ID GetVertexTextureBuffer() const {
        return _vertexTextureBuffer;
    }

    ID GetVertexValenceTextureBuffer() const {
        return _vertexValenceTextureBuffer;
    }

    ID GetQuadOffsetsTextureBuffer() const {
        return _quadOffsetsTextureBuffer;
    }

    ID GetQuadOffsetsBase(Far::PatchDescriptor::Type type) {
        if (type == Far::PatchDescriptor::GREGORY_BOUNDARY) {
            return _quadOffsetsBase[1];
        }
        return _quadOffsetsBase[0];
    }

protected:
    GLLegacyGregoryPatchTable();

private:
    ID _vertexTextureBuffer;
    ID _vertexValenceTextureBuffer;
    ID _quadOffsetsTextureBuffer;
    ID _quadOffsetsBase[2];       // gregory, boundaryGregory
};



}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_GL_LEGACY_GREGORY_PATCH_TABLE_H
