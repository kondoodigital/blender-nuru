//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_MTL_VERTEX_BUFFER_H
#define OPENSUBDIV3_OSD_MTL_VERTEX_BUFFER_H

#include "../version.h"
#include "../osd/mtlCommon.h"

@protocol MTLDevice;
@protocol MTLBuffer;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class CPUMTLVertexBuffer {
public:
    static CPUMTLVertexBuffer* Create(int numElements, int numVertices, MTLContext* context);

    void UpdateData(const float* src, int startVertex, int numVertices, MTLContext* context);

    int GetNumElements() const
    {
        return _numElements;
    }

    int GetNumVertices() const
    {
        return _numVertices;
    }

    float* BindCpuBuffer();
    id<MTLBuffer> BindMTLBuffer(MTLContext* context);

    id<MTLBuffer> BindVBO(MTLContext* context)
    {
        return BindMTLBuffer(context);
    }

protected:

    CPUMTLVertexBuffer(int numElements, int numVertices);

    bool allocate(MTLContext* context);

private:
    int _numElements;
    int _numVertices;
    id<MTLBuffer> _buffer;
    bool _dirty;
};

} //end namespace Osd

} //end namespace OPENSUBDIV_VERSION
    using namespace OPENSUBDIV_VERSION;

} //end namespace OpenSubdiv

#endif // OPENSUBDIV3_OSD_MTL_VERTEX_BUFFER_H
