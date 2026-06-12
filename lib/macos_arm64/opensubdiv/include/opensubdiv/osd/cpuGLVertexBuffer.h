//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_CPU_GL_VERTEX_BUFFER_H
#define OPENSUBDIV3_OSD_CPU_GL_VERTEX_BUFFER_H

#include "../version.h"

#include <cstddef>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

///
/// \brief Concrete vertex buffer class for cpu subdivision and OpenGL drawing.
///
/// CpuGLVertexBuffer implements CpuVertexBufferInterface and
/// GLVertexBufferInterface.
///
/// The buffer interop between Cpu and GL is handled automatically when a
/// client calls BindCpuBuffer and BindVBO methods.
///
class CpuGLVertexBuffer {
public:
    using ID = unsigned int;    // GLuint resource ID

    /// Creator. Returns NULL if error.
    static CpuGLVertexBuffer * Create(int numElements, int numVertices,
                                      void *deviceContext = NULL);

    /// Destructor.
    ~CpuGLVertexBuffer();

    /// This method is meant to be used in client code in order to provide
    /// coarse vertices data to Osd.
    void UpdateData(const float *src, int startVertex, int numVertices,
                    void *deviceContext = NULL);

    /// Returns how many elements defined in this vertex buffer.
    int GetNumElements() const;

    /// Returns how many vertices allocated in this vertex buffer.
    int GetNumVertices() const;

    /// Returns cpu memory. GL buffer will be mapped to cpu address
    /// if necessary.
    float * BindCpuBuffer();

    /// Returns the name of GL buffer object. If the buffer is mapped
    /// to cpu address, it will be unmapped back to GL.
    ID BindVBO(void *deviceContext = NULL);

protected:
    /// Constructor.
    CpuGLVertexBuffer(int numElements, int numVertices);

    /// Allocates VBO for this buffer. Returns true if success.
    bool allocate();

private:
    int _numElements;
    int _numVertices;
    ID _vbo;
    float *_cpuBuffer;
    bool _dataDirty;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_CPU_GL_VERTEX_BUFFER_H
