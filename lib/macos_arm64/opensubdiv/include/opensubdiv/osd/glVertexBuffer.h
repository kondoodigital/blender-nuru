//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_GL_VERTEX_BUFFER_H
#define OPENSUBDIV3_OSD_GL_VERTEX_BUFFER_H

#include "../version.h"

#include <cstddef>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

///
/// \brief Concrete vertex buffer class for GLSL subdivision and OpenGL drawing.
///
/// GLVertexBuffer implements GLVertexBufferInterface. An instance
/// of this buffer class can be passed to OsdGLComputeEvaluator.
///
class GLVertexBuffer {
public:
    using ID = unsigned int;    // GLuint resource ID

    /// Creator. Returns NULL if error.
    static GLVertexBuffer * Create(int numElements, int numVertices,
                                   void *deviceContext = NULL);

    /// Destructor.
    ~GLVertexBuffer();

    /// This method is meant to be used in client code in order to provide coarse
    /// vertices data to Osd.
    void UpdateData(const float *src, int startVertex, int numVertices,
                    void *deviceContext = NULL);

    /// Returns how many elements defined in this vertex buffer.
    int GetNumElements() const;

    /// Returns how many vertices allocated in this vertex buffer.
    int GetNumVertices() const;

    /// Returns the GL buffer object.
    ID BindVBO(void *deviceContext = NULL);

protected:
    /// Constructor.
    GLVertexBuffer(int numElements, int numVertices);

    /// Allocates VBO for this buffer.
    /// Returns true if success.
    bool allocate();

private:
    int _numElements;
    int _numVertices;
    ID _vbo;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_GL_VERTEX_BUFFER_H
