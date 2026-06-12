//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_CPU_VERTEX_BUFFER_H
#define OPENSUBDIV3_OSD_CPU_VERTEX_BUFFER_H

#include "../version.h"

#include <cstddef>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief Concrete vertex buffer class for CPU subdivision.
///
/// CpuVertexBuffer implements the VertexBufferInterface. An instance
/// of this buffer class can be passed to CpuEvaluator
///
class CpuVertexBuffer {
public:
    /// Creator. Returns NULL if error.
    static CpuVertexBuffer * Create(int numElements, int numVertices,
                                    void *deviceContext = NULL);

    /// Destructor.
    ~CpuVertexBuffer();

    /// This method is meant to be used in client code in order to provide
    /// coarse vertices data to Osd.
    void UpdateData(const float *src, int startVertex, int numVertices,
                    void *deviceContext = NULL);

    /// Returns how many elements defined in this vertex buffer.
    int GetNumElements() const;

    /// Returns how many vertices allocated in this vertex buffer.
    int GetNumVertices() const;

    /// Returns the address of CPU buffer
    float * BindCpuBuffer();

protected:
    /// Constructor.
    CpuVertexBuffer(int numElements, int numVertices);

private:
    int _numElements;
    int _numVertices;
    float *_cpuBuffer;
};


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_CPU_VERTEX_BUFFER_H
