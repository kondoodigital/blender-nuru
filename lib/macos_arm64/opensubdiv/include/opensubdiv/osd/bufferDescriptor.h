//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_BUFFER_DESCRIPTOR_H
#define OPENSUBDIV3_OSD_BUFFER_DESCRIPTOR_H

#include "../version.h"
#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief BufferDescriptor is a struct which describes buffer elements in
///        interleaved data buffers. Almost all Osd Evaluator APIs take
///        BufferDescriptors along with device-specific buffer objects.
///
///        The offset of BufferDescriptor can also be used to express a
///        batching offset if the data buffer is combined across multiple
///        objects together.
///
///        * Note that each element has the same data type (float)
///

//  example:
//       n
//  -----+----------------------------------------+-------------------------
//       |               vertex  0                |
//  -----+----------------------------------------+-------------------------
//       |  X  Y  Z  R  G  B  A Xu Yu Zu Xv Yv Zv |
//  -----+----------------------------------------+-------------------------
//       <------------- stride = 13 -------------->
//
//     - XYZ      (offset = n+0,  length = 3, stride = 13)
//     - RGBA     (offset = n+3,  length = 4, stride = 13)
//     - uTangent (offset = n+7,  length = 3, stride = 13)
//     - vTangent (offset = n+10, length = 3, stride = 13)
//
struct BufferDescriptor {

    /// Default Constructor
    BufferDescriptor() : offset(0), length(0), stride(0) { }

    /// Constructor
    BufferDescriptor(int o, int l, int s) : offset(o), length(l), stride(s) { }

    /// Returns the relative offset within a stride
    int GetLocalOffset() const {
        return stride > 0 ? offset % stride : 0;
    }

    /// True if the descriptor values are internally consistent
    bool IsValid() const {
        return ((length > 0) &&
                (length <= stride - GetLocalOffset()));
    }

    /// Resets the descriptor to default
    void Reset() {
        offset = length = stride = 0;
    }

    /// True if the descriptors are identical
    bool operator == (BufferDescriptor const &other) const {
        return (offset == other.offset &&
                length == other.length &&
                stride == other.stride);
    }

    /// True if the descriptors are not identical
    bool operator != (BufferDescriptor const &other) const {
        return !(this->operator==(other));
    }

    /// offset to desired element data
    int offset;
    /// number or length of the data
    int length;
    /// stride to the next element
    int stride;
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_BUFFER_DESCRIPTOR_H
