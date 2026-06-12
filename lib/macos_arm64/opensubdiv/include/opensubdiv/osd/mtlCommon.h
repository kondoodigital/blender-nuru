//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_MTL_COMMON_H
#define OPENSUBDIV3_OSD_MTL_COMMON_H

#include "../version.h"

#include <cstddef>

@protocol MTLDevice;
@protocol MTLCommandQueue;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class MTLContext {
public:
        id<MTLDevice> device = nullptr;
        id<MTLCommandQueue> commandQueue = nullptr;
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif //OPENSUBDIV3_OSD_MTL_COMMON_H
