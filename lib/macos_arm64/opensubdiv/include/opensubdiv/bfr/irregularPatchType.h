//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_BFR_IRREGULAR_PATCH_TYPE_H
#define OPENSUBDIV3_BFR_IRREGULAR_PATCH_TYPE_H

#include "../version.h"

#include <memory>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {

//
//  This header was intended to contain an abstract class providing the
//  common interface to potentially more than one representations of the
//  irregular patches.  But given only one implementation exists -- the
//  PatchTree -- that level of abstraction has been deferred.
//
//  So this header serves to encapsulate the forward declaration of that
//  single representation, along with related typedefs for internal use.
//
class PatchTree;

namespace internal {

    typedef PatchTree IrregularPatchType;

    typedef std::shared_ptr<const IrregularPatchType> IrregularPatchSharedPtr;

} // end namespace internal

} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_BFR_IRREGULAR_PATCH_TYPE */
