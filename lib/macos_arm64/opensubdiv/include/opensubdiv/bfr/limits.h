//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_BFR_LIMITS_H
#define OPENSUBDIV3_BFR_LIMITS_H

#include "../version.h"

#include "../far/types.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {

///
/// @brief Simple struct with limits related to topology
///
struct Limits {
    /// @brief Returns the maximum allowable valence for a vertex
    static int MaxValence()  { return Far::VALENCE_LIMIT; }

    /// @brief Returns the maximum allowable size for a face (number of
    ///        vertices)
    static int MaxFaceSize() { return Far::VALENCE_LIMIT; }
};

} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_BFR_LIMITS_H */
