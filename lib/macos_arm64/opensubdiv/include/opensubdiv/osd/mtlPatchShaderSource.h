//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_MTL_PATCH_SHADER_SOURCE_H
#define OPENSUBDIV3_OSD_MTL_PATCH_SHADER_SOURCE_H

#import "../version.h"

#import "../far/patchDescriptor.h"

#import <string>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief Provides shader source which can be used by client code.
class MTLPatchShaderSource {
public:
    /// \brief Returns shader source which can be used to evaluate
    /// position and first and second derivatives on piecewise parametric
    /// patches resulting from subdivision refinement.
    static std::string GetPatchBasisShaderSource();

    /// \brief Returns shader source which can be used while drawing
    /// piecewise parametric patches resulting from subdivision refinement,
    /// e.g. while using GPU HW tessellation.
    static std::string GetPatchDrawingShaderSource();

    /// \name Alternative methods
    /// \{
    /// These methods return shader source which can be used
    /// while drawing. Unlike the methods above, the source returned
    /// by these methods includes support for legacy patch types along
    /// with dependencies on specific resource bindings and interstage
    /// shader variable declarations.

    static std::string GetCommonShaderSource();

    static std::string GetVertexShaderSource(Far::PatchDescriptor::Type type);

    static std::string GetHullShaderSource(Far::PatchDescriptor::Type type);

    static std::string GetDomainShaderSource(Far::PatchDescriptor::Type type);

    /// These methods are deprecated. Clients should determine the
    /// patch type of a face-varying patch by inspecting the
    /// face-varying patch array descriptors.
    /// \brief Deprecated
    static std::string GetVertexShaderSource(
        Far::PatchDescriptor::Type type,
        Far::PatchDescriptor::Type fvarType);
    static std::string GetHullShaderSource(
        Far::PatchDescriptor::Type type,
        Far::PatchDescriptor::Type fvarType);
    static std::string GetDomainShaderSource(
        Far::PatchDescriptor::Type type,
        Far::PatchDescriptor::Type fvarType);

    /// @}

};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
    
} // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_MTL_PATCH_SHADER_SOURCE
