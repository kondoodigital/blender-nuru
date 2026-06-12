//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_HBRHOLEEDIT_H
#define OPENSUBDIV3_HBRHOLEEDIT_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrHoleEdit;

template <class T>
std::ostream& operator<<(std::ostream& out, const HbrHoleEdit<T>& path) {
    out << "edge path = (" << path.faceid << ' ';
    for (int i = 0; i < path.nsubfaces; ++i) {
        out << static_cast<int>(path.subfaces[i]) << ' ';
    }
    return out << ")";
}

template <class T>
class HbrHoleEdit : public HbrHierarchicalEdit<T> {

public:

    HbrHoleEdit(int _faceid, int _nsubfaces, unsigned char *_subfaces)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces) {
    }

    HbrHoleEdit(int _faceid, int _nsubfaces, int *_subfaces)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces) {
    }

    virtual ~HbrHoleEdit() {}

    friend std::ostream& operator<< <T> (std::ostream& out, const HbrHoleEdit<T>& path);

    virtual void ApplyEditToFace(HbrFace<T>* face) {
        if (HbrHierarchicalEdit<T>::GetNSubfaces() == face->GetDepth()) {
            face->SetHole();
        }
    }
};


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_HBRHOLEEDIT_H */
