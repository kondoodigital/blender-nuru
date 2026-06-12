//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_HBRFACEEDIT_H
#define OPENSUBDIV3_HBRFACEEDIT_H

#include "../hbr/hierarchicalEdit.h"

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrFaceEdit;

template <class T>
std::ostream& operator<<(std::ostream& out, const HbrFaceEdit<T>& path) {
    out << "face path = (" << path.faceid << ' ';
    for (int i = 0; i < path.nsubfaces; ++i) {
        out << static_cast<int>(path.subfaces[i]) << ' ';
    }
    return out << ")";
}

template <class T>
class HbrFaceEdit : public HbrHierarchicalEdit<T> {

public:

    HbrFaceEdit(int _faceid, int _nsubfaces, unsigned char *_subfaces, int _index, int _width, typename HbrHierarchicalEdit<T>::Operation _op, float *_edit)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), index(_index), width(_width), op(_op) {
        edit = new float[width];
        memcpy(edit, _edit, width * sizeof(float));
    }

    HbrFaceEdit(int _faceid, int _nsubfaces, int *_subfaces, int _index, int _width, typename HbrHierarchicalEdit<T>::Operation _op, float *_edit)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), index(_index), width(_width), op(_op) {
        edit = new float[width];
        memcpy(edit, _edit, width * sizeof(float));
    }

#ifdef PRMAN
    HbrFaceEdit(int _faceid, int _nsubfaces, unsigned char *_subfaces, int _index, int _width, typename HbrHierarchicalEdit<T>::Operation _op, RtToken _edit)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), index(_index), width(_width), op(_op) {
        edit = new float[width];
        RtString* sedit = (RtString*) edit;
        *sedit = _edit;
    }

    HbrFaceEdit(int _faceid, int _nsubfaces, int *_subfaces, int _index, int _width, typename HbrHierarchicalEdit<T>::Operation _op, RtToken _edit)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), index(_index), width(_width), op(_op) {
        edit = new float[width];
        RtString* sedit = (RtString*) edit;
        *sedit = _edit;
    }
#endif

    virtual ~HbrFaceEdit() {
        delete[] edit;
    }

    friend std::ostream& operator<< <T> (std::ostream& out, const HbrFaceEdit<T>& path);

    // Return index of variable this edit applies to
    int GetIndex() const { return index; }

    // Return width of the variable
    int GetWidth() const { return width; }

    // Get the numerical value of the edit
    const float* GetEdit() const { return edit; }

    // Get the type of operation
    typename HbrHierarchicalEdit<T>::Operation GetOperation() const { return op; }

    virtual void ApplyEditToFace(HbrFace<T>* face) {
        if (HbrHierarchicalEdit<T>::GetNSubfaces() == face->GetDepth()) {

            int oldUniformIndex = face->GetUniformIndex();

            // Any face below level 0 needs a new uniform index
            if (face->GetDepth() > 0) {
            face->SetUniformIndex(face->GetMesh()->NewUniformIndex());
            }

            // Apply edit
            face->GetVertex(0)->GetData().ApplyFaceEdit(oldUniformIndex, face->GetUniformIndex(), *const_cast<const HbrFaceEdit<T>*>(this));
        }
    }

private:
    int index;
    int width;
    typename HbrHierarchicalEdit<T>::Operation op;
    float* edit;
};


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_HBRFACEEDIT_H */
