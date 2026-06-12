//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_HBRFVAREDIT_H
#define OPENSUBDIV3_HBRFVAREDIT_H

#include "../hbr/hierarchicalEdit.h"
#include "../hbr/vertexEdit.h"

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrFVarEdit;

template <class T>
std::ostream& operator<<(std::ostream& out, const HbrFVarEdit<T>& path) {
    out << "vertex path = (" << path.faceid << ' ';
    for (int i = 0; i < path.nsubfaces; ++i) {
        out << static_cast<int>(path.subfaces[i]) << ' ';
    }
    return out << static_cast<int>(path.vertexid) << "), edit = (" << path.edit[0] << ',' << path.edit[1] << ',' << path.edit[2] << ')';
}

template <class T>
class HbrFVarEdit : public HbrHierarchicalEdit<T> {

public:

    HbrFVarEdit(int _faceid, int _nsubfaces, unsigned char *_subfaces, unsigned char _vertexid, int _index, int _width, int _offset, typename HbrHierarchicalEdit<T>::Operation _op, float *_edit)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), vertexid(_vertexid), index(_index), width(_width), offset(_offset), op(_op) {
        edit = new float[width];
        memcpy(edit, _edit, width * sizeof(float));
    }

    HbrFVarEdit(int _faceid, int _nsubfaces, int *_subfaces, int _vertexid, int _index, int _width, int _offset, typename HbrHierarchicalEdit<T>::Operation _op, float *_edit)
        : HbrHierarchicalEdit<T>(_faceid, _nsubfaces, _subfaces), vertexid(_vertexid), index(_index), width(_width), offset(_offset), op(_op) {
        edit = new float[width];
        memcpy(edit, _edit, width * sizeof(float));
    }

    virtual ~HbrFVarEdit() {
        delete[] edit;
    }

    // Return the vertex id (the last element in the path)
    unsigned char GetVertexID() const { return vertexid; }

    friend std::ostream& operator<< <T> (std::ostream& out, const HbrFVarEdit<T>& path);

    // Return index into the facevarying data
    int GetIndex() const { return index; }

    // Return width of the data
    int GetWidth() const { return width; }

    // Return offset of the data
    int GetOffset() const { return offset; }

    // Get the numerical value of the edit
    const float* GetEdit() const { return edit; }

    // Get the type of operation
    typename HbrHierarchicalEdit<T>::Operation GetOperation() const { return op; }

    virtual void ApplyEditToFace(HbrFace<T>* face) {
        if (HbrHierarchicalEdit<T>::GetNSubfaces() == face->GetDepth()) {
            // The edit will modify the data and almost certainly
            // create a discontinuity, so allocate storage for a new
            // copy of the existing data specific to the face (or use
            // one that already exists) and modify that
            HbrFVarData<T> &fvt = face->GetVertex(vertexid)->GetFVarData(face);
            if (fvt.GetFaceID() != face->GetID()) {
                // This is the generic fvt, allocate a new copy and edit it
                HbrFVarData<T> &newfvt = face->GetVertex(vertexid)->NewFVarData(face);
                newfvt.SetAllData(face->GetMesh()->GetTotalFVarWidth(), fvt.GetData(0));
                newfvt.ApplyFVarEdit(*const_cast<const HbrFVarEdit<T>*>(this));
            } else {
                fvt.ApplyFVarEdit(*const_cast<const HbrFVarEdit<T>*>(this));
            }
        }
    }

private:
    const unsigned char vertexid;
    const int index;
    const int width;
    const int offset;
    float* edit;
    typename HbrHierarchicalEdit<T>::Operation op;
};


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_HBRFVAREDIT_H */
