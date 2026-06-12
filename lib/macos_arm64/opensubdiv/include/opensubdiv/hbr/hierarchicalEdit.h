//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_HBRHIERARCHICALEDIT_H
#define OPENSUBDIV3_HBRHIERARCHICALEDIT_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrHierarchicalEdit;
template <class T> class HbrFace;
template <class T> class HbrVertex;

template <class T>
class HbrHierarchicalEdit {

public:
    typedef enum Operation {
        Set,
        Add,
        Subtract
    } Operation;

protected:

    HbrHierarchicalEdit(int _faceid, int _nsubfaces, unsigned char *_subfaces)
        : faceid(_faceid), nsubfaces(_nsubfaces) {
        subfaces = new unsigned char[_nsubfaces];
        for (int i = 0; i < nsubfaces; ++i) {
            subfaces[i] = _subfaces[i];
        }
    }

    HbrHierarchicalEdit(int _faceid, int _nsubfaces, int *_subfaces)
        : faceid(_faceid), nsubfaces(_nsubfaces) {
        subfaces = new unsigned char[_nsubfaces];
        for (int i = 0; i < nsubfaces; ++i) {
            subfaces[i] = static_cast<unsigned char>(_subfaces[i]);
        }
    }

public:
    virtual ~HbrHierarchicalEdit() {
        delete[] subfaces;
    }

    bool operator<(const HbrHierarchicalEdit& p) const {
        if (faceid < p.faceid) return true;
        if (faceid > p.faceid) return false;
        int minlength = nsubfaces;
        if (minlength > p.nsubfaces) minlength = p.nsubfaces;
        for (int i = 0; i < minlength; ++i) {
            if (subfaces[i] < p.subfaces[i]) return true;
            if (subfaces[i] > p.subfaces[i]) return false;
        }
        return (nsubfaces < p.nsubfaces);
    }

    // Return the face id (the first element in the path)
    int GetFaceID() const { return faceid; }

    // Return the number of subfaces in the path
    int GetNSubfaces() const { return nsubfaces; }

    // Return a subface element in the path
    unsigned char GetSubface(int index) const { return subfaces[index]; }

    // Determines whether this hierarchical edit is relevant to the
    // face in question
    bool IsRelevantToFace(HbrFace<T>* face) const;

    // Applys edit to face. All subclasses may override this method
    virtual void ApplyEditToFace(HbrFace<T>* /* face */) {}

    // Applys edit to vertex. Subclasses may override this method.
    virtual void ApplyEditToVertex(HbrFace<T>* /* face */, HbrVertex<T>* /* vertex */) {}

#ifdef PRMAN
    // Gets the effect of this hierarchical edit on the bounding box.
    // Subclasses may override this method
    virtual void ApplyToBound(struct bbox& /* box */, RtMatrix * /* mx */) const {}
#endif

protected:
    // ID of the top most face in the mesh which begins the path
    const int faceid;

    // Number of subfaces
    const int nsubfaces;

    // IDs of the subfaces
    unsigned char *subfaces;
};

template <class T>
class HbrHierarchicalEditComparator {
public:
    bool operator() (const HbrHierarchicalEdit<T>* path1, const HbrHierarchicalEdit<T>* path2) const {
        return (*path1 < *path2);
    }
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#include "../hbr/face.h"
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T>
bool
HbrHierarchicalEdit<T>::IsRelevantToFace(HbrFace<T>* face) const {

    // Key assumption: the face's first vertex edit is relevant to
    // that face. We will then compare ourselves to that edit and if
    // the first part of our subpath is identical to the entirety of
    // that subpath, this edit is relevant.

    // Calling code is responsible for making sure we don't
    // dereference a null pointer here
    HbrHierarchicalEdit<T>* p = *face->GetHierarchicalEdits();
    if (!p) return false;

    if (this == p) return true;

    if (faceid != p->faceid) return false;

    // If our path length is less than the face depth, it should mean
    // that we're dealing with another face somewhere up the path, so
    // we're not relevant
    if (nsubfaces < face->GetDepth()) return false;

    if (memcmp(subfaces, p->subfaces, face->GetDepth() * sizeof(unsigned char)) != 0) {
        return false;
    }
    return true;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_HBRHIERARCHICALEDIT_H */
