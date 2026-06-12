//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_NON_COPYABLE_H
#define OPENSUBDIV3_OSD_NON_COPYABLE_H

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

template <class T>
class NonCopyable {

protected:
    NonCopyable() {}
    ~NonCopyable() {}

private:
    NonCopyable(const NonCopyable &);
    T & operator = (const T &);
};

} // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_NON_COPYABLE_H
