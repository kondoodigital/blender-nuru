//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_OPENGL_H
#define OPENSUBDIV3_OSD_OPENGL_H

#if defined(__APPLE__)
    #include "TargetConditionals.h"
    #if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        #include <OpenGLES/ES2/gl.h>
    #else
        #if defined(OSD_USES_GLEW)
            #include <GL/glew.h>
        #else
            #include <OpenGL/gl3.h>
        #endif
    #endif
#elif defined(ANDROID)
    #include <GLES2/gl2.h>
#else
    #if defined(_WIN32)
        #define WIN32_LEAN_AND_MEAN
        #include <windows.h>
    #endif
    #if defined(OSD_USES_GLEW)
        #include <GL/glew.h>
    #else
        #include <GL/gl.h>
    #endif
#endif

#endif  // OPENSUBDIV3_OSD_OPENGL_H
