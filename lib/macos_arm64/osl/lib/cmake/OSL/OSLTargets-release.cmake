#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OSL::oslcomp" for configuration "Release"
set_property(TARGET OSL::oslcomp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OSL::oslcomp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liboslcomp.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/liboslcomp.dylib"
  )

list(APPEND _cmake_import_check_targets OSL::oslcomp )
list(APPEND _cmake_import_check_files_for_OSL::oslcomp "${_IMPORT_PREFIX}/lib/liboslcomp.dylib" )

# Import target "OSL::oslquery" for configuration "Release"
set_property(TARGET OSL::oslquery APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OSL::oslquery PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liboslquery.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/liboslquery.dylib"
  )

list(APPEND _cmake_import_check_targets OSL::oslquery )
list(APPEND _cmake_import_check_files_for_OSL::oslquery "${_IMPORT_PREFIX}/lib/liboslquery.dylib" )

# Import target "OSL::oslexec" for configuration "Release"
set_property(TARGET OSL::oslexec APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OSL::oslexec PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liboslexec.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/liboslexec.dylib"
  )

list(APPEND _cmake_import_check_targets OSL::oslexec )
list(APPEND _cmake_import_check_files_for_OSL::oslexec "${_IMPORT_PREFIX}/lib/liboslexec.dylib" )

# Import target "OSL::oslnoise" for configuration "Release"
set_property(TARGET OSL::oslnoise APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OSL::oslnoise PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liboslnoise.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/liboslnoise.dylib"
  )

list(APPEND _cmake_import_check_targets OSL::oslnoise )
list(APPEND _cmake_import_check_files_for_OSL::oslnoise "${_IMPORT_PREFIX}/lib/liboslnoise.dylib" )

# Import target "OSL::oslc" for configuration "Release"
set_property(TARGET OSL::oslc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OSL::oslc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/oslc"
  )

list(APPEND _cmake_import_check_targets OSL::oslc )
list(APPEND _cmake_import_check_files_for_OSL::oslc "${_IMPORT_PREFIX}/bin/oslc" )

# Import target "OSL::oslinfo" for configuration "Release"
set_property(TARGET OSL::oslinfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OSL::oslinfo PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/oslinfo"
  )

list(APPEND _cmake_import_check_targets OSL::oslinfo )
list(APPEND _cmake_import_check_files_for_OSL::oslinfo "${_IMPORT_PREFIX}/bin/oslinfo" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
