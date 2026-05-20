#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TBB::tbb" for configuration "RelWithDebInfo"
set_property(TARGET TBB::tbb APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TBB::tbb PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib64/libtbb.so.12.14"
  IMPORTED_SONAME_RELWITHDEBINFO "libtbb.so.12"
  )

list(APPEND _cmake_import_check_targets TBB::tbb )
list(APPEND _cmake_import_check_files_for_TBB::tbb "${_IMPORT_PREFIX}/lib64/libtbb.so.12.14" )

# Import target "TBB::tbbmalloc" for configuration "RelWithDebInfo"
set_property(TARGET TBB::tbbmalloc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TBB::tbbmalloc PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib64/libtbbmalloc.so.2.14"
  IMPORTED_SONAME_RELWITHDEBINFO "libtbbmalloc.so.2"
  )

list(APPEND _cmake_import_check_targets TBB::tbbmalloc )
list(APPEND _cmake_import_check_files_for_TBB::tbbmalloc "${_IMPORT_PREFIX}/lib64/libtbbmalloc.so.2.14" )

# Import target "TBB::tbbmalloc_proxy" for configuration "RelWithDebInfo"
set_property(TARGET TBB::tbbmalloc_proxy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TBB::tbbmalloc_proxy PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELWITHDEBINFO "TBB::tbbmalloc"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib64/libtbbmalloc_proxy.so.2.14"
  IMPORTED_SONAME_RELWITHDEBINFO "libtbbmalloc_proxy.so.2"
  )

list(APPEND _cmake_import_check_targets TBB::tbbmalloc_proxy )
list(APPEND _cmake_import_check_files_for_TBB::tbbmalloc_proxy "${_IMPORT_PREFIX}/lib64/libtbbmalloc_proxy.so.2.14" )

# Import target "TBB::tbbbind_2_5" for configuration "RelWithDebInfo"
set_property(TARGET TBB::tbbbind_2_5 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TBB::tbbbind_2_5 PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib64/libtbbbind_2_5.so.3.14"
  IMPORTED_SONAME_RELWITHDEBINFO "libtbbbind_2_5.so.3"
  )

list(APPEND _cmake_import_check_targets TBB::tbbbind_2_5 )
list(APPEND _cmake_import_check_files_for_TBB::tbbbind_2_5 "${_IMPORT_PREFIX}/lib64/libtbbbind_2_5.so.3.14" )

# Import target "TBB::irml" for configuration "RelWithDebInfo"
set_property(TARGET TBB::irml APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TBB::irml PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib64/libirml.so.1"
  IMPORTED_SONAME_RELWITHDEBINFO "libirml.so.1"
  )

list(APPEND _cmake_import_check_targets TBB::irml )
list(APPEND _cmake_import_check_files_for_TBB::irml "${_IMPORT_PREFIX}/lib64/libirml.so.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
