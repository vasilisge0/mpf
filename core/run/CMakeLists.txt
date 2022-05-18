add_executable(test1 mpf_main.c)
add_dependencies(test1 mpf)

#add_compile_options("-DMKL_ILP64 -m64 -I${MKL_INC}")

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

target_include_directories(test1 PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(test1
  PRIVATE ${LAPACK_LIB}
)

target_link_libraries(test1
  ${MPF_LIB}
  -Wl,--no-undefined -Wl,-rpath, -Wl,--no-as-needed
  ${GINKGO_LIB}
  ${BLAS_DEF_LIB}
  ${BLAS_AVX2_LIB}
  ${BLAS_LIB}
  ${LAPACK_LIB}
  -lpthread
  -liomp5
  -lm
  -ldl
)
