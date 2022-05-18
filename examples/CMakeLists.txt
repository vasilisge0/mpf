add_executable(mpf_test0_blk_probing_cg index_by_method/mpf_test0_blk_probing_cg.c)

add_dependencies(mpf_test0_blk_probing_cg mpf)
set_target_properties(mpf_test0_blk_probing_cg PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test0_blk_probing_cg PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_libraries(mpf_test0_blk_probing_cg
  -g -v -Wl,--no-as-needed ${MPF_LIB} ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} -liomp5 -lpthread -lm -ldl
)

add_executable(mpf_test1_blk_probing_blk_cg index_by_method/mpf_test1_blk_probing_blk_cg.c)

add_dependencies(mpf_test1_blk_probing_blk_cg mpf)
set_target_properties(mpf_test1_blk_probing_blk_cg PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test1_blk_probing_blk_cg PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_libraries(mpf_test1_blk_probing_blk_cg
  -g -v -Wl,--no-as-needed ${MPF_LIB} ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} -liomp5 -lpthread -lm -ldl
)

add_executable(mpf_test2_blk_probing_gbl_cg index_by_method/mpf_test2_blk_probing_gbl_cg.c)

add_dependencies(mpf_test2_blk_probing_gbl_cg mpf)
set_target_properties(mpf_test2_blk_probing_gbl_cg PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test2_blk_probing_gbl_cg PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_libraries(mpf_test2_blk_probing_gbl_cg
  -g -v -Wl,--no-as-needed ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

add_executable(mpf_test3_blk_probing_gmres index_by_method/mpf_test3_blk_probing_gmres.c)

add_dependencies(mpf_test3_blk_probing_gmres mpf)
set_target_properties(mpf_test3_blk_probing_gmres PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test3_blk_probing_gmres PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_test3_blk_probing_gmres
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_test3_blk_probing_gmres
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

add_executable(mpf_test4_blk_probing_blk_gmres index_by_method/mpf_test4_blk_probing_blk_gmres.c)

add_dependencies(mpf_test4_blk_probing_blk_gmres mpf)
set_target_properties(mpf_test4_blk_probing_blk_gmres PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test4_blk_probing_blk_gmres PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_test4_blk_probing_blk_gmres
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_test4_blk_probing_blk_gmres
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

add_executable(mpf_test5_blk_probing_gbl_gmres index_by_method/mpf_test5_blk_probing_gbl_gmres.c)

add_dependencies(mpf_test5_blk_probing_gbl_gmres mpf)
set_target_properties(mpf_test5_blk_probing_gbl_gmres PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test5_blk_probing_gbl_gmres PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_test5_blk_probing_gbl_gmres
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_test5_blk_probing_gbl_gmres
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

add_executable(mpf_test6_blk_probing_cg_spai index_by_method/mpf_test6_blk_probing_cg_spai.c)

add_dependencies(mpf_test6_blk_probing_cg_spai mpf)
set_target_properties(mpf_test6_blk_probing_cg_spai PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test6_blk_probing_cg_spai PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_test6_blk_probing_cg_spai
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_test6_blk_probing_cg_spai
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

add_executable(mpf_test7_blk_probing_cg index_by_method/mpf_test7_blk_probing_cg.c)

add_dependencies(mpf_test7_blk_probing_cg mpf)
set_target_properties(mpf_test7_blk_probing_cg PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test7_blk_probing_cg PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_test7_blk_probing_cg
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_test7_blk_probing_cg
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

add_executable(mpf_test8_blk_probing_cg index_by_method/mpf_test8_blk_probing_cg.c)

add_dependencies(mpf_test8_blk_probing_cg mpf)
set_target_properties(mpf_test8_blk_probing_cg PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test8_blk_probing_cg PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_test8_blk_probing_cg
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_test8_blk_probing_cg
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)


add_executable(mpf_test9_blk_probing_cg index_by_method/mpf_test9_blk_probing_cg.c)

add_dependencies(mpf_test9_blk_probing_cg mpf)
set_target_properties(mpf_test9_blk_probing_cg PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_test9_blk_probing_cg PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_test9_blk_probing_cg
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_test9_blk_probing_cg
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)


# mpf_application_0_uncov
add_executable(mpf_application_00_uncov index_by_application/mpf_application_00_uncov.c)

add_dependencies(mpf_application_00_uncov mpf)
set_target_properties(mpf_application_00_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_00_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_00_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_00_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)


# mpf_application_01_uncov
add_executable(mpf_application_01_uncov index_by_application/mpf_application_01_uncov.c)

add_dependencies(mpf_application_01_uncov mpf)
set_target_properties(mpf_application_01_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_01_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_01_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_01_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_02_uncov
add_executable(mpf_application_02_uncov index_by_application/mpf_application_02_uncov.c)

add_dependencies(mpf_application_02_uncov mpf)
set_target_properties(mpf_application_02_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_02_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_02_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_02_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_03_uncov
add_executable(mpf_application_03_uncov index_by_application/mpf_application_03_uncov.c)

add_dependencies(mpf_application_03_uncov mpf)
set_target_properties(mpf_application_03_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_03_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_03_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_03_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_04_uncov
add_executable(mpf_application_04_uncov index_by_application/mpf_application_04_uncov.c)

add_dependencies(mpf_application_04_uncov mpf)
set_target_properties(mpf_application_04_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_04_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_04_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_04_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_05_uncov
add_executable(mpf_application_05_uncov index_by_application/mpf_application_05_uncov.c)

add_dependencies(mpf_application_05_uncov mpf)
set_target_properties(mpf_application_05_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_05_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_05_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_05_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_06_uncov
add_executable(mpf_application_06_uncov index_by_application/mpf_application_06_uncov.c)

add_dependencies(mpf_application_06_uncov mpf)
set_target_properties(mpf_application_06_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_06_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_06_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_06_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_07_uncov
add_executable(mpf_application_07_uncov index_by_application/mpf_application_07_uncov.c)

add_dependencies(mpf_application_07_uncov mpf)
set_target_properties(mpf_application_07_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_07_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_07_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_07_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_08_uncov
add_executable(mpf_application_08_uncov index_by_application/mpf_application_08_uncov.c)

add_dependencies(mpf_application_08_uncov mpf)
set_target_properties(mpf_application_08_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_08_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_08_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_08_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_09_uncov
add_executable(mpf_application_09_uncov index_by_application/mpf_application_09_uncov.c)

add_dependencies(mpf_application_09_uncov mpf)
set_target_properties(mpf_application_09_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_09_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_09_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_09_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_10_uncov
add_executable(mpf_application_10_uncov index_by_application/mpf_application_10_uncov.c)

add_dependencies(mpf_application_10_uncov mpf)
set_target_properties(mpf_application_10_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_10_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_10_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_10_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_11_uncov
add_executable(mpf_application_11_uncov index_by_application/mpf_application_11_uncov.c)

add_dependencies(mpf_application_11_uncov mpf)
set_target_properties(mpf_application_11_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_11_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_11_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_11_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_12_uncov
add_executable(mpf_application_12_uncov index_by_application/mpf_application_12_uncov.c)

add_dependencies(mpf_application_12_uncov mpf)
set_target_properties(mpf_application_12_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_12_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_12_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_12_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_13_uncov
add_executable(mpf_application_13_uncov index_by_application/mpf_application_13_uncov.c)

add_dependencies(mpf_application_13_uncov mpf)
set_target_properties(mpf_application_13_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_13_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_13_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_13_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_14_uncov
add_executable(mpf_application_14_uncov index_by_application/mpf_application_14_uncov.c)

add_dependencies(mpf_application_14_uncov mpf)
set_target_properties(mpf_application_14_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_14_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_14_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_14_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)


# mpf_application_15_uncov
add_executable(mpf_application_15_uncov index_by_application/mpf_application_15_uncov.c)

add_dependencies(mpf_application_15_uncov mpf)
set_target_properties(mpf_application_15_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_15_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_15_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_15_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_16_uncov
add_executable(mpf_application_16_uncov index_by_application/mpf_application_16_uncov.c)

add_dependencies(mpf_application_16_uncov mpf)
set_target_properties(mpf_application_16_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_16_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_16_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_16_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_17_uncov
add_executable(mpf_application_17_uncov index_by_application/mpf_application_17_uncov.c)

add_dependencies(mpf_application_17_uncov mpf)
set_target_properties(mpf_application_17_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_17_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_17_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_17_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_18_uncov
add_executable(mpf_application_18_uncov index_by_application/mpf_application_18_uncov.c)

add_dependencies(mpf_application_18_uncov mpf)
set_target_properties(mpf_application_18_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_18_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_18_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_18_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_19_uncov
add_executable(mpf_application_19_uncov index_by_application/mpf_application_19_uncov.c)

add_dependencies(mpf_application_19_uncov mpf)
set_target_properties(mpf_application_19_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_19_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_19_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_19_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_20_uncov
add_executable(mpf_application_20_uncov index_by_application/mpf_application_20_uncov.c)

add_dependencies(mpf_application_20_uncov mpf)
set_target_properties(mpf_application_20_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_sOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFIleS} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_20_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_20_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_20_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_21_uncov
add_executable(mpf_application_21_uncov index_by_application/mpf_application_21_uncov.c)

add_dependencies(mpf_application_21_uncov mpf)
set_target_properties(mpf_application_21_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_sOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFIleS} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_21_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_21_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_21_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_22_uncov
add_executable(mpf_application_22_uncov index_by_application/mpf_application_22_uncov.c)

add_dependencies(mpf_application_22_uncov mpf)
set_target_properties(mpf_application_22_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_sOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFIleS} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_22_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_22_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_22_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)

# mpf_application_23_uncov
add_executable(mpf_application_23_uncov index_by_application/mpf_application_23_uncov.c)

add_dependencies(mpf_application_23_uncov mpf)
set_target_properties(mpf_application_23_uncov PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB_RECURSE CFILES "${CMAKE_sOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFIleS} PROPERTIES LANGUAGE CXX)

#include_directories(${LAPACK_LIB})
target_include_directories(mpf_application_23_uncov PUBLIC
  ${CMAKE_SOURCE_DIR}/core/types
  ${CMAKE_SOURCE_DIR}/core/run
  ${CMAKE_SOURCE_DIR}/core/solve
  ${CMAKE_SOURCE_DIR}/core/probe
  ${CMAKE_SOURCE_DIR}/core/io
  ${CMAKE_SOURCE_DIR}/core/blas
  ${CMAKE_SOURCE_DIR}/core/blas
  ${LAPACK_INC}
  ${GINKGO_INC}
)

target_link_directories(mpf_application_23_uncov
  PRIVATE ${LAPACK_LIB} ${MPF_LIB} ${GINKGO_LIB}
)

target_link_libraries(mpf_application_23_uncov
  -g -v -Wl,--no-as-needed  ${LAPACK_LIB} ${BLAS_THREAD_LIB} ${BLAS_LIB} ${MPF_LIB} -liomp5 -lpthread -lm -ldl
)
