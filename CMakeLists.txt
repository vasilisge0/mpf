cmake_minimum_required(VERSION 3.13.11)
project(mpf LANGUAGES C CXX VERSION 1.0 DESCRIPTION "A numerical library for multilevel probing methods.")

include(InstallRequiredSystemLibraries)
include(CPack)

set(mpf_VERSION_TAG "develop")
set(PROJECT_VERSION_TAG ${mpf_VERSION_TAG})

### Configure those variables ###

# this has to be modified
set(LAPACK_INC /opt/intel/mkl/include)
#set(GINKGO_INC /usr/local/include/include)
set(GINKGO_INC $ENV{HOME}/apps/ginkgo/include)

#find_library(LAPACK_LIB libmkl_intel_ilp64.so)
#find_library(BLAS_THREAD_LIB libmkl_intel_thread.so)
#find_library(BLAS_LIB libmkl_core.so)
#find_library(BLAS_AVX2_LIB libmkl_avx2.so)
#find_library(BLAS_DEF_LIB libmkl_def.so)

set(LAPACK_LIB /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_ilp64.so)
set(BLAS_THREAD_LIB /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so)
set(BLAS_LIB /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so)
set(BLAS_AVX2_LIB /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so.2)
set(BLAS_DEF_LIB /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.2)

#find_library(GINKGO_LIB /usr/local/lib/libginkgo.so)
#set(GINKGO_LIB /usr/local/lib/libginkgo.so)
set(GINKGO_LIB $ENV{HOME}/apps/ginkgo/build/lib/libginkgo.so)

# MPF executable
set(MPF_LIB ${PROJECT_SOURCE_DIR}/build/core/libmpf.so)

add_compile_options(-DMKL_ILP64 -m64)

### Project subdirectories ###

add_subdirectory(core)
add_subdirectory(core/run)
add_subdirectory(examples)

### Instalation ###

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(TARGETS test1 DESTINATION lib)
install(TARGETS mpf_test0_blk_probing_cg DESTINATION examples/index_by_method)
install(TARGETS mpf_test1_blk_probing_blk_cg DESTINATION examples/index_by_method)
install(TARGETS mpf_test2_blk_probing_gbl_cg DESTINATION examples/index_by_method)
install(TARGETS mpf_test3_blk_probing_gmres DESTINATION examples/index_by_method)
install(TARGETS mpf_test4_blk_probing_blk_gmres DESTINATION examples/index_by_method)
install(TARGETS mpf_test5_blk_probing_gbl_gmres DESTINATION examples/index_by_method)
install(TARGETS mpf_test6_blk_probing_cg_spai DESTINATION examples/index_by_method)
install(TARGETS mpf_test7_blk_probing_cg DESTINATION examples/index_by_method)
install(TARGETS mpf_test8_blk_probing_cg DESTINATION examples/index_by_method)
install(TARGETS mpf_test9_blk_probing_cg DESTINATION examples/index_by_method)
install(TARGETS mpf_application_00_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_01_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_02_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_03_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_04_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_05_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_06_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_07_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_08_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_09_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_10_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_11_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_12_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_13_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_14_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_15_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_16_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_17_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_18_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_19_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_20_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_21_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_22_uncov DESTINATION examples/index_by_application)
install(TARGETS mpf_application_23_uncov DESTINATION examples/index_by_application)

install(FILES
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test0_blk_probing_cg_run.sh"
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test1_blk_probing_blk_cg_run.sh"
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test2_blk_probing_gbl_cg_run.sh"
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test3_blk_probing_gmres_run.sh"
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test4_blk_probing_blk_gmres_run.sh"
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test5_blk_probing_gbl_gmres_run.sh"
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test6_blk_probing_cg_spai_run.sh"
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test7_blk_probing_cg_run.sh"
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test8_blk_probing_cg_run.sh"
        "${PROJECT_SOURCE_DIR}/examples/index_by_method/mpf_test9_blk_probing_cg_run.sh"
        DESTINATION examples/index_by_method)

install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_00_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_01_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_02_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_03_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_04_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_05_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_06_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_07_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_08_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_09_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_10_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_11_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_12_uncov.sh"
        DESTINATION examples/index_by_application)
install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_13_uncov.sh"
        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_14_uncov.sh"
#        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_15_uncov.sh"
#        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_16_uncov.sh"
#        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_17_uncov.sh"
#        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_18_uncov.sh"
#        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_19_uncov.sh"
#        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_20_uncov.sh"
#        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_21_uncov.sh"
#        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_22_uncov.sh"
#        DESTINATION examples/index_by_application)
#install(FILES "${PROJECT_SOURCE_DIR}/examples/index_by_application/mpf_application_23_uncov.sh"
#        DESTINATION examples/index_by_application)

install(FILES
        "${PROJECT_SOURCE_DIR}/core/run/mpf.h"
        "${PROJECT_SOURCE_DIR}/core/types/mpf_types.h"
        "${PROJECT_SOURCE_DIR}/core/types/mpf_aux.h"
        "${PROJECT_SOURCE_DIR}/core/solve/mpf_solve.h"
        "${PROJECT_SOURCE_DIR}/core/probe/mpf_probe.h"
        "${PROJECT_SOURCE_DIR}/core/io/mpf_mmio.h"
        "${PROJECT_SOURCE_DIR}/core/blas/mpf_blas_mkl_internal.h"
        "${PROJECT_SOURCE_DIR}/core/blas/mpf_blas.h"
        DESTINATION include)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/matlab" DESTINATION .)
#FIX: specify external data directory and check it exists. If it does copy the
#data to install folder.
#install(DIRECTORY "${PROJECT_SOURCE_DIR}/data" DESTINATION .)
install(DIRECTORY DESTINATION output)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR})

file(GLOB_RECURSE CFILES "${CMAKE_SOURCE_DIR}/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX )
