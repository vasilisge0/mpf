#! /bin/sh
cmake -G "Unix Makefiles" -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_PREFIX_PATH="$HOME/intel/mkl/lib/intel64;$HOME/ginkgo_install/lib" -DCMAKE_INSTALL_PREFIX="$HOME/apps/mpf_install" -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -S . -B build

#
#-DCMAKE_INSTALL_PREFIX=/home/vasilis/mpf_install  
