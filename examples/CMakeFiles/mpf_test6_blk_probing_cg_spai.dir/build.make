# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vasilis/mpf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vasilis/mpf

# Include any dependencies generated for this target.
include examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/flags.make

examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.o: examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/flags.make
examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.o: examples/mpf_test6_blk_probing_cg_spai.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vasilis/mpf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.o"
	cd /home/vasilis/mpf/examples && /usr/bin/g++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.o -c /home/vasilis/mpf/examples/mpf_test6_blk_probing_cg_spai.c

examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.i"
	cd /home/vasilis/mpf/examples && /usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vasilis/mpf/examples/mpf_test6_blk_probing_cg_spai.c > CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.i

examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.s"
	cd /home/vasilis/mpf/examples && /usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vasilis/mpf/examples/mpf_test6_blk_probing_cg_spai.c -o CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.s

# Object files for target mpf_test6_blk_probing_cg_spai
mpf_test6_blk_probing_cg_spai_OBJECTS = \
"CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.o"

# External object files for target mpf_test6_blk_probing_cg_spai
mpf_test6_blk_probing_cg_spai_EXTERNAL_OBJECTS =

examples/mpf_test6_blk_probing_cg_spai: examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/mpf_test6_blk_probing_cg_spai.c.o
examples/mpf_test6_blk_probing_cg_spai: examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/build.make
examples/mpf_test6_blk_probing_cg_spai: /home/vasilis/intel/mkl/lib/intel64/libmkl_intel_ilp64.so
examples/mpf_test6_blk_probing_cg_spai: /home/vasilis/intel/mkl/lib/intel64/libmkl_intel_thread.so
examples/mpf_test6_blk_probing_cg_spai: /home/vasilis/intel/mkl/lib/intel64/libmkl_core.so
examples/mpf_test6_blk_probing_cg_spai: build/core/libmpf.so
examples/mpf_test6_blk_probing_cg_spai: examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vasilis/mpf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mpf_test6_blk_probing_cg_spai"
	cd /home/vasilis/mpf/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/build: examples/mpf_test6_blk_probing_cg_spai

.PHONY : examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/build

examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/clean:
	cd /home/vasilis/mpf/examples && $(CMAKE_COMMAND) -P CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/clean

examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/depend:
	cd /home/vasilis/mpf && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vasilis/mpf /home/vasilis/mpf/examples /home/vasilis/mpf /home/vasilis/mpf/examples /home/vasilis/mpf/examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/mpf_test6_blk_probing_cg_spai.dir/depend

