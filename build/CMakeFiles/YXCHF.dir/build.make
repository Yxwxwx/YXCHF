# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/e/code/YXCHF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/e/code/YXCHF/build

# Include any dependencies generated for this target.
include CMakeFiles/YXCHF.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/YXCHF.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/YXCHF.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/YXCHF.dir/flags.make

CMakeFiles/YXCHF.dir/src/main.cpp.o: CMakeFiles/YXCHF.dir/flags.make
CMakeFiles/YXCHF.dir/src/main.cpp.o: /mnt/e/code/YXCHF/src/main.cpp
CMakeFiles/YXCHF.dir/src/main.cpp.o: CMakeFiles/YXCHF.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/mnt/e/code/YXCHF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/YXCHF.dir/src/main.cpp.o"
	/usr/sbin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/YXCHF.dir/src/main.cpp.o -MF CMakeFiles/YXCHF.dir/src/main.cpp.o.d -o CMakeFiles/YXCHF.dir/src/main.cpp.o -c /mnt/e/code/YXCHF/src/main.cpp

CMakeFiles/YXCHF.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/YXCHF.dir/src/main.cpp.i"
	/usr/sbin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/code/YXCHF/src/main.cpp > CMakeFiles/YXCHF.dir/src/main.cpp.i

CMakeFiles/YXCHF.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/YXCHF.dir/src/main.cpp.s"
	/usr/sbin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/code/YXCHF/src/main.cpp -o CMakeFiles/YXCHF.dir/src/main.cpp.s

# Object files for target YXCHF
YXCHF_OBJECTS = \
"CMakeFiles/YXCHF.dir/src/main.cpp.o"

# External object files for target YXCHF
YXCHF_EXTERNAL_OBJECTS =

YXCHF: CMakeFiles/YXCHF.dir/src/main.cpp.o
YXCHF: CMakeFiles/YXCHF.dir/build.make
YXCHF: /usr/lib/libopenblas.so.0.3
YXCHF: CMakeFiles/YXCHF.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/mnt/e/code/YXCHF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable YXCHF"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/YXCHF.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/YXCHF.dir/build: YXCHF
.PHONY : CMakeFiles/YXCHF.dir/build

CMakeFiles/YXCHF.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/YXCHF.dir/cmake_clean.cmake
.PHONY : CMakeFiles/YXCHF.dir/clean

CMakeFiles/YXCHF.dir/depend:
	cd /mnt/e/code/YXCHF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/e/code/YXCHF /mnt/e/code/YXCHF /mnt/e/code/YXCHF/build /mnt/e/code/YXCHF/build /mnt/e/code/YXCHF/build/CMakeFiles/YXCHF.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/YXCHF.dir/depend
