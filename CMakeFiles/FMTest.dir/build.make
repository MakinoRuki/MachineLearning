# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ruki/Documents/Developer/MachineLearning

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ruki/Documents/Developer/MachineLearning

# Include any dependencies generated for this target.
include CMakeFiles/FMTest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FMTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FMTest.dir/flags.make

CMakeFiles/FMTest.dir/FM_test.cpp.o: CMakeFiles/FMTest.dir/flags.make
CMakeFiles/FMTest.dir/FM_test.cpp.o: FM_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/ruki/Documents/Developer/MachineLearning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FMTest.dir/FM_test.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FMTest.dir/FM_test.cpp.o -c /Users/ruki/Documents/Developer/MachineLearning/FM_test.cpp

CMakeFiles/FMTest.dir/FM_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FMTest.dir/FM_test.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ruki/Documents/Developer/MachineLearning/FM_test.cpp > CMakeFiles/FMTest.dir/FM_test.cpp.i

CMakeFiles/FMTest.dir/FM_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FMTest.dir/FM_test.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ruki/Documents/Developer/MachineLearning/FM_test.cpp -o CMakeFiles/FMTest.dir/FM_test.cpp.s

# Object files for target FMTest
FMTest_OBJECTS = \
"CMakeFiles/FMTest.dir/FM_test.cpp.o"

# External object files for target FMTest
FMTest_EXTERNAL_OBJECTS =

FMTest: CMakeFiles/FMTest.dir/FM_test.cpp.o
FMTest: CMakeFiles/FMTest.dir/build.make
FMTest: CMakeFiles/FMTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/ruki/Documents/Developer/MachineLearning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FMTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FMTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FMTest.dir/build: FMTest

.PHONY : CMakeFiles/FMTest.dir/build

CMakeFiles/FMTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FMTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FMTest.dir/clean

CMakeFiles/FMTest.dir/depend:
	cd /Users/ruki/Documents/Developer/MachineLearning && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ruki/Documents/Developer/MachineLearning /Users/ruki/Documents/Developer/MachineLearning /Users/ruki/Documents/Developer/MachineLearning /Users/ruki/Documents/Developer/MachineLearning /Users/ruki/Documents/Developer/MachineLearning/CMakeFiles/FMTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FMTest.dir/depend

