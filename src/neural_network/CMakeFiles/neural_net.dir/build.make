# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jiguangshen/HPC_MachineLearning/MLcpp/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiguangshen/HPC_MachineLearning/MLcpp/src

# Include any dependencies generated for this target.
include neural_network/CMakeFiles/neural_net.dir/depend.make

# Include the progress variables for this target.
include neural_network/CMakeFiles/neural_net.dir/progress.make

# Include the compile flags for this target's objects.
include neural_network/CMakeFiles/neural_net.dir/flags.make

neural_network/CMakeFiles/neural_net.dir/pch.cpp.o: neural_network/CMakeFiles/neural_net.dir/flags.make
neural_network/CMakeFiles/neural_net.dir/pch.cpp.o: neural_network/pch.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jiguangshen/HPC_MachineLearning/MLcpp/src/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object neural_network/CMakeFiles/neural_net.dir/pch.cpp.o"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/neural_net.dir/pch.cpp.o -c /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/pch.cpp

neural_network/CMakeFiles/neural_net.dir/pch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neural_net.dir/pch.cpp.i"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/pch.cpp > CMakeFiles/neural_net.dir/pch.cpp.i

neural_network/CMakeFiles/neural_net.dir/pch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neural_net.dir/pch.cpp.s"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/pch.cpp -o CMakeFiles/neural_net.dir/pch.cpp.s

neural_network/CMakeFiles/neural_net.dir/pch.cpp.o.requires:
.PHONY : neural_network/CMakeFiles/neural_net.dir/pch.cpp.o.requires

neural_network/CMakeFiles/neural_net.dir/pch.cpp.o.provides: neural_network/CMakeFiles/neural_net.dir/pch.cpp.o.requires
	$(MAKE) -f neural_network/CMakeFiles/neural_net.dir/build.make neural_network/CMakeFiles/neural_net.dir/pch.cpp.o.provides.build
.PHONY : neural_network/CMakeFiles/neural_net.dir/pch.cpp.o.provides

neural_network/CMakeFiles/neural_net.dir/pch.cpp.o.provides.build: neural_network/CMakeFiles/neural_net.dir/pch.cpp.o
.PHONY : neural_network/CMakeFiles/neural_net.dir/pch.cpp.o.provides.build

neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o: neural_network/CMakeFiles/neural_net.dir/flags.make
neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o: neural_network/neuron.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jiguangshen/HPC_MachineLearning/MLcpp/src/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/neural_net.dir/neuron.cpp.o -c /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/neuron.cpp

neural_network/CMakeFiles/neural_net.dir/neuron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neural_net.dir/neuron.cpp.i"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/neuron.cpp > CMakeFiles/neural_net.dir/neuron.cpp.i

neural_network/CMakeFiles/neural_net.dir/neuron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neural_net.dir/neuron.cpp.s"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/neuron.cpp -o CMakeFiles/neural_net.dir/neuron.cpp.s

neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o.requires:
.PHONY : neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o.requires

neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o.provides: neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o.requires
	$(MAKE) -f neural_network/CMakeFiles/neural_net.dir/build.make neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o.provides.build
.PHONY : neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o.provides

neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o.provides.build: neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o
.PHONY : neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o.provides.build

neural_network/CMakeFiles/neural_net.dir/layer.cpp.o: neural_network/CMakeFiles/neural_net.dir/flags.make
neural_network/CMakeFiles/neural_net.dir/layer.cpp.o: neural_network/layer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jiguangshen/HPC_MachineLearning/MLcpp/src/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object neural_network/CMakeFiles/neural_net.dir/layer.cpp.o"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/neural_net.dir/layer.cpp.o -c /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/layer.cpp

neural_network/CMakeFiles/neural_net.dir/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neural_net.dir/layer.cpp.i"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/layer.cpp > CMakeFiles/neural_net.dir/layer.cpp.i

neural_network/CMakeFiles/neural_net.dir/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neural_net.dir/layer.cpp.s"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/layer.cpp -o CMakeFiles/neural_net.dir/layer.cpp.s

neural_network/CMakeFiles/neural_net.dir/layer.cpp.o.requires:
.PHONY : neural_network/CMakeFiles/neural_net.dir/layer.cpp.o.requires

neural_network/CMakeFiles/neural_net.dir/layer.cpp.o.provides: neural_network/CMakeFiles/neural_net.dir/layer.cpp.o.requires
	$(MAKE) -f neural_network/CMakeFiles/neural_net.dir/build.make neural_network/CMakeFiles/neural_net.dir/layer.cpp.o.provides.build
.PHONY : neural_network/CMakeFiles/neural_net.dir/layer.cpp.o.provides

neural_network/CMakeFiles/neural_net.dir/layer.cpp.o.provides.build: neural_network/CMakeFiles/neural_net.dir/layer.cpp.o
.PHONY : neural_network/CMakeFiles/neural_net.dir/layer.cpp.o.provides.build

neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o: neural_network/CMakeFiles/neural_net.dir/flags.make
neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o: neural_network/nnet.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jiguangshen/HPC_MachineLearning/MLcpp/src/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/neural_net.dir/nnet.cpp.o -c /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/nnet.cpp

neural_network/CMakeFiles/neural_net.dir/nnet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neural_net.dir/nnet.cpp.i"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/nnet.cpp > CMakeFiles/neural_net.dir/nnet.cpp.i

neural_network/CMakeFiles/neural_net.dir/nnet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neural_net.dir/nnet.cpp.s"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/nnet.cpp -o CMakeFiles/neural_net.dir/nnet.cpp.s

neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o.requires:
.PHONY : neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o.requires

neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o.provides: neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o.requires
	$(MAKE) -f neural_network/CMakeFiles/neural_net.dir/build.make neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o.provides.build
.PHONY : neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o.provides

neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o.provides.build: neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o
.PHONY : neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o.provides.build

# Object files for target neural_net
neural_net_OBJECTS = \
"CMakeFiles/neural_net.dir/pch.cpp.o" \
"CMakeFiles/neural_net.dir/neuron.cpp.o" \
"CMakeFiles/neural_net.dir/layer.cpp.o" \
"CMakeFiles/neural_net.dir/nnet.cpp.o"

# External object files for target neural_net
neural_net_EXTERNAL_OBJECTS =

/home/jiguangshen/HPC_MachineLearning/MLcpp/lib/libneural_net.a: neural_network/CMakeFiles/neural_net.dir/pch.cpp.o
/home/jiguangshen/HPC_MachineLearning/MLcpp/lib/libneural_net.a: neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o
/home/jiguangshen/HPC_MachineLearning/MLcpp/lib/libneural_net.a: neural_network/CMakeFiles/neural_net.dir/layer.cpp.o
/home/jiguangshen/HPC_MachineLearning/MLcpp/lib/libneural_net.a: neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o
/home/jiguangshen/HPC_MachineLearning/MLcpp/lib/libneural_net.a: neural_network/CMakeFiles/neural_net.dir/build.make
/home/jiguangshen/HPC_MachineLearning/MLcpp/lib/libneural_net.a: neural_network/CMakeFiles/neural_net.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library /home/jiguangshen/HPC_MachineLearning/MLcpp/lib/libneural_net.a"
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && $(CMAKE_COMMAND) -P CMakeFiles/neural_net.dir/cmake_clean_target.cmake
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neural_net.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
neural_network/CMakeFiles/neural_net.dir/build: /home/jiguangshen/HPC_MachineLearning/MLcpp/lib/libneural_net.a
.PHONY : neural_network/CMakeFiles/neural_net.dir/build

neural_network/CMakeFiles/neural_net.dir/requires: neural_network/CMakeFiles/neural_net.dir/pch.cpp.o.requires
neural_network/CMakeFiles/neural_net.dir/requires: neural_network/CMakeFiles/neural_net.dir/neuron.cpp.o.requires
neural_network/CMakeFiles/neural_net.dir/requires: neural_network/CMakeFiles/neural_net.dir/layer.cpp.o.requires
neural_network/CMakeFiles/neural_net.dir/requires: neural_network/CMakeFiles/neural_net.dir/nnet.cpp.o.requires
.PHONY : neural_network/CMakeFiles/neural_net.dir/requires

neural_network/CMakeFiles/neural_net.dir/clean:
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network && $(CMAKE_COMMAND) -P CMakeFiles/neural_net.dir/cmake_clean.cmake
.PHONY : neural_network/CMakeFiles/neural_net.dir/clean

neural_network/CMakeFiles/neural_net.dir/depend:
	cd /home/jiguangshen/HPC_MachineLearning/MLcpp/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiguangshen/HPC_MachineLearning/MLcpp/src /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network /home/jiguangshen/HPC_MachineLearning/MLcpp/src /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network /home/jiguangshen/HPC_MachineLearning/MLcpp/src/neural_network/CMakeFiles/neural_net.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : neural_network/CMakeFiles/neural_net.dir/depend

