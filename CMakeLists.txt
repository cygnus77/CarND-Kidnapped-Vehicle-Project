cmake_minimum_required(VERSION 2.8.11)
project(PARTICLE_FILTER)

# Build the particle filter project and solution.
# Use C++11
set(SRCS src/main.cpp src/particle_filter.cpp)
set_source_files_properties(${SRCS} PROPERTIES COMPILE_FLAGS "-std=c++0x -D_USE_MATH_DEFINES")

# Create the executable
add_executable(particle_filter ${SRCS})











