cmake_minimum_required(VERSION 3.22)
project(YXCHF)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS ":  No build type selected, default is release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(UNIX)
  if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

# Enable OpenMP support for MSVC
if(MSVC)
  if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(YXCHF ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# Find and link Boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(YXCHF PUBLIC ${Boost_LIBRARIES})
