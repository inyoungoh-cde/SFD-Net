# LGD - geometric feature extractor
#
# Cross-platform build (Linux/GCC, Windows/MSVC, macOS/Clang).
# On Windows the hand-maintained LGD.sln / LGD.vcxproj can be used instead;
# both build systems compile the exact same sources.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/LGD data output 40

cmake_minimum_required(VERSION 3.10)
project(LGD LANGUAGES CXX)

# Default to an optimized build: the Visual Studio solution only ships a
# Release configuration, so the CMake build matches it when nothing is given.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(LGD
	main.cpp
	runner.cpp
	algorithms.h
	FileData.h
	platform.h
	runner.h
)

# Header-only third-party libraries (Eigen, nanoflann)
target_include_directories(LGD PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party_includes")

find_package(OpenMP REQUIRED)
target_link_libraries(LGD PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
	target_compile_definitions(LGD PRIVATE _CRT_SECURE_NO_WARNINGS)
	target_compile_options(LGD PRIVATE /W3)
else()
	target_compile_options(LGD PRIVATE -Wall)
endif()

# GCC 8 still keeps std::filesystem in a separate library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
	target_link_libraries(LGD PRIVATE stdc++fs)
endif()

message(STATUS "LGD: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, build type ${CMAKE_BUILD_TYPE}")
