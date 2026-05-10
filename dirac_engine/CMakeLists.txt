cmake_minimum_required(VERSION 3.16)
project(DiracFX VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release builds get full optimisation; Debug gets sanitisers
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -fsanitize=address,undefined")

include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Shared library (for Python ctypes) ───────────────────────────────────────
add_library(dirac_fx SHARED src/capi.cpp)
target_include_directories(dirac_fx PRIVATE include)
set_target_properties(dirac_fx PROPERTIES
    OUTPUT_NAME "dirac_fx"
    PREFIX ""        # no "lib" prefix on some platforms
    POSITION_INDEPENDENT_CODE ON
)

# ── Static library ────────────────────────────────────────────────────────────
add_library(dirac_fx_static STATIC src/capi.cpp)
target_include_directories(dirac_fx_static PRIVATE include)

# ── Standalone demo executable ────────────────────────────────────────────────
add_executable(dirac_demo examples/dirac_demo.cpp)
target_include_directories(dirac_demo PRIVATE include)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS dirac_fx dirac_demo
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include/dirac_fx)

# ── Print summary ─────────────────────────────────────────────────────────────
message(STATUS "DiracFX ${PROJECT_VERSION}")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ std    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Shared lib : dirac_fx (for Python ctypes)")
message(STATUS "  Demo       : dirac_demo")
