cmake_minimum_required(VERSION 3.20)

project(periodic_heat)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-math-errno -ffinite-math-only")

find_package(OpenMP    REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(VTK       REQUIRED COMPONENTS IOImage IOXML)

#   FFTW3 CMake is broken (https://github.com/FFTW/fftw3/issues/130), need pkg-config 
pkg_check_modules(FFTW IMPORTED_TARGET REQUIRED fftw3)

#   pkg-config is missing the fftw3_threads library
find_library(FFTW_DOUBLE_THREADS_LIB NAMES "fftw3_threads" PATHS ${PKG_FFTW_LIBRARY_DIRS} REQUIRED)

include_directories(PkgConfig::FFTW)

add_executable(solver_sine src/solver_sine.cc)
add_executable(solver_step src/solver_step.cc)

target_link_libraries(solver_sine PRIVATE OpenMP::OpenMP_CXX ${VTK_LIBRARIES} PkgConfig::FFTW)
target_link_libraries(solver_step PRIVATE OpenMP::OpenMP_CXX ${VTK_LIBRARIES} PkgConfig::FFTW ${FFTW_DOUBLE_THREADS_LIB})
vtk_module_autoinit(TARGETS solver_sine solver_step MODULES ${VTK_LIBRARIES})
