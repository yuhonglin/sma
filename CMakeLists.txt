cmake_minimum_required (VERSION 2.6)

enable_language(Fortran)

project (sma)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE DEBUG)

set (CMAKE_CXX_STANDARD 11)

find_package( BLAS REQUIRED )

file(GLOB cppsource *.cpp)
file(GLOB hppsource *.hpp)
file(GLOB hsource *.h)
file(GLOB fsource *.f)

add_executable (sma ${cppsource} ${hppsource} ${hsource} ${fsource})
target_link_libraries(sma ${BLAS_LIBRARIES})
