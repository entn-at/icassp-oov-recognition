cmake_minimum_required(VERSION 3.9.5)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-Ofast ")

project(condutor)

include_directories("/remote/idiap.svm/temp.speech01/rbraun/code/kaldi/tools/openfst-1.6.7/include")
include_directories("libs/")

add_subdirectory(libs/pybind11)
pybind11_add_module(fast libs/fast.cc libs/fst-wrapper.cc)

set_target_properties(fast PROPERTIES LIBRARY_OUTPUT_NAME "fast")

target_link_libraries(fast PRIVATE "-L/remote/idiap.svm/temp.speech01/rbraun/code/kaldi/tools/openfst-1.6.7/lib" -lfstscript -lfst)
