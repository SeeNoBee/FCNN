cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

project(FCNN CXX)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(FCNN src/main.cpp)

target_include_directories(FCNN PUBLIC include)

install(TARGETS FCNN
RUNTIME DESTINATION bin)

option(FCNN_INSTALL_MNIST_SCRIPT "Copy MNIST dowload script to installation directory" TRUE)
if (FCNN_INSTALL_MNIST_SCRIPT)
    install(FILES MNIST_download.py DESTINATION bin)
endif()

option(FCNN_INSTALL_MNIST "Download MNIST dataset to installation directory" FALSE)
if(FCNN_INSTALL_MNIST)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(FCNN_MNIST_TRAIN_IMAGES_URL http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz CACHE STRING "")
    set(FCNN_MNIST_TRAIN_LABELS_URL http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz CACHE STRING "")
    set(FCNN_MNIST_TEST_IMAGES_URL http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz CACHE STRING "")
    set(FCNN_MNIST_TEST_LABELS_URL http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz CACHE STRING "")
    configure_file(MNISTDownload.cmake.temp MNISTDownload.cmake)
    install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/MNISTDownload.cmake)
endif()