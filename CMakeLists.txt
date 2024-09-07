cmake_minimum_required(VERSION 3.10)

# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Project name
project(BeamformerProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA standard to C++17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Find Python 3
find_package(Python3 REQUIRED COMPONENTS Development)

# Include Python 3 and NumPy headers
include_directories(${Python3_INCLUDE_DIRS})
include_directories(/home/omkar/anaconda3/lib/python3.11/site-packages/numpy/core/include/)

# Set the FFTW3_DIR explicitly to the directory containing FFTW3Config.cmake
set(FFTW3_DIR "/usr/local/lib/cmake/fftw3")

# Find FFTW3 library
find_package(FFTW3 REQUIRED)

# Include FFTW3 headers
include_directories(${FFTW3_INCLUDE_DIRS})

link_directories("/usr/lib/gcc/x86_64-linux-gnu/10")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/10")

# Add the executable
add_executable(beamformer 
    beamformer.cu
    _readsignal.cpp
    _signalprocessing.cpp
)

# Link FFTW3 libraries
target_link_libraries(beamformer ${FFTW3_LIBRARIES})

# Link Python 3 libraries
target_link_libraries(beamformer ${Python3_LIBRARIES})

# Find Boost libraries (if not in a standard location, specify BOOST_ROOT)
find_package(Boost REQUIRED COMPONENTS filesystem system)

# Include Boost headers
include_directories(${Boost_INCLUDE_DIRS})

# Link Boost libraries
target_link_libraries(beamformer ${Boost_LIBRARIES})

# Link CUDA and cuFFT libraries
find_package(CUDA REQUIRED)
target_link_libraries(beamformer ${CUDA_LIBRARIES} cufft)
