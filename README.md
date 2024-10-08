# Beamformer++: Beamform an image in less than a second on RTX 4090!

## Requirements

* Cuda-toolkit
* FFTW3

    ```wget https://www.fftw.org/fftw-3.3.10.tar.gz```
    
    ```tar xf fftw-3.3.10.tar.gz```
    
    ```cd fftw-3.3.10/```
    
    ```./configure```
    
    ```make```
    
    ```sudo make install```

* Locate fftw3 cmake files. ```sudo find \ -name fftw3```. Mostly it will be here "/usr/local/lib/cmake/fftw3"

* Copy [FFTW3LibraryDepends.cmake](FFTW3LibraryDepends.cmake) to the above path or what ever path you have.

## Build the files

  ```cd Beamformer```
  
  ```mkdir build```
  
  ```cmake .. && make```
  
  ```./beamformer```

## Change configs if needed

* make changes in [auxilary.h](auxilary.h)

* This will create a file "output_image.csv" in the base folder

* Read it in python using numpy or opencv. Reshape it to (150x150)

* ![reconstructed_image.png](reconstructed_image.png)

## Usage with python3

* In the main folder, run ```python setup.py build_ext --inplace```
* In [setup.py](setup.py), look for ```sm``` number and change the number accordingly to your gpu [CUDA compute capability number](https://developer.nvidia.com/cuda-gpus). E.g. 4090 has sm number of 7.5, therefore ```sm_75```.
* Example is shown in [fastbeamform.ipynb](fastbeamform.ipynb)

## Comparison on RTX 4090

* Numpy Version: ```111.013301 seconds```
* Pytorch Version (CUDA Tensors): ```4.952884 seconds```
* Custom Beamforming CUDA kernel: ```634.817 ms```
