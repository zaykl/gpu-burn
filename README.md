# gpu-burn
Multi-GPU CUDA stress test(support int8 and float compare)
http://wili.cc/blog/gpu-burn.html

# Easy docker build and run

```
git clone https://github.com/zaykl/gpu-burn
cd gpu-burn
sudo docker build -t gpu_burn .
#burn with all gpu
sudo docker run --rm --gpus all gpu_burn
#burn with single gpu
sudo docker run --rm --gpus "device=0" gpu_burn
#interact with gpu_burn
sudo docker run -it --gpus all gpu_burn /bin/bash
#load with local data
sudo docker run -it -v ./:/data --gpus "device=0" gpu_burn /bin/bash
```

# Binary packages

https://repology.org/project/gpu-burn/versions

# Building
To build GPU Burn:

`make`

To remove artifacts built by GPU Burn:

`make clean`

GPU Burn builds with a default Compute Capability of 5.0.
To override this with a different value:

`make COMPUTE=<compute capability value>`

CFLAGS can be added when invoking make to add to the default
list of compiler flags:

`make CFLAGS=-Wall`

LDFLAGS can be added when invoking make to add to the default
list of linker flags:

`make LDFLAGS=-lmylib`

NVCCFLAGS can be added when invoking make to add to the default
list of nvcc flags:

`make NVCCFLAGS=-ccbin <path to host compiler>`

CUDAPATH can be added to point to a non standard install or
specific version of the cuda toolkit (default is 
/usr/local/cuda):

`make CUDAPATH=/usr/local/cuda-<version>`

CCPATH can be specified to point to a specific gcc (default is
/usr/bin):

`make CCPATH=/usr/local/bin`

CUDA_VERSION and IMAGE_DISTRO can be used to override the base
images used when building the Docker `image` target, while IMAGE_NAME
can be set to change the resulting image tag:

`make IMAGE_NAME=myregistry.private.com/gpu-burn CUDA_VERSION=12.0.1 IMAGE_DISTRO=ubuntu22.04 image`

# Usage

    GPU Burn
    Usage: gpu_burn [OPTIONS] [TIME]
    
    -m X   Use X MB of memory
    -m N%  Use N% of the available GPU memory
    -d     Use int8
    -tc    Try to use Tensor cores (if available)
    -l     List all GPUs in the system
    -i N   Execute only on GPU N
    -h     Show this help message
    
    Example:
    gpu_burn -d 3600
