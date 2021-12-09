# CUDA Fortran basics

This is a series of simple codes that perform some basic, but very instructive, operations on CUDA devices using CUDA Fortran. It is entirely based on a wonderful series of NVIDIA Developer's blog posts by Greg Ruetsch, which you can find [here](https://developer.nvidia.com/blog/easy-introduction-cuda-fortran/). The posts were published almost a decade ago. A lot of what is covered, especially regarding *global memory access*, has been largely optimized over the evolution of NVIDIA accelerators families. However, if you are interested in a quite detailed description of what is happening and how your computing instructions are executed by the device at the hardware level, these codes, along with the explanations in the blog, will demonstrate to valuable content. Furthermore, it illustrates a lot of the CUDA Fortran features. 

I ran all of the codes on a V100-SMX2 32GB. I opted to manually type each one of them, adding several comments to places I found it useful. I also tried, at some point of this effort, to document the development by opening a thematic pull request and making as many commits as reason would allow, so that you can go over them and see what has changed from commit to commit.

The codes come in four different main topics, which are detailed below. I included ```Makefile```s, but they may not directly work for you. However, all you will need to compile the codes is the NVHPC Fortran compiler, ```nvfortran```. I have used v21.7. At some point, I also used NVIDIA's profiler, ```nvprof```, so you may very well want to have he whole HPC SDK stack available to you.



## [devicequery](./devicequery)


## [metrics](./metrics) 

## [datatransfer](./datatransfer) 

## [memaccess](./memaccess) 
