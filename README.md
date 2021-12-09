# CUDA Fortran basics

This is a series of simple codes that perform some basic, but very instructive, operations on CUDA devices using CUDA Fortran. It is entirely based on a wonderful series of NVIDIA Developer's blog posts by Greg Ruetsch, which you can find [here](https://developer.nvidia.com/blog/easy-introduction-cuda-fortran/). The posts were published almost a decade ago. A lot of what is covered, especially regarding *global memory access*, has been largely optimized over the evolution of NVIDIA accelerators families. However, if you are interested in a quite detailed description of what is happening and how your computing instructions are executed by the device at the hardware level, these codes, along with the explanations in the blog, will demonstrate to be valuable content. Furthermore, it illustrates a lot of the CUDA Fortran features. 

I ran all of the codes on a V100-SMX2 32GB. I opted to manually type each one of them, adding several comments to places I found it useful. I also tried, at some point of this effort, to document the development by opening a thematic pull request and making as many commits as reason would allow, so that you can go over them and see what has changed from commit to commit.

The codes come in four different main topics, which are detailed below. I included ```Makefile```s, but they may not directly work for you. However, all you will need to compile the codes is the NVHPC Fortran compiler, ```nvfortran```. I have used v21.7. At some point, I also used NVIDIA's profiler, ```nvprof```, so you may very well want to have he whole HPC SDK stack available to you.



## [devicequery](./devicequery)
[devicequery.f90](/devicequery/devicequery.f90) is a minimal code that uses ```cudaGetDeviceProperties``` to list some properties of the device in qhich your code will be running.

## [metrics](./metrics)
Here there are a handful of direct implementations of the calculations of some performance metrics when executing CUDA kernels. For a complex application it will be very hard to tell exactly how many operations (read-write, fps and so on) your kernel is performing, for whcih situation a profiler is adequate. However, to discuss the basics, this is quite useful.

- [plain](./metrics/plain): contains a simple kernel that performs SAXPY. The performance of this kernel is then addressed over the remaining folders.
- [timing](./metrics/timing): two examples on how to time kernel executions. One uses Fortran's intrinsic clock, the other uses ```cudaEvents```.
- [bandwidth](./metrics/bandwidth): simple calculation of the bandwidth for a kernel execution (total number of operations, multiplied by the number of bits involved in each of them, divided by the kernel execution time).
- [gflops](./metrics/gflops): since the SAXPY kernel is a multiply-add operation, we can easily calculate the floating point operations per second achieved by the kernel from the previous bandwidth calculation.
- [errorcheck](./metrics/errorcheck): simple way to investigate possible errors over each call to CUDA Fortran library calls.

## [datatransfer](./datatransfer) 

## [memaccess](./memaccess) 
