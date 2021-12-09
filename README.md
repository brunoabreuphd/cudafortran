# CUDA Fortran basics

This is a series of simple codes that perform some basic, but very instructive, operations on CUDA devices using CUDA Fortran. It is entirely based on a wonderful series of NVIDIA Developer's blog posts by Greg Ruetsch, which you can find [here](https://developer.nvidia.com/blog/easy-introduction-cuda-fortran/). The posts were published almost a decade ago (as of December 2021). A lot of what is covered, especially regarding *global memory access*, has been largely optimized over the evolution of NVIDIA accelerators families. However, if you are interested in a quite detailed description of what is happening and how your computing instructions are executed by the device at the hardware level, these codes, along with the explanations in the blog, will demonstrate to be valuable content. Furthermore, it illustrates a lot of the CUDA Fortran features. 

I ran all the codes on a V100-SMX2 32GB. I opted to manually type each one of them, adding several comments to places I found it useful. I also tried, at some point of this effort, to document the development by opening a thematic pull request and making as many commits as reason would allow, so that you can go over them and see what has changed from commit to commit.

The codes come in four different main topics, which are detailed below. I included ```Makefile```s, but they may not directly work for you. However, all you will need to compile the codes is the NVHPC Fortran compiler, ```nvfortran```. I have used v21.7. At some point, I also used NVIDIA's profiler, ```nvprof```, so you may very well want to have he whole HPC SDK stack available to you.



## [devicequery](./devicequery)
[devicequery.f90](/devicequery/devicequery.f90) is a minimal code that uses ```cudaGetDeviceProperties``` to list some properties of the device in qhich your code will be running. PR [#3](https://github.com/babreu-ncsa/cudafortran/pull/3) added this code to the repository.

## [metrics](./metrics)
Here there are a handful of direct implementations of the calculations of some performance metrics when executing CUDA kernels. For a complex application it will be very hard to tell exactly how many operations (read-write, fps and so on) your kernel is performing, for whcih situation a profiler is adequate. However, to discuss the basics, this is quite useful.

- [plain](./metrics/plain): contains a simple kernel that performs SAXPY. The performance of this kernel is then addressed over the remaining folders.
- [timing](./metrics/timing): two examples on how to time kernel executions. One uses Fortran's intrinsic clock, the other uses ```cudaEvents```.
- [bandwidth](./metrics/bandwidth): simple calculation of the bandwidth for a kernel execution (total number of operations, multiplied by the number of bits involved in each of them, divided by the kernel execution time). See PR [#1](https://github.com/babreu-ncsa/cudafortran/pull/1) for development.
- [gflops](./metrics/gflops): since the SAXPY kernel is a multiply-add operation, we can easily calculate the floating point operations per second achieved by the kernel from the previous bandwidth calculation. See PR [#2](https://github.com/babreu-ncsa/cudafortran/pull/2) for develoment.
- [errorcheck](./metrics/errorcheck): simple way to investigate possible errors over each call to CUDA Fortran library calls. See PR [#4](https://github.com/babreu-ncsa/cudafortran/pull/4) for development.

## [datatransfer](./datatransfer)
- [simple](./datatransfer/simple): this is a simple code to send data from host to device, and then back from device to host.
- [pinned](./datatransfer/pinned): here, the bandwidth for two types of datatransfer between host and device is calculated. One uses host-pageable memory, the other uses host-pinned-memory.
- [overlap](./datatransfer/overlap): data transfers are, by default, synchronous operations, while kernel executions are asynchronous operations. This code explores the concept of *CUDA streams* to try to take advantage of these features, achieving high performance by overlapping data transfers with kernel executions in a controlled fashion.

For the applications resulting from source codes in this folder, it is interesting to see how much time is being spent in data transfers by using ```nvprof```. The development is somehow documented on PR [#5](https://github.com/babreu-ncsa/cudafortran/pull/5).

## [memaccess](./memaccess)
In similarity to the host (CPU) memory hierarchy, the device memory also has a structure that, if explored correctly, can yield significant performance improvements to CUDA kernels. The development of the codes below can be tracked on PR [#6](https://github.com/babreu-ncsa/cudafortran/pull/6).
### [global](./memaccess/global)
This is a simple demonstration of how accessing contiguous (using an *offset* kernel) and non-contiguous (using a *stride* kernel) memory can impact performance.

### [shared](./memaccess/shared)
CUDA thread blocks have much faster access to their local, shared memory (compared to the device's global memory). This folder contains two codes that show how to allocate data into these caches and how to access them.
- [simple](./memaccess/shared/simple): this shows a few different ways on how shared memory can be employed and compares their bandwidth.
- [transposemat](./memaccess/shared/transposemat): an application of the previous concepts where matrix tranposition achieves almost-ideal bandwidth.

### [mixed](./memaccess/mixed)
Of course that, in a reallistic scenario, one will need to deal with data living and being transferred from and to both shared and global memory. The modern CUDA versions indeed do a very good job on managing this process efficiently. However, to obtain the most of performance, some tuning can be very useful. 

This folder contains an application that implements numerical derivatives using the Finite Difference method on a cubic grid. It is demonstrated that performance can be affected by the way the tiling decomposition (using stencils) is performed.
