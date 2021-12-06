!! This code is entirely based on the following NVIDIA blog post:
!! https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-fortran-kernels/

module memkernels
        ! here we define floating point precisions
        integer, parameter :: singlePrecision = kind(0.0)
        integer, parameter :: doublePrecision = kind(0.0d0)
        ! here we choose one of them to use
        integer, parameter :: fp_kind = singlePrecision
contains
        attributes(global) subroutine offset(a, s)
        ! offsets array element by 1
                real(fp_kind) :: a(*)
                integer, value :: s
                integer :: i

                i = blockDim%x*(blockIdx%x-1) + threadIdx%x + s
                a(i) = a(i) + 1
        end subroutine offset

        attributes(global) subroutine stride(a, s)
        ! strides array element by 1
                real(fp_kind) :: a(*)
                integer, value :: s
                integer :: :: i

                i = 1 + (blockDim%x*(blockIdx%x-1) + threadIdx%x) * s
                a(i) = a(i) + 1
        end subroutine stride

end module memkernels

