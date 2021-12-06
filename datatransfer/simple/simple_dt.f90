!!
!! This code is entirely base on the following NVIDIA blog:
!! https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-fortran/
!!
program profile
        use cudafor
        implicit none
        integer, parameter :: N = 1024
        real :: a(N,N)
        real, device :: a_d(N,N)

        ! give a some values
        a = 0
        ! transfer to device
        a_d = a
        ! transfer back to host
        a = a_d
end program profile
