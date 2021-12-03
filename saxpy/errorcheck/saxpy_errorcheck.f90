!! This code is entirely based on the following NVIDIA blog:
!! https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-fortran/

module mathOps
contains
        attributes(global) subroutine saxpy(x, y, a)
                !!
                !! saxpy kernel performed in the GPU
                !! the attributes(global) qualifier
                !! differentiates this kernel from
                !! any host subroutines      
                !!
                implicit none
                !! no "device" is needed: everything within the 
                !! kernel resides in the device
                real :: x(:), y(:)
                !! a was never transferred, in host code
                !! Fortran passes arguments by reference,
                !! so here we must tell it to pass by value
                real, value :: a
                integer :: i, n
                n = size(x)
                !! Here the parallelization kicks in
                !! Each thread needs to know which element to get,
                !! as here we want each thread to process a single
                !! element.
                !! blockDim: equivalent to the thread block specified by host
                !! blockIdx, threadIx: identify block and thread within block
                !! The expression below generates a global index. Note:
                !! CUDA Fortran has unit offset for these.
                i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
                !! ensure we are not out-of-bounds
                if (i <= n) y(i) = y(i) + a*x(i)
        end subroutine saxpy
end module mathOps


program testSaxpy
        !!
        !! This is the host code
        !!
        use mathOps     ! contains saxpy kernel
        use cudafor     ! CUDA Fortran definitions
        implicit none
        integer, parameter :: N = 40000
        real :: x(N), y(N), a                   ! host arrays and scalar a
        real, device :: x_d(N), y_d(N)          ! device arrays
        type(dim3) :: grid, tBlock

        ! these will hold errors
        integer :: ierrSync, ierrAsync

        ! set the number of thread blocks to pass all N elements
        ! dim3 is a derived type for 3d structures (cuda grid)
        tBlock = dim3(256,1,1)
        grid = dim3(ceiling(real(N)/tBlock%x), 1, 1)

        ! set values in the host
        x = 1.0; y = 2.0; a = 2.0
        ! transfer data to device
        x_d = x
        y_d = y
        ! launch the saxpy kernel
        ! <<< r, s >>> is the execution config
        ! r: number of thread blocks in the grid
        ! s: number of threads in a block
        call saxpy<<<grid, tBlock>>>(x_d, y_d, a)
        ! check for errors
        ierrSync = cudaGetLastError()
        ierrAsync = cudaDeviceSynchronize()
        if (ierrSync /= cudaSuccess) then
                write(*,*) 'Sync kernel error: ', cudaGetErrorString(ierrSync)
        endif
        if (ierrAsync /= cudaSuccess) then
                write(*,*) 'Async kernel error: ', cudaGetErrorString(ierrAsync)
        endif 
        ! transfer result back to host
        y = y_d
        write(*,*) 'Max error: ', maxval(abs(y-4.0))
end program testSaxpy
