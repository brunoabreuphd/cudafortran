!! This code is entirely based on the following NVIDIA blog post:
!! https://developer.nvidia.com/blog/using-shared-memory-cuda-fortran/
!!
module sharedmemkernels
        implicit none
        integer, device :: n_d
contains
        ! kernels to reverse the order of an array
        ! t is the direct order index
        ! tr is the reversed order index
        attributes(global) subroutine staticReverse(d)
        ! store to shared mem, sync, load
                real :: d(:)
                integer :: t, tr
                real, shared :: s(64)    ! static block
                
                t = threadIdx%x
                tr = size(d) - t + 1
                ! store from global to shared mem
                s(t) = d(t)
                ! sync: make sure all threads have completed
                call syncthreads()
                ! load from shared to global
                d(t) = s(tr)
        end subroutine staticReverse

        attributes(global) subroutine dynamicReverse1(d)
        ! same as above, but s is not static
                real :: d(:)
                integer :: t, tr
                ! dynamic allocation: size is implicitly determined
                ! from the third execution config parameter
                ! when the kernel is launched
                real, shared :: s(*)
                
                t = threadIdx%x
                tr = size(d) - t + 1
                s(t) = d(t)
                call syncthreads()
                d(t) = s(tr)
        end subroutine dynamicReverse1

        attributes(global) subroutine dynamicReverse2(d, nSize)
        ! same as before, s size comes from the call
                real :: d(nSize)
                integer, value :: nSize
                integer :: t, tr
                ! dynamic block: nSize is used to declare the size and
                ! it comes from parameters in the kernel call
                real, shared :: s(nSize)

                t = threadIdx%x
                tr = nSize - t + 1
                s(t) = d(t)
                call syncthreads()
                d(t) = s(tr)
        end subroutine dynamicReverse2
                         
        attributes(global) subroutine dynamicReverse3(d)
        ! same, s size comes from device-stored value
                real :: d(n_d)
                ! dynamic block: n_d comes from a device-stored
                ! variable that is copyed in host code
                real, shared :: s(n_d)
                integer :: t, tr
                
                t = threadIdx%x
                tr = n_d - t + 1
                s(t) = d(t)
                call syncthreads()
                d(t) = s(tr)
        end subroutine dynamicReverse3

end module sharedmemkernels


program sharedmem
        use cudafor
        use sharedmemkernels
        implicit none
        ! size of arrays
        integer, parameter :: n = 64
        ! host and device arrays
        real :: a(n), r(n), d(n)
        real, device :: d_d(n)
        ! cuda thread blocks parameters
        type(dim3) :: grid, threadBlock
        ! aux variables
        integer :: i

        
        ! defining the grid
        threadBlock = dim3(n,1,1)
        grid = dim3(1,1,1)
        ! assign array values
        do i = 1, n
                a(i) = i
                r(i) = n - i + 1
        enddo


        ! STATIC SHARED MEMORY
        ! get data into device
        d_d = a
        ! launch static kernel
        call staticReverse<<<grid, threadBlock>>>(d_d)
        ! copy data back to host
        d = d_d
        write(*,*) 'Static case max error: ', maxval(abs(r-d))
        write(*,*)


        ! DYNAMIC SHARED MEM V1 
        d_d = a
        ! the third argument in the kernel here is the amount 
        ! of shared memory invoked (in bytes)
        call dynamicReverse1<<<grid, threadBlock, 4*threadBlock%x>>>(d_d)
        d = d_d
        write(*,*) 'Dynamic case V1 max error: ', maxval(abs(r-d))
        write(*,*)


        ! DYNAMIC SHARED MEM V2 
        d_d = a
        call dynamicReverse2<<<grid, threadBlock, 4*threadBlock%x>>>(d_d,n)
        d = d_d
        write(*,*) 'Dynamic case V2 max error: ', maxval(abs(r-d))
        write(*,*)


        ! DYNAMIC SHARED MEM V3
        n_d = n
        d_d = a
        call dynamicReverse3<<<grid, threadBlock, 4*threadBlock%x>>>(d_d)
        d = d_d
        write(*,*) 'Dynamic case V3 max error: ', maxval(abs(r-d))
        write(*,*)

end program sharedmem
