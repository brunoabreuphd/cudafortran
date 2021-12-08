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
                integer :: i

                i = 1 + (blockDim%x*(blockIdx%x-1) + threadIdx%x) * s
                a(i) = a(i) + 1
        end subroutine stride

end module memkernels


program offsetAndStride
        use cudafor
        use memkernels
        implicit none
        
        ! size integers
        integer, parameter :: nMB = 4 
        integer, parameter :: blockSize = 256
        integer :: n
        ! device array
        real(fp_kind), device, allocatable :: a_d(:)
        ! CUDA event, device properties
        type(cudaEvent) :: startEvent, stopEvent
        type(cudaDeviceProp) :: prop
        ! aux variables
        real :: time
        integer :: istat, i

        ! get device props
        istat = cudaGetDeviceProperties(prop, 0)
        write(*,'(/,"Device: ",a)') 'Device: ', trim(prop%name)
        write(*,'("Transfer size (MB): ",i0)') nMB

        ! print out precision
        if(kind(a_d) == singlePrecision) then
                write(*,'(a,/)') 'Single Precision'
        else
                write(*,'(a,/)') 'Double Precision'
        endif

        ! allocate device array
        n = nMB * 1024 * 1024 / fp_kind
        allocate(a_d(n*33))

        ! create cuda events
        istat = cudaEventCreate(startEvent)
        istat = cudaEventCreate(stopEvent)

        
        ! OFFSET KERNEL
        write(*,*) 'Offset, Bandwidth (GB/s):'
        call offset<<<n/blockSize, blockSize>>>(a_d,0)
        do i = 0, 32
                a_d = 0.0
                istat = cudaEventRecord(startEvent,0)
                call offset<<<n/blockSize, blockSize>>>(a_d, i)
                istat = cudaEventRecord(stopEvent,0)
                istat = cudaEventSynchronize(stopEvent)
                istat = cudaEventElapsedTime(time, startEvent, stopEvent)
                ! print bw   
                write(*,*) i, 2*nMB/time*(1.e+3/1024)
        enddo
        write(*,*)


        ! STRIDE KERNEL
        write(*,*) 'Stride, Bandwidth (GB/s):'
        call stride<<<n/blockSize, blockSize>>>(a_d,1)
        do i = 1, 32
                a_d = 0.0
                istat = cudaEventRecord(startEvent,0)
                call stride<<<n/blockSize, blockSize>>>(a_d, i)
                istat = cudaEventRecord(stopEvent,0)
                istat = cudaEventSynchronize(stopEvent)
                istat = cudaEventElapsedTime(time, startEvent, stopEvent)
                ! print bw   
                write(*,*) i, 2*nMB/time*(1.e+3/1024)
        enddo
        write(*,*)
        
        

        ! clean up
        istat = cudaEventDestroy(startEvent)
        istat = cudaEventDestroy(stopEvent)
        deallocate(a_d)

end program offsetAndStride

