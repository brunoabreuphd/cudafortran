!!
!! This code is completely based on the following NIVIDIA blog:
!! https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-fortran/
!!
program pinned_dt
        use cudafor
        implicit none

        integer, parameter :: nElements = 40*1024*1024

        ! host arrays
        ! pageable
        real :: a_pag(nElements), b_pag(nElements)
        ! pinned
        real, allocatable, pinned :: a_pin(:), b_pin(:)

        ! device arrays
        real, device :: a_d(nElements)

        ! cuda events for timing
        type(cudaEvent) :: startEvent, stopEvent

        ! other variables
        type(cudaDeviceProp) :: prop
        real :: time
        integer :: istat, i
        logical :: pinnedFlag

        
        ! allocate and initialize arrays
        do i = 1, nElements
                a_pag(i) = i
        enddo
        b_pag = 0.0

        allocate(a_pin(nElements), b_pin(nElements), &
                STAT=istat, PINNED=pinnedFlag)

        if (istat /= 0) then
                write(*,*) 'Allocation of pinned arrays failed'
                pinnedFlag = .false.
        else
                if (.not. pinnedFlag) write(*,*) 'Pinned allocation failed'
        endif

        if (pinnedFlag) then
                a_pin = a_pag
                b_pin = 0.0
        endif

        istat = cudaEventCreate(startEvent)
        istat = cudaEventCreate(stopEvent)

        
        ! get some device props
        istat = cudaGetDeviceProperties(prop,0)
        write(*,*)
        write(*,*) 'Device: ', trim(prop%name)
        write(*,*) 'Transfer size (MB): ', 4*nElements / 1024. / 1024.


        ! do pageable data transfer
        write(*,*)
        write(*,*) 'Pageable transfers'
        ! host to device
        istat = cudaEventRecord(startEvent, 0)
        a_d = a_pag
        istat = cudaEventRecord(stopEvent, 0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        write(*,*) '  Host to Device bandwidth (GB/s): ', &
                nElements*4*1e-6/time
        ! device to host
        istat = cudaEventRecord(startEvent, 0)
        b_pag = a_d
        istat = cudaEventRecord(stopEvent, 0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        write(*,*) '  Device to Host bandwidth (GB/s): ', &
                nElements*4*1e-6/time
        ! check errors
        if ( any(a_pag /= b_pag)) then
                write(*,*) '*** Pageable transfers failed ***'
        endif


        ! do pinned data transfers
        if (pinnedFlag) then
                write(*,*)
                write(*,*) 'Pinned transfers'
                ! host to device
                istat = cudaEventRecord(startEvent, 0)
                a_d = a_pin
                istat = cudaEventRecord(stopEvent, 0)
                istat = cudaEventSynchronize(stopEvent)
                istat = cudaEventElapsedTime(time, startEvent, stopEvent)
                write(*,*) '  Host to Device bandwidth (GB/s): ', &
                        nElements*4*1e-6/time
                ! device to host
                istat = cudaEventRecord(startEvent, 0)
                b_pin = a_d
                istat = cudaEventRecord(stopEvent, 0)
                istat = cudaEventSynchronize(stopEvent)
                istat = cudaEventElapsedTime(time, startEvent, stopEvent)
                write(*,*) '  Device to Host bandwidth (GB/s): ', &
                        nElements*4*1e-6/time
                ! check errors
                if ( any(a_pin /= b_pin)) then
                        write(*,*) '*** Pinned transfers failed ***'
                endif
        endif

        ! cleanup
        if (allocated(a_pin)) deallocate(a_pin)
        if (allocated(b_pin)) deallocate(b_pin)
        istat = cudaEventDestroy(startEvent)
        istat = cudaEventDestroy(stopEvent)
        
end program pinned_dt
