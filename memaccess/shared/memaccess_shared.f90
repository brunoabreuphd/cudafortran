module sharedmemkernels
        implicit none
        integer, device :: n_d
contains
        attributes(global) subroutine staticReverse(d)
        ! store to shared mem, sync, load
                real :: d(:)
                integer :: t, tr
                real, shared :: s(64)    ! static block
                
                t = threadIdx%x
                tr = size(d) - t + 1
                ! store
                s(t) = d(t)
                ! sync
                call syncthreads()
                ! load
                d(t) = s(tr)
        end subroutine staticReverse

        attributes(global) subroutine dynamicReverse1(d)
        ! same as above, but s is not static
                real :: d(:)
                integer :: t, tr
                real, shared :: s(*)    ! dynamic block
                
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
                real, shared :: s(n_d)
                integer :: t, tr
                
                t = threadIdx%x
                tr = n_d - t + 1
                s(t) = d(t)
                call syncthreads()
                d(t) = s(tr)
        end subroutine dynamicReverse3

end module sharedmemkernels
