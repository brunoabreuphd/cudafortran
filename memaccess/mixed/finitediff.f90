!! This code is entirely based on the following NVIDIA's blog post:
!! https://developer.nvidia.com/blog/finite-difference-methods-cuda-fortran-part-1/
!! As a requirement, reprocued below is the License header
!!
! Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!  * Redistributions of source code must retain the above copyright
!    notice, this list of conditions and the following disclaimer.
!  * Redistributions in binary form must reproduce the above copyright
!    notice, this list of conditions and the following disclaimer in the
!    documentation and/or other materials provided with the distribution.
!  * Neither the name of NVIDIA CORPORATION nor the names of its
!    contributors may be used to endorse or promote products derived
!    from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
! EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
! PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
! OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


module derivative_m
        use cudafor

        ! set cubic grid and arrays
        integer, parameter :: mx = 64, my = 64, mz = 64
        real :: x(mx), y(my), z(mz)

        ! shared memory tiles are (m*, *Pencils)
        ! sPencils: each thread calculates derivative at one point
        ! lPencils: coalescing in y,z, each thread has to calculate
        ! derivative at multiple points
        integer, parameter :: sPencils = 4
        integer, parameter :: lPencils = 32

        ! grids and blocks for each type of Pencil
        type(dim3) :: grid_sp(3), block_sp(3)
        type(dim3) :: grid_lp(3), block_lp(3)

        ! stencil coefficients (eight-order here)
        real, constant :: ax_c, bx_c, cx_c, dx_c

contains
        
        subroutine setDerivativeParameters()
        ! host code to set constant data
                implicit none

                real :: dsinv           ! stencil pre-coefficient
                integer :: i, j, k      ! grid index helpers

                ! make sure an integer number of Pencils fit in the grid
                if (mod(my,sPencils) /= 0) then
                        write(*,*) 'my must be a multiple of sPencils (x-derivative)'
                        stop
                endif
                if (mod(mx, sPencils) /= 0) then
                        write(*,*) 'mx must be a multiple of sPencils (y-derivative)'
                        stop
                endif
                if (mod(mz, sPencils) /= 0) then
                        write(*,*) 'mz must be a multiple of sPencils (z-derivative)'
                        stop
                endif
                if (mod(my,lPencils) /= 0) then
                        write(*,*) 'my must be a multiple of lPencils'
                        stop
                endif
                if (mod(mx, lPencils) /= 0) then
                        write(*,*) 'mx must be a multiple of lPencils'
                        stop
                endif
                if (mod(mz, lPencils) /= 0) then
                        write(*,*) 'mz must be a multiple of lPencils'
                        stop
                endif

                ! stencil weights
                dsinv = real(mx - 1)
                do i = 1, mx
                        x(i) = (i-1.0)/(mx-1.0)
                enddo
                ax_c = 4.0 / 5.0 * dsinv
                bx_c = -1.0 / 5.0 * dsinv
                cx_c = 4.0 / 105.0 * dsinv
                dx_c = -1.0 / 280.0 * dsinv

                ! configure grid and blocks for the Pencils
                grid_sp(1) = dim3(my/sPencils, mz, 1)
                block_sp(1) = dim3(mx, sPencils, 1)
                grid_lp(1) = dim3(my/lPencils, mz, 1)
                block_lp(1) = dim3(mx, lPencils, 1)

                grid_sp(2) = dim3(mx/sPencils, mz, 1)
                block_sp(2) = dim3(sPencils, my, 1)
                grid_lp(2) = dim3(mx/lPencils, mz, 1)
                ! we want to use the same number of threads as for (1)
                ! so if using lPencils instead of sPencils in one dimension
                ! we multiply the other by the rescaling factor
                block_lp(2) = dim3(lPencils, my*sPencils/lPencils, 1)   

                grid_sp(3) = dim3(mx/sPencils, my, 1)
                block_sp(3) = dim3(sPencils, mz, 1)
                grid_lp(3) = dim3(mx/lPencils, my, 1)
                block_lp(3) = dim3(lPencils, mz*sPencils/lPencils, 1)   

        end subroutine setDerivativeParameters


        attributes(global) subroutine derivative_x(f, df)
        ! kernel to calculate x derivatives
                implicit none
                ! function and derivative
                real, intent(in) :: f(mx,my,mz)
                real, intent(out) :: df(mx,my,mz)
                ! pencil 
                real, shared :: f_s(-3:mx+4, sPencils)
                integer :: i, j, k, j_l

                ! find the index for each thread to work on
                i = threadIdx%x
                j = (blockIdx%x-1)*blockDim%y + threadIdx%y
                ! j_l is the shared memory version of j
                j_l = threadIdx%y
                k = blockIdx%y

                f_s(i, j_l) = f(i,j,k)

                call syncthreads()

                ! periodic images to shared memory array
                if (i <= 4) then
                        f_s(i-4, j_l) = f_s(mx+i-5, j_l)
                        f_s(mx+i, j_l) = f_s(i+1, j_l)
                endif

                call syncthreads()

                ! derivative expression
                df(i,j,k) = &
                        (ax_c*( f_s(i+1,j_l) - f_s(i-1,j_l) )) &
                        + bx_c*( f_s(i+2,j_l) - f_s(i-2,j_l) ) &
                        + cx_c*( f_s(i+3,j_l) - f_s(i-3,j_l) ) &
                        + dx_c*( f_s(i+4,j_l) - f_s(i-4,j_l) ) 

        end subroutine derivative_x

        attributes(global) subroutine derivative_x_lPencils(f, df)
        ! this is another version of the derivative for lPencil, 64x32 tiles
        ! with the same number of threads as before
                implicit none
                real, intent(in) :: f(mx,my,mz)
                real, intent(out) :: df(mx,my,mz)
                real, shared :: f_s(-3:mx+4, lPencils)
                integer :: i, j, k, j_l, jBase

                i = threadIdx%x
                jBase = (blockIdx%x-1) * lPencils
                k = blockIdx%y

                do j_l = threadIdx%y, lPencils, blockDim%y
                        j = jBase + j_l
                        f_s(i, j_l) = f(i,j,k)
                enddo
                call syncthreads()

                if(i <= 4) then
                        do j_l = threadIdx%y, lPencils, blockDim%y
                                f_s(i-4, j_l) = f_s(mx+i-5, j_l)
                                f_s(mx+i, j_l) = f_s(i+1, j_l)
                        enddo
                endif
                call syncthreads()

                do j_l = threadIdx%y, lPencils, blockDim%y
                        j = jBase + j_l
                        df(i,j,k) = &
                                (ax_c*( f_s(i+1,j_l) - f_s(i-1,j_l) )) &
                                + bx_c*( f_s(i+2,j_l) - f_s(i-2,j_l) ) &
                                + cx_c*( f_s(i+3,j_l) - f_s(i-3,j_l) ) &
                                + dx_c*( f_s(i+4,j_l) - f_s(i-4,j_l) ) 
                enddo

        end subroutine derivative_x_lPencils
                        

end module derivative_m


program finitediff
        use cudafor
        use derivative_m
        implicit none

        real, parameter :: fx = 1.0, fy = 1.0, fz = 1.0
        integer, parameter :: nReps = 20
        ! allocate host and device arrays
        real :: f(mx,my,mz), df(mx,my,mz), sol(mx,my,mz)
        real, device :: f_d(mx,my,mz), df_d(mx,my,mz)
        ! cuda Events and props
        type(cudaEvent) :: startEvent, stopEvent
        type(cudaDeviceProp) :: prop
        ! aux variables
        real :: twopi, error, maxError
        real :: time
        integer :: i, j, k, istat


        ! print device props
        istat = cudagetDeviceProperties(prop, 0)
        write(*, "(/, 'Device Name: ',a)") trim(prop%name)
        write(*, "('Compute capability: ', i0, '.', i0)") prop%major, prop%minor

        ! initialize constants
        twopi = 8.0 * atan(1.d0)
        call setDerivativeParameters()

        ! create cuda events
        istat = cudaEventCreate(startEvent)
        istat = cudaEventCreate(stopEvent)


        !! X-DERIVATIVE USING 64x4 TILE (sPencil)
        write(*, "(/,'x derivatives')")
        ! give values to the function
        do i = 1, mx
                f(i,:,:) = cos(fx*twopi*(i-1.0)/(mx-1))
        enddo
        ! copy values from host to device
        f_d = f
        df_d = 0.0
        ! warm up the kernel
        call derivative_x<<<grid_sp(1), block_sp(1)>>>(f_d, df_d)
        ! launch kernel repeatedly
        istat = cudaEventRecord(startEvent, 0)
        do i = 1, nReps
                call derivative_x<<<grid_sp(1), block_sp(1)>>>(f_d, df_d)
        enddo
        istat = cudaEventRecord(stopEvent, 0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        ! copy differential from device to host
        df = df_d
        ! write analytical solution
        do i = 1, mx
                sol(i,:,:) = -fx*twopi*sin(fx*twopi*(i-1.0)/(mx-1))
        enddo
        ! compare numerical to analytical
        error = sqrt(sum((sol - df)**2)/(mx*my*mz))
        maxError = maxval(abs(sol - df))
        ! print results
        write(*,"(/,' Using shared memory tile of x=', i0, ', y=', i0)") &
                        mx, sPencils
        write(*,*) ' RMS error: ', error
        write(*,*) ' Max error: ', maxError
        write(*,*) ' Avg execution time (ms): ', time/nReps
        write(*,*) ' Avg Bandwidth (GB/s): ', 2.0*1000*sizeof(f)/(1024**3 * time/nReps)
        

        !! X-DERIVATIVE USING EXTENDED TILE (lPencils))
        do i = 1, mx
                f(i,:,:) = cos(fx*twopi*(i-1.0)/(mx-1))
        enddo
        f_d = f
        df_d = 0.0
        call derivative_x_lPencils<<<grid_lp(1), block_lp(1)>>>(f_d, df_d)
        istat = cudaEventRecord(startEvent, 0)
        do i = 1, nReps
                call derivative_x_lPencils<<<grid_lp(1), block_lp(1)>>>(f_d, df_d)
        enddo
        istat = cudaEventRecord(stopEvent, 0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        df = df_d
        do i = 1, mx
                sol(i,:,:) = -fx*twopi*sin(fx*twopi*(i-1.0)/(mx-1))
        enddo
        error = sqrt(sum((sol - df)**2)/(mx*my*mz))
        maxError = maxval(abs(sol - df))
        write(*,"(/,' Using shared memory tile of x=', i0, ', y=', i0)") &
                        mx, lPencils
        write(*,*) ' RMS error: ', error
        write(*,*) ' Max error: ', maxError
        write(*,*) ' Avg execution time (ms): ', time/nReps
        write(*,*) ' Avg Bandwidth (GB/s): ', 2.0*1000*sizeof(f)/(1024**3 * time/nReps)


        ! cleanup
        istat = cudaEventDestroy(startEvent)
        istat = cudaEventDestroy(stopEvent)

end program finitediff
