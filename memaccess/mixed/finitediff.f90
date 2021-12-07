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
        real :: x(mx), y(my), x(mz)

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
end module derivative_m
