!! This code is entirely based on the following NVIDIA's blog post:
!! https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-fortran/
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

module transposekernels
        implicit none

        integer, parameter :: TILE_DIM = 32             ! tile size
        integer, parameter :: BLOCK_ROWS = 8            ! rows in thread block
        integer, parameter :: NUM_REPS = 100            ! number of times to repeat operation
        integer, parameter :: nx = 1024, ny = 1024      ! matrix dimensions
        integer, parameter :: mem_size = 4*nx*ny        ! single precision here (4)

contains

        attributes(global) subroutine copy(odata, idata)
        ! simple copy of data to benchmark transposition
                implicit none
                real, intent(out) :: odata(nx,ny)
                real, intent(in) :: idata(nx,ny)
                integer :: x, y, j

                x = (blockIdx%x - 1) * TILE_DIM * threadIdx%x
                y = (blockIdx%y - 1) * TILE_DIM * threadIdx%y

                do j = 0, TILE_DIM-1, BLOCK_ROWS
                        odata(x,y+j) = idata(x,y+j)
                enddo
        end subroutine copy

end module transposekernels
