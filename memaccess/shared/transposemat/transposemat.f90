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

                x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
                y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

                do j = 0, TILE_DIM-1, BLOCK_ROWS
                        odata(x,y+j) = idata(x,y+j)
                enddo
        end subroutine copy

        attributes(global) subroutine copySharedMem(odata, idata)
        ! copy data using shared mem
                implicit none
                real, intent(out) :: odata(nx,ny)
                real, intent(in) :: idata(nx,ny)
                real, shared :: tile(TILE_DIM,TILE_DIM)
                integer :: x, y, j

                x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
                y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

                ! copy data to tile
                do j = 0, TILE_DIM-1, BLOCK_ROWS
                        tile(threadIdx%x,threadIdx%y+j) = idata(x,y+j)
                enddo
                ! sync threads
                call syncthreads()
                ! copy data from tile
                do j = 0, TILE_DIM-1, BLOCK_ROWS
                        odata(x,y+j) = tile(threadIdx%x,threadIdx%y+j)
                enddo
        end subroutine copySharedMem

        attributes(global) subroutine transposeNaive(odata, idata)
        ! direct, simple transposition
                implicit none
                real, intent(out) :: odata(nx,ny)
                real, intent(in) :: idata(nx,ny)
                integer :: x, y, j

                x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
                y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

                do j = 0, TILE_DIM-1, BLOCK_ROWS
                        odata(y+j,x) = idata(x,y+j)
                enddo
        end subroutine transposeNaive

        attributes(global) subroutine transposeCoalesced(odata, idata)
        ! transpose using shared memory
        ! this makes threads in a warp access global memory at the same time
                implicit none
                real, intent(out) :: odata(nx,ny)
                real, intent(in) :: idata(nx,ny)
                real, shared :: tile(TILE_DIM,TILE_DIM)
                integer :: x, y, j

                x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
                y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

                ! copy data to tile
                do j = 0, TILE_DIM-1, BLOCK_ROWS
                        tile(threadIdx%x,threadIdx%y+j) = idata(x,y+j)
                enddo
                ! sync threads
                call syncthreads()

                x = (blockIdx%y - 1) * TILE_DIM + threadIdx%x
                y = (blockIdx%x - 1) * TILE_DIM + threadIdx%y

                ! copy data from tile
                do j = 0, TILE_DIM-1, BLOCK_ROWS
                        odata(x,y+j) = tile(threadIdx%y+j, threadIdx%x)
                enddo
        end subroutine transposeCoalesced

        attributes(global) subroutine transposeNoBanksConflict(odata, idata)
        ! transpose using shared memory
        ! this makes threads in a warp access global memory at the same time
        ! but now without mem banks conflict
                implicit none
                real, intent(out) :: odata(nx,ny)
                real, intent(in) :: idata(nx,ny)
                ! padding the tile eliminates mem bank conflicts
                real, shared :: tile(TILE_DIM+1,TILE_DIM)
                integer :: x, y, j

                x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
                y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

                ! copy data to tile
                do j = 0, TILE_DIM-1, BLOCK_ROWS
                        tile(threadIdx%x,threadIdx%y+j) = idata(x,y+j)
                enddo
                ! sync threads
                call syncthreads()

                x = (blockIdx%y - 1) * TILE_DIM + threadIdx%x
                y = (blockIdx%x - 1) * TILE_DIM + threadIdx%y

                ! copy data from tile
                do j = 0, TILE_DIM-1, BLOCK_ROWS
                        odata(x,y+j) = tile(threadIdx%y+j, threadIdx%x)
                enddo
        end subroutine transposeNoBanksConflict

end module transposekernels


program transposemat
        use cudafor
        use transposekernels
        implicit none
        ! grid and block
        type(dim3) :: dimGrid, dimBlock
        ! cuda Events to track time
        type(cudaEvent) :: startEvent, stopEvent
        real :: time
        ! cuda Device props
        type(cudaDeviceProp) :: prop
        ! host arrays
        real :: h_idata(nx,ny), h_cdata(nx,ny), h_tdata(ny,nx), gold(ny,nx)
        ! device arrays
        real, device :: d_idata(nx,ny), d_cdata(nx,ny), d_tdata(ny,nx)
        ! aux integers
        integer :: i, j, istat

        ! check if nx,ny is a multiple of TILE_DIM
        if (mod(nx, TILE_DIM) /= 0 .or. mod(ny, TILE_DIM) /= 0) then
                write(*,*) 'nx, ny must be a multiple of TILE_DIM'
                stop
        endif
        ! check if TIME_DIM is a multiple of BLOCK_ROWS
        if (mod(TILE_DIM, BLOCK_ROWS) /= 0) then
                write(*,*) 'TILE_DIM must be a multiple of BLOCK_ROWS'
                stop
        endif

        ! define grid and blocks
        dimGrid = dim3(nx/TILE_DIM, ny/TILE_DIM, 1)
        dimBlock = dim3(TILE_DIM, BLOCK_ROWS, 1)

        ! print out device and selected parameters
        istat = cudaGetDeviceProperties(prop, 0)
        write(*, '("Device: ", a)') trim(prop%name)
        write(*, '("Matrix size: ", i5, i5)') nx, ny
        write(*, '("Block size: ", i3, i3)') TILE_DIM, BLOCK_ROWS
        write(*, '("Tile size: ", i3, i3)') TILE_DIM, TILE_DIM
        write(*, '("CUDA Grid: ", i4, i4, i4)') dimGrid%x, dimGrid%y, dimGrid%z
        write(*, '("Block dims: ", i4, i4, i4)') dimBlock%x, dimBlock%y, dimBlock%z

        ! give values to host arrays
        do j = 1, ny
                do i = 1, nx
                        h_idata(i,j) = i + (j-1)*nx
                enddo
        enddo
        ! transpose it using Fortran implicit routine
        gold = transpose(h_idata)

        ! send it to device
        d_idata = h_idata
        d_tdata = 0.0
        d_cdata = 0.0

        ! create cuda events
        istat = cudaEventCreate(startEvent)
        istat = cudaEventCreate(stopEvent)


        ! start calling and measuring the kernels
        write(*,'(/,a25,a25, a25)') 'Routine', 'Bandwidth (GB/s)'


        ! COPY KERNEL
        write(*,'(a25)', advance='NO') 'copy'
        ! warm up gpu with a single call
        call copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata)
        ! trigger events and call kernel several times
        istat = cudaEventRecord(startEvent, 0)
        do i = 1, NUM_REPS
                call copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata)
        enddo
        istat = cudaEventRecord(stopEvent, 0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        ! send copied data back to host 
        h_cdata = d_cdata
        ! verify data, calculate bandwidth
        call postprocess(h_idata, h_cdata, time)
                


        ! COPY SHARED MEM KERNEL
        write(*,'(a25)', advance='NO') 'copy shared mem'
        d_cdata = 0.0
        call copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata)
        istat = cudaEventRecord(startEvent, 0)
        do i = 1, NUM_REPS
                call copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata)
        enddo
        istat = cudaEventRecord(stopEvent, 0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        h_cdata = d_cdata
        call postprocess(h_idata, h_cdata, time)
        


        ! NAIVE TRANSPOSITION KERNEL
        write(*,'(a25)', advance='NO') 'naive transposition'
        d_tdata = 0.0
        call transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata)
        istat = cudaEventRecord(startEvent, 0)
        do i = 1, NUM_REPS
                call transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata)
        enddo
        istat = cudaEventRecord(stopEvent, 0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        h_tdata = d_tdata
        call postprocess(gold, h_tdata, time)
        

contains
        subroutine postprocess(ref, res, t)
        ! checks on bandwidth and results
                real, intent(in) :: ref(:,:), res(:,:), t
                if (all(res == ref)) then
                        write(*, '(f20.2)') 2.0*1000*mem_size / (10**9 * t/NUM_REPS)
                else
                        write(*,'(a20)') '*** Failed! ***'
                endif
        end subroutine postprocess


end program transposemat

