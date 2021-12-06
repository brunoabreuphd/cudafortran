!! This code is entirely based on the following NVIDIA's blog post:
!! https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-fortran/
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

module sillykernels
contains
        attributes(global) subroutine sillykernel1(a, offset)
                implicit none
                real :: a(*)
                integer, value :: offset
                integer :: i
                real :: c, s, x

                i = offset + threadIdx%x + (blockIdx%x-1)*blockDim%x
                x = i
                s = sin(x)
                c = cos(x)
                a(i) = a(i) + sqrt(s**2+c**2)
        end subroutine sillykernel1
end module sillykernels


program overlap_dt
        use cudafor
        use sillykernels
        implicit none
        ! block size and number of CUDA streams
        integer, parameter :: blockSize = 256, nStreams=4
        ! array size
        integer, parameter :: n = 4*1024*blockSize*nStreams
        ! array must be pinned to memory
        real, pinned, allocatable :: a(:)
        ! device array
        real, device :: a_d(n)
        ! variable for CUDA stream
        integer(kind=cuda_stream_kind) :: stream(nStreams)
        ! variable for CUDA Eventes
        type(cudaEvent) :: startEvent, stopEvent, placeholderEvent
        ! CUDA device properties
        type(cudaDeviceProp) :: prop
        ! auxiliary variables   
        real :: time
        integer :: i, istat, offset, streamSize = n/nStreams
        logical :: pinnedFlag

        ! get device name
        istat = cudaGetDeviceProperties(prop, 0)
        write(*,"(' Device: ', a,/)") trim(prop%name)

        ! allocate pinned host memory
        allocate(a(n), STAT=istat, PINNED=pinnedFlag)
        if (istat /= 0) then
                write(*,*) 'Allocation of array failed'
                stop
        else
                if (.not. pinnedFlag) then
                        write(*,*) 'Pinned allocation failed'
                endif
        endif

        ! create cuda events and streams
        istat = cudaEventCreate(startEvent)
        istat = cudaEventCreate(stopEvent)
        istat = cudaEventCreate(placeholderEvent)
        do i = 1, nStreams
                istat = cudaStreamCreate(stream(i))
        enddo


        !! SEQUENTIAL data transfer + kernel execute
        a = 0
        istat = cudaEventRecord(startEvent, 0)
        ! data transfer host -> kernel
        a_d = a
        ! launch kernel
        call sillykernel1<<<n/blockSize, blockSize>>>(a_d, 0)
        ! data transfer back
        a = a_d
        ! synchornize events and get time
        istat = cudaEventRecord(stopEvent, 0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        write(*,*) 'Time for sequential transfer and execute (ms): ', time
        write(*,*) ' Max error: ', maxval(abs(a-1.0))

end program overlap_dt
