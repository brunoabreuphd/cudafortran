# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

exec = saxpy_cudaevents

$(exec): saxpy_cudaevents.o
	$(FC) $(FFLAGS) -o $@ $^

saxpy_cudaevents.o: saxpy_cudaevents.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(exec)
