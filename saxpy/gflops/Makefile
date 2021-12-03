# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

exec = saxpy_gflops

$(exec): saxpy_gflops.o
	$(FC) $(FFLAGS) -o $@ $^

saxpy_gflops.o: saxpy_gflops.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(exec)
