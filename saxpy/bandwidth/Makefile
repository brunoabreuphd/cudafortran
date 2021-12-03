# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

exec = saxpy_bandwidth

$(exec): saxpy_bandwidth.o
	$(FC) $(FFLAGS) -o $@ $^

saxpy_bandwidth.o: saxpy_bandwidth.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(exec)
