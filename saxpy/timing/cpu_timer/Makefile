# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

exec = saxpy_cputimer

$(exec): saxpy_cputimer.o
	$(FC) $(FFLAGS) -o $@ $^

saxpy_cputimer.o: saxpy_cputimer.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(exec)
