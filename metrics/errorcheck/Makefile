# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

exec = saxpy_errorcheck

$(exec): saxpy_errorcheck.o
	$(FC) $(FFLAGS) -o $@ $^

saxpy_errorcheck.o: saxpy_errorcheck.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(exec)
