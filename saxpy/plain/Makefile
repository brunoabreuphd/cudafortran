# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

exec = saxpy

$(exec): saxpy.o
	$(FC) $(FFLAGS) -o $@ $^

saxpy.o: saxpy.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(exec)
