# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

execs = transposemat

all: $(execs)

transposemat: transposemat.o
	$(FC) $(FFLAGS) -o $@ $^
transposemat.o: transposemat.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
