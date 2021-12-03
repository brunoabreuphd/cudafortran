# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

execs = bandwidth

all: $(execs)

bandwidth: bandwidth.o
	$(FC) $(FFLAGS) -o $@ $^
bandwidth.o: bandwidth.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
