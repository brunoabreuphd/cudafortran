# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

execs = devicequery

all: $(execs)

devicequery: devicequery.o
	$(FC) $(FFLAGS) -o $@ $^
devicequery.o: devicequery.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
