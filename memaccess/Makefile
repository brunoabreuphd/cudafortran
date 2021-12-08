# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

execs = memaccess

all: $(execs)

memaccess: memaccess.o
	$(FC) $(FFLAGS) -o $@ $^
memaccess.o: memaccess.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
