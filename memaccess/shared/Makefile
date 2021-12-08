# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

execs = memaccess_shared

all: $(execs)

memaccess_shared: memaccess_shared.o
	$(FC) $(FFLAGS) -o $@ $^
memaccess_shared.o: memaccess_shared.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
