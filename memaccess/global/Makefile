# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

execs = memaccess_global

all: $(execs)

memaccess_global: memaccess_global.o
	$(FC) $(FFLAGS) -o $@ $^
memaccess_global.o: memaccess_global.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
