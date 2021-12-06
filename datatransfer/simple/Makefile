# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

execs = simple_dt

all: $(execs)

simple_dt: simple_dt.o
	$(FC) $(FFLAGS) -o $@ $^
simple_dt.o: simple_dt.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
