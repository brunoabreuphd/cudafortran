# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

execs = pinned_dt

all: $(execs)

pinned_dt: pinned_dt.o
	$(FC) $(FFLAGS) -o $@ $^
pinned_dt.o: pinned_dt.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
