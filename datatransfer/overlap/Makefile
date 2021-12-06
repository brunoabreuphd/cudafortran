# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda

execs = overlap_dt

all: $(execs)

overlap_dt: overlap_dt.o
	$(FC) $(FFLAGS) -o $@ $^
overlap_dt.o: overlap_dt.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
