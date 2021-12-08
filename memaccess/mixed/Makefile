# module load nvhpc/21.7 on Bridges-2
FC=nvfortran
FFLAGS=-Mcuda=ptxinfo

execs = finitediff

all: $(execs)

finitediff: finitediff.o
	$(FC) $(FFLAGS) -o $@ $^
finitediff.o: finitediff.f90
	$(FC) $(FFLAGS) -c $^

clean:
	rm *.o *.mod $(execs)
