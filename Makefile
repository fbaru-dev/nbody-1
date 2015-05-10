all:
	mpicc src/nbody.c -o bin/gccnbody.exe -std=gnu99 -Ofast -funroll-loops -march=core-avx-i -Wall -pedantic -lrt -lm -lmpi -g
	mpicc-vt -vt:cc mpicc src/nbody.c -o bin/vtgccnbody.exe -std=gnu99 -Ofast -funroll-loops -march=core-avx-i -Wall -pedantic -lrt -lm -lmpi

ulg:
	mpicc src/nbody.c -o bin/gccnbody.exe -std=gnu99 -Ofast -funroll-loops -march=barcelona -Wall -pedantic -lrt -lm -lmpi -g
	mpicc-vt -vt:cc mpicc src/nbody.c -o bin/vtgccnbody.exe -std=gnu99 -Ofast -funroll-loops -march=barcelona -Wall -pedantic -lrt -lm -lmpi


clean:
	rm bin/*.exe
