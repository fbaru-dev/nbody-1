all:
	gcc src/nbody.c -o bin/gccnbody.exe -std=gnu99 -Ofast -funroll-loops -fopenmp -march=core-avx-i -Wall -pedantic -lrt -lm
	icc src/nbody.c -o bin/iccnbody.exe -std=gnu99 -O3 -xAVX -restrict -fopenmp -Wall -pedantic -lrt -lm

debug:	
	gcc src/nbody.c -o bin/gccnbody.exe -std=gnu99 -fopenmp -mavx -Wall -pedantic -lrt -lm -g
	icc src/nbody.c -o bin/iccnbody.exe -std=gnu99 -fopenmp -xAVX -Wall -pedantic -lrt -lm -g

clean:
	rm bin/*.exe
