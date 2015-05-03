all:
	gcc src/nbody.c -o bin/gccnbody.exe -std=gnu99 -Ofast -funroll-loops -march=core-avx-i -Wall -pedantic -lrt -lm
	icc src/nbody.c -o bin/iccnbody.exe -std=gnu99 -O3 -xAVX -restrict -Wall -pedantic -lrt -lm


clean:
	rm bin/*.exe
