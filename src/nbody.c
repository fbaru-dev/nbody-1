#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>



// Algorithm parameters
#define G 0.1
#define NBODIES 1000
#define NTIMESTEPS 5000
#define STEPSIZE 0.0001
#define FORCELIMIT 0.0001

//#define PRINTPOS 1

// Choose precision
#define DOUBLEPREC 1
	typedef double real_t;
//#define SINGLEPREC 1
//	typedef float real_t;


// Function prototypes
void TimeStep(
	FILE * plotfile,
	const int timeStep,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void ComputeAccel(
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);
void ComputeAccelAVX(
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void UpdatePositions(
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay);

void SetInitialConditions(
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	real_t * restrict mass);

void PrintPositions(FILE * plotfile, const real_t * restrict rx, const real_t * restrict ry);

double ErrorCheck(const real_t * restrict rx);


// Utility functions
double GetWallTime(void);




int main(void)
{
	double timeElapsed;

	// files for printing
	FILE * datfile = fopen("plots/pos.dat","w");
	FILE * plotfile = fopen("plots/plot.plt","w");

	// Allocate arrays
	real_t * rx;
	real_t * ry;
	real_t * vx;
	real_t * vy;
	real_t * ax;
	real_t * ay;
	real_t * mass;
	rx = _mm_malloc(NBODIES * sizeof *rx,32);
	ry = _mm_malloc(NBODIES * sizeof *ry,32);
	vx = _mm_malloc(NBODIES * sizeof *vx,32);
	vy = _mm_malloc(NBODIES * sizeof *vy,32);
	ax = _mm_malloc(NBODIES * sizeof *ax,32);
	ay = _mm_malloc(NBODIES * sizeof *ay,32);
	mass = _mm_malloc(NBODIES * sizeof *mass,32);

	SetInitialConditions(rx,ry, vx,vy, ax,ay, mass);



#ifdef PRINTPOS
	fprintf(plotfile, "set term pngcairo enhanced size 1024,768\n");
	fprintf(plotfile, "set output \"test.png\"\n");
	fprintf(plotfile, "set grid\n");
	fprintf(plotfile, "set key off\n");
	fprintf(plotfile, "set xrange [-100:100]\n");
	fprintf(plotfile, "set yrange [-100:100]\n");
	fprintf(plotfile, "plot \\\n");
	for (int i = 0; i < NBODIES-1; i++) {
		fprintf(plotfile, "\"pos.dat\" using %d:%d with linespoints,\\\n", 2*i+1, 2*i+2);
	}
	fprintf(plotfile, "\"pos.dat\" using %d:%d with linespoints\n", 2*(NBODIES-1)+1, 2*(NBODIES-1)+2);
#endif



	timeElapsed = GetWallTime();
	for (int n = 0; n < NTIMESTEPS; n++) {
		TimeStep(datfile, n, rx,ry, vx,vy, ax,ay, mass);
	}
	timeElapsed = GetWallTime() - timeElapsed;
	printf("MegaUpdates/second: %lf. Error: %le\n", NTIMESTEPS*NBODIES/timeElapsed/1000000.0, ErrorCheck(rx));



	_mm_free(rx);
	_mm_free(ry);
	_mm_free(vx);
	_mm_free(vy);
	_mm_free(ax);
	_mm_free(ay);
	_mm_free(mass);
	fclose(plotfile);
	fclose(datfile);
	return 0;
}



void TimeStep(
	FILE * plotfile,
	const int timeStep,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	ComputeAccelAVX(rx,ry, ax,ay, mass);
	UpdatePositions(rx,ry, vx,vy, ax,ay);
#ifdef PRINTPOS
	if (timeStep % 1000 == 0) {
		PrintPositions(plotfile, rx,ry);
	}
#endif
}


void ComputeAccel(
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	double distx, disty, sqrtRecipDist;

	for (int i = 0; i < NBODIES; i++) {
		for (int j = i+1; j < NBODIES; j++) {
			distx = rx[i] - rx[j];
			disty = ry[i] - ry[j];
			sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
			ax[i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ax[j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);

			// This version with a force-limiting term stops nearby bodies experiencing arbitrarily high
			// forces. Important for numerical stability, but not for performance testing.
//			ax[i] += (mass[j] * distx * pow(sqrtRecipDist*sqrtRecipDist+FORCELIMIT,3.0/2.0));
//			ay[i] += (mass[j] * disty * pow(sqrtRecipDist*sqrtRecipDist+FORCELIMIT,3.0/2.0));
//			ax[j] -= (mass[i] * distx * pow(sqrtRecipDist*sqrtRecipDist+FORCELIMIT,3.0/2.0));
//			ay[j] -= (mass[i] * disty * pow(sqrtRecipDist*sqrtRecipDist+FORCELIMIT,3.0/2.0));
		}
	}

}


void ComputeAccelAVX(
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	double distx, disty, sqrtRecipDist;

	// limit of vectorized loop is the multiple of for including or below NBODIES
	// May have to "clean up" a few bodies at the end
	const int jVecMax = 4*(NBODIES/4);

	for (int i = 0; i < NBODIES; i++) {

		// Vectorized j loop starts at multiple of 4 >= i+1
		const int jVecMin = (4*((i)/4)+4) > NBODIES ? NBODIES : (4*((i)/4)+4);

		// first initial non-vectorized part
		for (int j = i+1; j < jVecMin; j++) {
			distx = rx[i] - rx[j];
			disty = ry[i] - ry[j];
			sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
			ax[i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ax[j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
		}

		// if we have already finished, break out of the loop early
		if (jVecMax < jVecMin) break;


		// main vectorized part
		__m256d rxiVec = _mm256_set1_pd(rx[i]);
		__m256d ryiVec = _mm256_set1_pd(ry[i]);
		__m256d massiVec = _mm256_set1_pd(mass[i]);
		__m256d axiUpdVec = _mm256_set1_pd(0.0);
		__m256d ayiUpdVec = _mm256_set1_pd(0.0);

		for (int j = jVecMin; j < jVecMax; j+=4) {
			__m256d rxjVec = _mm256_load_pd(&rx[j]);
			__m256d ryjVec = _mm256_load_pd(&ry[j]);
			__m256d axjVec = _mm256_load_pd(&ax[j]);
			__m256d ayjVec = _mm256_load_pd(&ay[j]);
			__m256d massjVec = _mm256_load_pd(&mass[j]);

			__m256d distxVec = _mm256_sub_pd(rxiVec, rxjVec);
			__m256d distyVec = _mm256_sub_pd(ryiVec, ryjVec);

			__m256d sqrtRecipDistVec = _mm256_div_pd(_mm256_set1_pd(1.0),
			                                         _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(distxVec,distxVec),
			                                                                      _mm256_mul_pd(distyVec,distyVec))));

			axiUpdVec = _mm256_add_pd(axiUpdVec,_mm256_mul_pd(massjVec,
			                                                  _mm256_mul_pd(distxVec,
			                                                                _mm256_mul_pd(sqrtRecipDistVec,
			                                                                              _mm256_mul_pd(sqrtRecipDistVec,
			                                                                                            sqrtRecipDistVec)))));
			ayiUpdVec = _mm256_add_pd(ayiUpdVec,_mm256_mul_pd(massjVec,
			                                                  _mm256_mul_pd(distyVec,
			                                                                _mm256_mul_pd(sqrtRecipDistVec,
			                                                                              _mm256_mul_pd(sqrtRecipDistVec,
			                                                                                            sqrtRecipDistVec)))));
			axjVec = _mm256_sub_pd(axjVec,_mm256_mul_pd(massiVec,
			                                                  _mm256_mul_pd(distxVec,
			                                                                _mm256_mul_pd(sqrtRecipDistVec,
			                                                                              _mm256_mul_pd(sqrtRecipDistVec,
			                                                                                            sqrtRecipDistVec)))));
			ayjVec = _mm256_sub_pd(ayjVec,_mm256_mul_pd(massiVec,
			                                                  _mm256_mul_pd(distyVec,
			                                                                _mm256_mul_pd(sqrtRecipDistVec,
			                                                                              _mm256_mul_pd(sqrtRecipDistVec,
			                                                                                            sqrtRecipDistVec)))));

			_mm256_store_pd(&ax[j],axjVec);
			_mm256_store_pd(&ay[j],ayjVec);
		}

		// Now need to sum elements of axiUpdVec,ayiUpdVec and add to ax[i],ay[i]
		axiUpdVec = _mm256_hadd_pd(axiUpdVec,axiUpdVec);
		ayiUpdVec = _mm256_hadd_pd(ayiUpdVec,ayiUpdVec);
		ax[i] += ((double*)&axiUpdVec)[0] + ((double*)&axiUpdVec)[2];
		ay[i] += ((double*)&ayiUpdVec)[0] + ((double*)&ayiUpdVec)[2];


		// final non-vectorized part, iff we didn't already run up to a jVecMin which is larger than jVecMax
		for (int j = jVecMax; j < NBODIES; j++) {
			distx = rx[i] - rx[j];
			disty = ry[i] - ry[j];
			sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
			ax[i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ax[j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
		}
	}

}

void UpdatePositions(
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay)
{
	for (int i = 0; i < NBODIES; i++) {
			//new force values in .ax, .ay
			//update pos and vel
			vx[i] += (-G)*STEPSIZE * ax[i];
			vy[i] += (-G)*STEPSIZE * ay[i];
			rx[i] += STEPSIZE * vx[i];
			ry[i] += STEPSIZE * vy[i];
			//zero accel values to avoid an extra loop in ComputeAccel
			ax[i] = 0;
			ay[i] = 0;
		}

}


void SetInitialConditions(
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	real_t * restrict mass)
{
	// Set random initial conditions.
	for (int i = 0; i < NBODIES; i++) {
		rx[i] = ((double)rand()*(double)100/(double)RAND_MAX)*pow(-1,rand()%2);
		ry[i] = ((double)rand()*(double)100/(double)RAND_MAX)*pow(-1,rand()%2);
		vx[i] = (rand()%3)*pow(-1,rand()%2);
		vy[i] = (rand()%3)*pow(-1,rand()%2);
		ax[i] = 0;
		ay[i] = 0;
		mass[i] = 1000;
	}
}

double ErrorCheck(const real_t * restrict rx)
{
	// Compute sum of x coordinates. Can use to check consistency between versions.
	double sumx = 0;
	for (int i = 0; i < NBODIES; i++) {
		sumx += rx[i];
	}
	return sumx;
}


void PrintPositions(FILE * file, const real_t * restrict rx, const real_t * restrict ry)
{
	for(int i = 0; i < NBODIES-1; i++) {
		fprintf(file, "%le %le ", rx[i],ry[i]);
	}
	fprintf(file, "%le %le\n",rx[NBODIES-1],ry[NBODIES-1]);
}





double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}
