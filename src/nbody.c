#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>



// Algorithm parameters
#define G 0.1
#define STEPSIZE 0.0001
#define FORCELIMIT 0.0001

//#define PRINTPOS 1

// Choose precision
#define DOUBLEPREC 1
	typedef double real_t;
//	#define VECWIDTH 1
//	#define AVX 1
//		#define VECWIDTH 4
	#define SSE 1
		#define VECWIDTH 2


//#define SINGLEPREC 1
//	typedef float real_t;
//	#define AVX 1
//		#define VECWIDTH 8
//	#define SSE 1
//		#define VECWIDTH 4
//#define VECWIDTH 1


// Function prototypes
void RunSimulation(const int ntimeSteps, const int nBodies);

void TimeStep(
	FILE * plotfile,
	const int timeStep,
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void ComputeAccel(
	const int nBodies,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void ComputeAccelVec(
	const int nBodies,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void UpdatePositions(
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay);

void SetInitialConditions(
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	real_t * restrict mass);

void PrintPositions(FILE * plotfile, const int nBodies, const real_t * restrict rx, const real_t * restrict ry);
double ErrorCheck(const int nBodies, const real_t * restrict rx);

// Utility functions
double GetWallTime(void);




int main(void)
{
	const int nTimeSteps = 2000;
	//const int nTimeSteps = 100;

/*
	for (int nBodies = 10; nBodies < 200; nBodies += 10) {
		RunSimulation(nTimeSteps, nBodies);
	}
	for (int nBodies = 100; nBodies < 300; nBodies += 20) {
		RunSimulation(nTimeSteps, nBodies);
	}
	for (int nBodies = 300; nBodies < 1000; nBodies += 100) {
		RunSimulation(nTimeSteps, nBodies);
	}
	for (int nBodies = 1000; nBodies <= 1000; nBodies += 500) {
		RunSimulation(nTimeSteps, nBodies);
	}
*/
	RunSimulation(nTimeSteps, 4000);
	printf("Complete!\n");
	return 0;
}


void RunSimulation(const int nTimeSteps, const int nBodies)
{
	double timeElapsed;

	// files for printing
	char filename[25];
	sprintf(filename, "plots/pos%d.dat", nBodies);
	FILE * datfile = fopen(filename,"w");
	FILE * plotfile = fopen("plots/plot.plt","w");

	// Allocate arrays
	real_t * rx;
	real_t * ry;
	real_t * vx;
	real_t * vy;
	real_t * ax;
	real_t * ay;
	real_t * mass;
	rx =   _mm_malloc(nBodies * sizeof *rx,32);
	ry =   _mm_malloc(nBodies * sizeof *ry,32);
	vx =   _mm_malloc(nBodies * sizeof *vx,32);
	vy =   _mm_malloc(nBodies * sizeof *vy,32);
	ax =   _mm_malloc(nBodies * sizeof *ax,32);
	ay =   _mm_malloc(nBodies * sizeof *ay,32);
	mass = _mm_malloc(nBodies * sizeof *mass,32);

	SetInitialConditions(nBodies, rx,ry, vx,vy, ax,ay, mass);



#ifdef PRINTPOS
	fprintf(plotfile, "set term pngcairo enhanced size 1024,768\n");
	fprintf(plotfile, "set output \"test.png\"\n");
	fprintf(plotfile, "set grid\n");
	fprintf(plotfile, "set key off\n");
	fprintf(plotfile, "set xrange [-100:100]\n");
	fprintf(plotfile, "set yrange [-100:100]\n");
	fprintf(plotfile, "plot \\\n");
	for (int i = 0; i < nBodies-1; i++) {
		fprintf(plotfile, "\"pos.dat\" using %d:%d with linespoints,\\\n", 2*i+1, 2*i+2);
	}
	fprintf(plotfile, "\"pos.dat\" using %d:%d with linespoints\n", 2*(nBodies-1)+1, 2*(nBodies-1)+2);
#endif



	timeElapsed = GetWallTime();
	for (int n = 0; n < nTimeSteps; n++) {
		TimeStep(datfile, n, nBodies, rx,ry, vx,vy, ax,ay, mass);
	}
	timeElapsed = GetWallTime() - timeElapsed;
//	printf("nBodies: %4d, MegaUpdates/second: %lf. Error: %le\n", nBodies, nTimeSteps*nBodies/timeElapsed/1000000.0, ErrorCheck(nBodies, rx));
	printf("%4d Time %lf MTSps %le Sumx %.15le\n", nBodies, timeElapsed, nTimeSteps/timeElapsed/1000000.0, ErrorCheck(nBodies, rx));



	_mm_free(rx);
	_mm_free(ry);
	_mm_free(vx);
	_mm_free(vy);
	_mm_free(ax);
	_mm_free(ay);
	_mm_free(mass);
	fclose(plotfile);
	fclose(datfile);


}


void TimeStep(
	FILE * plotfile,
	const int timeStep,
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
#if defined(AVX) || defined(SSE)
	ComputeAccelVec(nBodies, rx,ry, ax,ay, mass);
#else
	ComputeAccel(nBodies, rx,ry, ax,ay, mass);
#endif

	UpdatePositions(nBodies, rx,ry, vx,vy, ax,ay);

#ifdef PRINTPOS
	if (timeStep % 1000 == 0) {
		PrintPositions(plotfile, nBodies, rx,ry);
	}
#endif
}


void ComputeAccel(
	const int nBodies,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	double distx, disty, sqrtRecipDist;

	for (int i = 0; i < nBodies; i++) {
		for (int j = i+1; j < nBodies; j++) {
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


void ComputeAccelVec(
	const int nBodies,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	double distx, disty, sqrtRecipDist;

	// limit of vectorized loop is the multiple of VECWIDTH <= NBODIES
	// (May have to "clean up" a few bodies at the end)
	const int jVecMax = VECWIDTH*(nBodies/VECWIDTH);

	for (int i = 0; i < nBodies; i++) {

		// Vectorized j loop starts at multiple of 4 >= i+1
		const int jVecMin = (VECWIDTH*((i)/VECWIDTH)+VECWIDTH) > nBodies ? nBodies : (VECWIDTH*((i)/VECWIDTH)+VECWIDTH);

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


		// main vectorized part. Here we have code for both AVX and SSE
#ifdef AVX
		__m256d rxiVec = _mm256_set1_pd(rx[i]);
		__m256d ryiVec = _mm256_set1_pd(ry[i]);
		__m256d massiVec = _mm256_set1_pd(mass[i]);
		__m256d axiUpdVec = _mm256_set1_pd(0.0);
		__m256d ayiUpdVec = _mm256_set1_pd(0.0);

		for (int j = jVecMin; j < jVecMax; j+=VECWIDTH) {
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
			// cube:
			sqrtRecipDistVec = _mm256_mul_pd(sqrtRecipDistVec,_mm256_mul_pd(sqrtRecipDistVec,sqrtRecipDistVec));

			// multiply into distxVec and distyVec
			distxVec = _mm256_mul_pd(distxVec,sqrtRecipDistVec);
			distyVec = _mm256_mul_pd(distyVec,sqrtRecipDistVec);

			// update accelerations
			axiUpdVec = _mm256_add_pd(axiUpdVec,_mm256_mul_pd(massjVec,distxVec));
			ayiUpdVec = _mm256_add_pd(ayiUpdVec,_mm256_mul_pd(massjVec,distyVec));
			axjVec = _mm256_sub_pd(axjVec,_mm256_mul_pd(massiVec,distxVec));
			ayjVec = _mm256_sub_pd(ayjVec,_mm256_mul_pd(massiVec,distyVec));

			_mm256_store_pd(&ax[j],axjVec);
			_mm256_store_pd(&ay[j],ayjVec);
		}

		// Now need to sum elements of axiUpdVec,ayiUpdVec and add to ax[i],ay[i]
		axiUpdVec = _mm256_hadd_pd(axiUpdVec,axiUpdVec);
		ayiUpdVec = _mm256_hadd_pd(ayiUpdVec,ayiUpdVec);
		ax[i] += ((double*)&axiUpdVec)[0] + ((double*)&axiUpdVec)[2];
		ay[i] += ((double*)&ayiUpdVec)[0] + ((double*)&ayiUpdVec)[2];
#endif

#ifdef SSE
		__m128d rxiVec = _mm_set1_pd(rx[i]);
		__m128d ryiVec = _mm_set1_pd(ry[i]);
		__m128d massiVec = _mm_set1_pd(mass[i]);
		__m128d axiUpdVec = _mm_set1_pd(0.0);
		__m128d ayiUpdVec = _mm_set1_pd(0.0);

		for (int j = jVecMin; j < jVecMax; j+=VECWIDTH) {
			__m128d rxjVec = _mm_load_pd(&rx[j]);
			__m128d ryjVec = _mm_load_pd(&ry[j]);
			__m128d axjVec = _mm_load_pd(&ax[j]);
			__m128d ayjVec = _mm_load_pd(&ay[j]);
			__m128d massjVec = _mm_load_pd(&mass[j]);

			__m128d distxVec = _mm_sub_pd(rxiVec, rxjVec);
			__m128d distyVec = _mm_sub_pd(ryiVec, ryjVec);

			__m128d sqrtRecipDistVec = _mm_div_pd(_mm_set1_pd(1.0),
			                                         _mm_sqrt_pd(_mm_add_pd(_mm_mul_pd(distxVec,distxVec),
			                                                                      _mm_mul_pd(distyVec,distyVec))));
			// cube:
			sqrtRecipDistVec = _mm_mul_pd(sqrtRecipDistVec,_mm_mul_pd(sqrtRecipDistVec,sqrtRecipDistVec));

			// multiply into distxVec and distyVec
			distxVec = _mm_mul_pd(distxVec,sqrtRecipDistVec);
			distyVec = _mm_mul_pd(distyVec,sqrtRecipDistVec);

			// update accelerations
			axiUpdVec = _mm_add_pd(axiUpdVec,_mm_mul_pd(massjVec,distxVec));
			ayiUpdVec = _mm_add_pd(ayiUpdVec,_mm_mul_pd(massjVec,distyVec));
			axjVec = _mm_sub_pd(axjVec,_mm_mul_pd(massiVec,distxVec));
			ayjVec = _mm_sub_pd(ayjVec,_mm_mul_pd(massiVec,distyVec));


			_mm_store_pd(&ax[j],axjVec);
			_mm_store_pd(&ay[j],ayjVec);
		}

		// Now need to sum elements of axiUpdVec,ayiUpdVec and add to ax[i],ay[i]
		axiUpdVec = _mm_hadd_pd(axiUpdVec,axiUpdVec);
		ayiUpdVec = _mm_hadd_pd(ayiUpdVec,ayiUpdVec);
		ax[i] += ((double*)&axiUpdVec)[0];
		ay[i] += ((double*)&ayiUpdVec)[0];
#endif


		// final non-vectorized part, iff we didn't already run up to a jVecMin which is larger than jVecMax
		for (int j = jVecMax; j < nBodies; j++) {
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
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay)
{
	for (int i = 0; i < nBodies; i++) {
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
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	real_t * restrict mass)
{
/*
	// Set random initial conditions.
	for (int i = 0; i < nBodies; i++) {
		rx[i] = ((double)rand()*(double)100/(double)RAND_MAX)*pow(-1,rand()%2);
		ry[i] = ((double)rand()*(double)100/(double)RAND_MAX)*pow(-1,rand()%2);
		vx[i] = (rand()%3)*pow(-1,rand()%2);
		vy[i] = (rand()%3)*pow(-1,rand()%2);
		ax[i] = 0;
		ay[i] = 0;
		mass[i] = 1000;
	}
*/
	//Some deterministic initial conditions (for testing openmpi build's weird differences)
	for (int i = 0; i < nBodies; i++) {
		rx[i] = 500*i*pow(-1,i)*sin(i);
		ry[i] = -500*i*pow(-1,i)*cos(i);
		vx[i] = 10*i*i*pow(-1,i);
		vy[i] = -5*i*i*pow(-1,i);
		ax[i] = 0;
		ay[i] = 0;
		mass[i] = 1000;
	}
}

double ErrorCheck(const int nBodies, const real_t * restrict rx)
{
	// Compute sum of x coordinates. Can use to check consistency between versions.
	double sumx = 0;
	for (int i = 0; i < nBodies; i++) {
		sumx += rx[i];
	}
	return sumx;
}


void PrintPositions(FILE * file, const int nBodies, const real_t * restrict rx, const real_t * restrict ry)
{
	for(int i = 0; i < nBodies-1; i++) {
		fprintf(file, "%le %le ", rx[i],ry[i]);
	}
	fprintf(file, "%le %le\n",rx[nBodies-1],ry[nBodies-1]);
}





double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}
