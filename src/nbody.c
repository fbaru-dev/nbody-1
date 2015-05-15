#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>



// Algorithm parameters
#define G 0.1
#define STEPSIZE 0.0001
#define FORCELIMIT 0.0001

// MPI
#define MASTER 0
#define IFMASTER if(myRank==0)
#define IFNOTMASTER if(myRank>0)

// Choose precision
#define DOUBLEPREC 1
#define MPI_REAL_T MPI_DOUBLE
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
	const int timeStep,
	const int *nBodiesBoundaries,
	const int nBodiesPadded,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void ComputeAccel(
	const int *nBodiesBoundaries,
	const int nBodiesPadded,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void ComputeAccelVec(
	const int *nBodiesBoundaries,
	const int nBodiesPadded,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void UpdateAllPositions(
	const int *nBodiesBoundaries,
	const int nBodiesPadded,
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

void MPIDistributeInitialConditions(
	const int nBodies,
	const int *nBodiesBoundaries,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	real_t * restrict mass);

double ErrorCheck(const int nBodies, const real_t * restrict rx);

// Utility functions
double GetWallTime(void);




int main(int argc, char** argv)
{
	int totalRanks, myRank;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
IFMASTER printf("MPI: totalRanks = %d          \n",totalRanks);


	const int nTimeSteps = 2000;

	RunSimulation(nTimeSteps,NBODIES);


	MPI_Finalize();
	return 0;
}


void RunSimulation(const int nTimeSteps, const int nBodies)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	const int threadCount = omp_get_max_threads();


	// Share bodies between ranks, with the aim of creating an even work-share.
	int * nBodiesBoundaries = malloc((totalRanks+1) * sizeof (int));
	nBodiesBoundaries[0] = 0;
	for (int r = 0; r < totalRanks-1; r++) {
		nBodiesBoundaries[r+1] = (int)(nBodies - (totalRanks + sqrt((double)((nBodies-1)*(double)totalRanks*(-(r+1)*nBodies +(nBodies-1)*totalRanks))))/(double)totalRanks);
	}
	nBodiesBoundaries[totalRanks] = nBodies;
	

	// if nBodies is small this scheme does not work, so share evenly if nBodiesBoundaries[1] <= 0
	if (nBodiesBoundaries[1] <= 0) {
		nBodiesBoundaries[0] = 0;
		for (int r = 1; r < totalRanks; r++) {
			nBodiesBoundaries[r] = nBodiesBoundaries[r-1] + nBodies/totalRanks;
		}
		nBodiesBoundaries[totalRanks] = nBodies;
	}

	printf("rank %d has share %d, (%d to %d)\n",myRank, nBodiesBoundaries[myRank+1]-nBodiesBoundaries[myRank],nBodiesBoundaries[myRank],nBodiesBoundaries[myRank+1]);

	// Allocate arrays large enough to hold all particles on *each* rank
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
	mass = _mm_malloc(nBodies * sizeof *mass,32);

	// In parallel, each thread needs its own acceleration array, so that it can update the totals in it. If nBodies is
	// not a multiple of 4, threads with id > 0 will not be accessing 32byte aligned values when they load 4 doubles into
	// an __m256d, so we need to pad the array to the multiple of 4 above nBodies.
	const int nBodiesPadded = 4*((nBodiesBoundaries[totalRanks]-1)/4)+4;
	ax =   _mm_malloc(nBodiesPadded * threadCount * sizeof *ax,32);
	ay =   _mm_malloc(nBodiesPadded * threadCount * sizeof *ay,32);



	// Before the master thread distributes initial conditions, each thread should touch its memory
	// to exploit first-touch NUMA allocation
	#pragma omp parallel for default(none) shared(ax,ay)
	for (int tid = 0; tid < threadCount; tid++) {
		if (tid == omp_get_thread_num()) {
			memset(&(ax[nBodiesPadded*tid]), 0.0, nBodiesPadded * sizeof *ax);
			memset(&(ay[nBodiesPadded*tid]), 0.0, nBodiesPadded * sizeof *ay);
		}
	}
	// Local block of rx,ry,mass accessed the most, so ensure this is allocated properly.
	#pragma omp parallel for default(none) shared(rx,ry,mass,nBodiesBoundaries,myRank) schedule(static,8)
	for (int i = nBodiesBoundaries[myRank]; i < nBodiesBoundaries[myRank+1]; i++) {
		rx[i] = 0;
		ry[i] = 0;
		mass[i] = 0;
	}


	// Master thread sets the initial conditions, and then sends the relevant data to ranks, ie, a broadcast of position,
	// velocity, acceleration and mass arrays
IFMASTER SetInitialConditions(nBodies, rx,ry, vx,vy, ax,ay, mass);
	MPIDistributeInitialConditions(nBodies, nBodiesBoundaries, rx,ry, vx,vy, ax,ay, mass);


	double timeElapsed;
	timeElapsed = GetWallTime();
	for (int n = 0; n < nTimeSteps; n++) {
		TimeStep(n, nBodiesBoundaries, nBodiesPadded, rx,ry, vx,vy, ax,ay, mass);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	timeElapsed = GetWallTime() - timeElapsed;

	// Only care about the master thread's timing.
IFMASTER printf("%4d Time %lf MTSps %le Sumx %.15le\n", nBodies, timeElapsed, nTimeSteps/timeElapsed/1000000.0, ErrorCheck(nBodies, rx));


	free(nBodiesBoundaries);
	_mm_free(rx);
	_mm_free(ry);
	_mm_free(vx);
	_mm_free(vy);
	_mm_free(ax);
	_mm_free(ay);
	_mm_free(mass);
}


void TimeStep(
	const int timeStep,
	const int *nBodiesBoundaries,
	const int nBodiesPadded,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
	const int nBodies = nBodiesBoundaries[totalRanks];

#if defined(AVX) || defined(SSE)
	ComputeAccelVec(nBodiesBoundaries, nBodiesPadded, rx,ry, ax,ay, mass);
#else
	ComputeAccel(nBodiesBoundaries, nBodiesPadded, rx,ry, ax,ay, mass);
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	// Now each rank is holding partial sums of all acceleration values. Need to do a global reduction
	MPI_Allreduce(MPI_IN_PLACE, ax, nBodies, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, ay, nBodies, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

	UpdateAllPositions(nBodiesBoundaries, nBodiesPadded, rx,ry, vx,vy, ax,ay);
}


void ComputeAccel(
	const int *nBodiesBoundaries,
	const int nBodiesPadded,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	const int threadCount = omp_get_max_threads();

	#pragma omp parallel for default(none) shared(ax,ay,rx,ry,mass,nBodiesBoundaries,myRank,totalRanks) schedule(static,8)
	for (int i = nBodiesBoundaries[myRank]; i < nBodiesBoundaries[myRank+1]; i++) {
		int tid = omp_get_thread_num();
		for (int j = i+1; j < nBodiesBoundaries[totalRanks]; j++) {
			real_t distx = rx[i] - rx[j];
			real_t disty = ry[i] - ry[j];
			real_t sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
			ax[tid*nBodiesPadded+i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[tid*nBodiesPadded+i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ax[tid*nBodiesPadded+j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[tid*nBodiesPadded+j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);

		}
	}
	#pragma omp parallel for default(none), shared(ax,ay,nBodiesBoundaries,totalRanks) schedule(static)
	for (int i = 0; i < nBodiesBoundaries[totalRanks]; i++) {
		for (int j = 1; j < threadCount; j++) {
			ax[i] += ax[j*nBodiesPadded + i];
			ay[i] += ay[j*nBodiesPadded + i];
		}
	}

}


void ComputeAccelVec(
	const int *nBodiesBoundaries,
	const int nBodiesPadded,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	const int threadCount = omp_get_max_threads();

	// limit of vectorized loop is the multiple of VECWIDTH <= NBODIES
	// (May have to "clean up" a few bodies at the end)
	const int jVecMax = VECWIDTH*(nBodiesBoundaries[totalRanks]/VECWIDTH);

	#pragma omp parallel for default(none) shared(ax,ay,rx,ry,mass,nBodiesBoundaries,myRank,totalRanks) schedule(static,8)
	for (int i = nBodiesBoundaries[myRank]; i < nBodiesBoundaries[myRank+1]; i++) {
		// this must be INSIDE the parallel region
		const int tid = omp_get_thread_num();

		// Vectorized j loop starts at multiple of 4 >= i+1
		const int jVecMin = (VECWIDTH*((i+1)/VECWIDTH)+VECWIDTH) > nBodiesBoundaries[totalRanks] ? nBodiesBoundaries[totalRanks] : (VECWIDTH*((i+1)/VECWIDTH)+VECWIDTH);

		// first initial non-vectorized part
		for (int j = i+1; j < jVecMin; j++) {
			real_t distx = rx[i] - rx[j];
			real_t disty = ry[i] - ry[j];
			real_t sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
			ax[tid*nBodiesPadded + i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[tid*nBodiesPadded + i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ax[tid*nBodiesPadded + j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[tid*nBodiesPadded + j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
		}

		// Continue iff jVecMin <= jVecMax: otherwise we have already finished. This happens when we are right at the
		// end of the i loop
		if (jVecMin <= jVecMax) {


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
				__m256d axjVec = _mm256_load_pd(&ax[tid*nBodiesPadded + j]);
				__m256d ayjVec = _mm256_load_pd(&ay[tid*nBodiesPadded + j]);
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

				_mm256_store_pd(&ax[tid*nBodiesPadded + j],axjVec);
				_mm256_store_pd(&ay[tid*nBodiesPadded + j],ayjVec);
			}

			// Now need to sum elements of axiUpdVec,ayiUpdVec and add to ax[i],ay[i]
			axiUpdVec = _mm256_hadd_pd(axiUpdVec,axiUpdVec);
			ayiUpdVec = _mm256_hadd_pd(ayiUpdVec,ayiUpdVec);
			ax[tid*nBodiesPadded + i] += ((double*)&axiUpdVec)[0] + ((double*)&axiUpdVec)[2];
			ay[tid*nBodiesPadded + i] += ((double*)&ayiUpdVec)[0] + ((double*)&ayiUpdVec)[2];
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
				__m128d axjVec = _mm_load_pd(&ax[tid*nBodiesPadded + j]);
				__m128d ayjVec = _mm_load_pd(&ay[tid*nBodiesPadded + j]);
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


				_mm_store_pd(&ax[tid*nBodiesPadded + j],axjVec);
				_mm_store_pd(&ay[tid*nBodiesPadded + j],ayjVec);
			}

			// Now need to sum elements of axiUpdVec,ayiUpdVec and add to ax[i],ay[i]
			axiUpdVec = _mm_hadd_pd(axiUpdVec,axiUpdVec);
			ayiUpdVec = _mm_hadd_pd(ayiUpdVec,ayiUpdVec);
			ax[tid*nBodiesPadded + i] += ((double*)&axiUpdVec)[0];
			ay[tid*nBodiesPadded + i] += ((double*)&ayiUpdVec)[0];
#endif


			// final non-vectorized part, iff we didn't already run up to a jVecMin which is larger than jVecMax
			for (int j = jVecMax; j < nBodiesBoundaries[totalRanks]; j++) {
				real_t distx = rx[i] - rx[j];
				real_t disty = ry[i] - ry[j];
				real_t sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
				ax[tid*nBodiesPadded + i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
				ay[tid*nBodiesPadded + i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
				ax[tid*nBodiesPadded + j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
				ay[tid*nBodiesPadded + j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			}
		}
	}

	// Combine ax,ay sums from threads
	#pragma omp parallel for default(none), shared(ax,ay,nBodiesBoundaries,totalRanks) schedule(static)
	for (int i = 0; i < nBodiesBoundaries[totalRanks]; i++) {
		for (int j = 1; j < threadCount; j++) {
			ax[i] += ax[j*nBodiesPadded + i];
			ay[i] += ay[j*nBodiesPadded + i];
		}
	}

}


void UpdateAllPositions(
	const int *nBodiesBoundaries,
	const int nBodiesPadded,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	const int threadCount = omp_get_max_threads();


	for (int i = nBodiesBoundaries[0]; i < nBodiesBoundaries[totalRanks]; i++) {
			//new force values in ax, ay
			//update pos and vel
			vx[i] += (-G)*STEPSIZE * ax[i];
			vy[i] += (-G)*STEPSIZE * ay[i];
			rx[i] += STEPSIZE * vx[i];
			ry[i] += STEPSIZE * vy[i];
		}

	//zero rank's accel values ready for next timestep
	memset(ax, 0, nBodiesPadded * threadCount * sizeof *ax);
	memset(ay, 0, nBodiesPadded * threadCount * sizeof *ay);

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

void MPIDistributeInitialConditions(
	const int nBodies,
	const int *nBodiesBoundaries,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	real_t * restrict mass)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	// Send all positions, accelerations and masses to all ranks.
	MPI_Bcast(rx, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(ry, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(vx, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(vy, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(ax, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(ay, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(mass, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);
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


double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}
