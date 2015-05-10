#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <mpi.h>



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
	#define VECWIDTH 1
//	#define AVX 1
//		#define VECWIDTH 4
//	#define SSE 1
//		#define VECWIDTH 2

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
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void ComputeAccel(
	const int *nBodiesBoundaries,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void ComputeAccelVec(
	const int *nBodiesBoundaries,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void UpdatePositions(
	const int *nBodiesBoundaries,
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

void PrintPositions(FILE * plotfile, const int nBodies, const real_t * restrict rx, const real_t * restrict ry);
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


	const int nTimeSteps = 20000;
	//const int nTimeSteps = 100;

/*
	for (int nBodies = 10; nBodies < 100; nBodies += 10) {
		MPI_Barrier(MPI_COMM_WORLD);
		RunSimulation(nTimeSteps, nBodies);
	}
	for (int nBodies = 100; nBodies < 300; nBodies += 20) {
		MPI_Barrier(MPI_COMM_WORLD);
		RunSimulation(nTimeSteps, nBodies);
	}
	for (int nBodies = 300; nBodies < 1000; nBodies += 100) {
		MPI_Barrier(MPI_COMM_WORLD);
		RunSimulation(nTimeSteps, nBodies);
	}
	for (int nBodies = 1000; nBodies <= 1000; nBodies += 500) {
		MPI_Barrier(MPI_COMM_WORLD);
		RunSimulation(nTimeSteps, nBodies);
	}
	* */
	RunSimulation(nTimeSteps,4000);

	MPI_Finalize();
	return 0;
}


void RunSimulation(const int nTimeSteps, const int nBodies)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	// Work out how to share the bodies among the ranks.
	// For now, share evenly. Final rank does any extra that don't divide.
	int * nBodiesShare = malloc(totalRanks * sizeof (int));
	for (int r = 0; r < totalRanks-1; r++) {
		nBodiesShare[r] = nBodies/totalRanks;
	}
	nBodiesShare[totalRanks-1] = nBodies/totalRanks + nBodies%totalRanks;

	// Compute rank boundaries of bodies. Useful for communications if every rank knows these. Each rank will be looping
	// from nBodiesBoundaries[rank] up to nBodiesBoundaries[rank+1].
	int * nBodiesBoundaries = malloc((totalRanks+1) * sizeof (int));
	nBodiesBoundaries[0] = 0;
	for (int r = 1; r <= totalRanks; r++) {
		nBodiesBoundaries[r] = nBodiesBoundaries[r-1]+nBodiesShare[r-1];
	}


	// Allocate arrays. Each rank needs position, mass, accel arrays large enough to hold ALL bodies.
	// Velocity array only needs to hold the rank's share of bodies.
	// The master thread allocates enough for everything, so it can set the initial conditions and then broadcast them.
	int myBodiesAlloc = nBodiesShare[myRank];
IFMASTER myBodiesAlloc = nBodies;
	real_t * rx;
	real_t * ry;
	real_t * vx;
	real_t * vy;
	real_t * ax;
	real_t * ay;
	real_t * mass;
	rx =   _mm_malloc(nBodies * sizeof *rx,32);
	ry =   _mm_malloc(nBodies * sizeof *ry,32);
	ax =   _mm_malloc(nBodies * sizeof *ax,32);
	ay =   _mm_malloc(nBodies * sizeof *ay,32);
	mass = _mm_malloc(nBodies * sizeof *mass,32);
	vx =   _mm_malloc(myBodiesAlloc * sizeof *vx,32);
	vy =   _mm_malloc(myBodiesAlloc * sizeof *vy,32);

	// Master thread sets the initial conditions, and then sends the relevant data to ranks, ie, a broadcast of position,
	// accel and mass array, and a send of the relevant section of velocity.
IFMASTER SetInitialConditions(nBodies, rx,ry, vx,vy, ax,ay, mass);
	MPIDistributeInitialConditions(nBodies, nBodiesBoundaries, rx,ry, vx,vy, ax,ay, mass);


	double timeElapsed;
	timeElapsed = GetWallTime();
	for (int n = 0; n < nTimeSteps; n++) {
		TimeStep(n, nBodiesBoundaries, rx,ry, vx,vy, ax,ay, mass);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	timeElapsed = GetWallTime() - timeElapsed;
	// Only care about the master thread's timing.
IFMASTER printf("%4d %le %le\n", nBodies, nTimeSteps/timeElapsed/1000000.0, ErrorCheck(nBodies, rx));


	free(nBodiesShare);
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
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	MPI_Barrier(MPI_COMM_WORLD);
#if defined(AVX) || defined(SSE)
	ComputeAccelVec(nBodiesBoundaries, rx,ry, ax,ay, mass);
#else
	ComputeAccel(nBodiesBoundaries, rx,ry, ax,ay, mass);
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	UpdatePositions(nBodiesBoundaries, rx,ry, vx,vy, ax,ay);
}


void ComputeAccel(
	const int *nBodiesBoundaries,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	const int nBodies = nBodiesBoundaries[totalRanks];

	double distx, disty, sqrtRecipDist;

	for (int i = nBodiesBoundaries[myRank]; i < nBodiesBoundaries[myRank+1]; i++) {
		for (int j = i+1; j < nBodies; j++) {
			distx = rx[i] - rx[j];
			disty = ry[i] - ry[j];
			sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
			ax[i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ax[j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);

		}
	}

	// Now each rank is holding partial sums of all acceleration values. Need to do a global reduction
	MPI_Allreduce(MPI_IN_PLACE, ax, nBodies, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);
}


void ComputeAccelVec(
	const int *nBodiesBoundaries,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	double distx, disty, sqrtRecipDist;
	const int nBodies = nBodiesBoundaries[totalRanks];

	// limit of vectorized loop is the multiple of VECWIDTH <= NBODIES
	// (May have to "clean up" a few bodies at the end)
	const int jVecMax = VECWIDTH*(nBodies/VECWIDTH);

	for (int i = nBodiesBoundaries[myRank]; i < nBodiesBoundaries[myRank+1]; i++) {

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

	// Now each rank is holding partial sums of all acceleration values. Need to do a global reduction
	MPI_Allreduce(MPI_IN_PLACE, ax, nBodies, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

}

void UpdatePositions(
	const int *nBodiesBoundaries,
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


	for (int i = nBodiesBoundaries[myRank]; i < nBodiesBoundaries[myRank+1]; i++) {
			//new force values in .ax, .ay
			//update pos and vel
			vx[i-nBodiesBoundaries[myRank]] += (-G)*STEPSIZE * ax[i];
			vy[i-nBodiesBoundaries[myRank]] += (-G)*STEPSIZE * ay[i];
			rx[i] += STEPSIZE * vx[i-nBodiesBoundaries[myRank]];
			ry[i] += STEPSIZE * vy[i-nBodiesBoundaries[myRank]];
		}

	//zero rank's accel values ready for next timestep
	memset(ax, 0, nBodiesBoundaries[totalRanks] * sizeof *ax);
	memset(ax, 0, nBodiesBoundaries[totalRanks] * sizeof *ay);

	// Now we need to broadcast the updated positions to every rank.
	for (int r = 0; r < totalRanks; r++) {
		MPI_Bcast(&rx[nBodiesBoundaries[r]], nBodiesBoundaries[r+1]-nBodiesBoundaries[r], MPI_REAL_T, r, MPI_COMM_WORLD);
		MPI_Bcast(&ry[nBodiesBoundaries[r]], nBodiesBoundaries[r+1]-nBodiesBoundaries[r], MPI_REAL_T, r, MPI_COMM_WORLD);
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
	MPI_Bcast(ax, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(ay, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(mass, nBodies, MPI_REAL_T, MASTER, MPI_COMM_WORLD);

	// Master sends shares of velocity
IFMASTER {
	for (int r = 1; r < totalRanks; r++) {
		MPI_Send(&vx[nBodiesBoundaries[r]], nBodiesBoundaries[r+1]-nBodiesBoundaries[r], MPI_REAL_T, r, r, MPI_COMM_WORLD);
		MPI_Send(&vy[nBodiesBoundaries[r]], nBodiesBoundaries[r+1]-nBodiesBoundaries[r], MPI_REAL_T, r, r, MPI_COMM_WORLD);
	}
}

IFNOTMASTER {
	MPI_Recv(vx, nBodiesBoundaries[myRank+1]-nBodiesBoundaries[myRank], MPI_REAL_T, MASTER, myRank, MPI_COMM_WORLD, NULL);
	MPI_Recv(vy, nBodiesBoundaries[myRank+1]-nBodiesBoundaries[myRank], MPI_REAL_T, MASTER, myRank, MPI_COMM_WORLD, NULL);
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
