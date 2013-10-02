//#define DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "main.h"
#include "CPUSpMV.h"
//in cuda3.x , using extern "C" in *.cu file: extern "C" void gpuSpMV_CSR(SpMatCSR *sparseMatrixCSR, MY_FLOAT_TYPE vectorB[], MY_FLOAT_TYPE vectorResult[], int procRank, int isFinalIteration);
#include "aprod.h"

#include "FileIO.h"
#include "vecops.h"
#include "aprod_impl.h"
#include "tridiagonal.h"



/*
 * If define GPUALL=1, then vector x and y are in GPU
 * Otherwise, x and y are in CPU
 * */
void aprod(int version, int procRank, int numProc, int mode, int m /*row*/,
		int n /*column*/, MY_FLOAT_TYPE x[]/*v len=n*/,
		MY_FLOAT_TYPE y[] /*u=perPorcVectorB len=m, assign NULL if PLSQRV3*/, void *UsrWork,
		int isFinalIteration, int iteration)
{
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("aprod");
#endif
	int i;

///////////////////////////////////////PLSQRv3///////////////////////////////////////////////////////////////////////////////////////////////

		PPDataV3_TRI * p = (PPDataV3_TRI *)  UsrWork;

		if (mode == 1) {//y <- y+Ax
#ifdef BARRIER 
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
			GPTLstart("aprod1_spmv");
#endif

			aprod_mode_1_tridiagonal(procRank, numProc, iteration, isFinalIteration, p);

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
			GPTLstop("aprod1_spmv");
#endif
		}//end of PLSQR3 mode=1


		/*PLSQR3 Aprod mode = 2*/

		else if (mode == 2){//x<-x+AT*y
#ifdef BARRIER 
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
			GPTLstart("aprod2_spmvT");
#endif

			aprod_mode_2_tridiagonal(procRank, numProc, iteration, isFinalIteration, p);

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
			GPTLstop("aprod2_spmvT");
#endif
		}//end of mode =2


#ifdef GPTL
	GPTLstop("aprod");
#endif
}
