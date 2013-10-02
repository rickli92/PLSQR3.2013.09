/*
 * aprod_impl.h
 *
 *  Created on: May 14, 2012
 *      Author: hhuang1
 */

#ifndef APROD_IMPL_H_
#define APROD_IMPL_H_



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
#include "vector_comm.h"

void aprod_plsqr2(int version, int procRank, int numProc, int mode, int m /*row*/,
int n /*column*/, MY_FLOAT_TYPE x[]/*v len=n*/,
MY_FLOAT_TYPE y[] /*u=perPorcVectorB len=m, assign NULL if PLSQRV3*/, void *UsrWork,
int isFinalIteration, int iteration);


void aprod_mode_2_onediagonal(int procRank, int numProc, int iteration, int isFinalIteration,PerProcDataV3 * p);

void aprod_mode_1_onediagonal(int procRank, int numProc, int iteration, int isFinalIteration,PerProcDataV3 * p);

void aprod_mode_2_tridiagonal(int procRank, int numProc, int iteration, int isFinalIteration, PPDataV3_TRI * p);

void aprod_mode_1_tridiagonal(int procRank, int numProc, int iteration, int isFinalIteration, PPDataV3_TRI * p);


#endif /* APROD_IMPL_H_ */
