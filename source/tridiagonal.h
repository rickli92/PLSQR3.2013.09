/*
 * tridiagonal.h
 *
 *  Created on: Dec 6, 2011
 *      Author: hh
 */

#ifndef TRIDIAGONAL_H_
#define TRIDIAGONAL_H_

#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

void initPPDataV3_TRI(int procRank, int numProc, PPDataV3_TRI * p, SpMatCSC * matCSC_k,
		SpMatCSR * matCSR_d, SpMatCSC * spMatCSC_d_bycol, int numRow_k, int startCol_perProc_k, int endCol_perProc_k,
		int startRowIdx_perProc_d, int numRow_perProc_d);


void finalPPDataV3_TRI(PPDataV3_TRI * p);

//void determineVEC_POS(int rank, int size, VEC_POS * vp, int * vec_damp, int vecSize);

void projectingAColumnToX (const SpMatCSR * matCSR_d, int * projectedVectorX);

void projectingAColumnToX_v2_countLength(const SpMatCSR * matCSR_d, int * startIdx, int * len);

void projectingAColumnToX_v2 (SpMatCSR * matCSR_d, int * projectedVectorX, int  statIdx, int  len);

void projectingARowToY(const SpMatCSC * matCSC_d, int * projectingARowToY);

void projectingARowToY_v2_countLength(const SpMatCSC * matCSC_d, int * startIdx, int * len);

void projectingARowToY_v2(const SpMatCSC * matCSC_d, int * projectingARowToY, const int startIdx, const int len);

void printPPDataV3_TRIToFile(int rank, int size, const PPDataV3_TRI * p, char * name);

#endif /* TRIDIAGONAL_H_ */

