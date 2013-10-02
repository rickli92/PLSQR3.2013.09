/*
 * vector_comm.h
 *
 *  Created on: May 14, 2012
 *      Author: hhuang1
 */

#ifndef VECTOR_COMM_H_
#define VECTOR_COMM_H_
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "main.h"
#include "CPUSpMV.h"
//in cuda3.x , using extern "C" in *.cu file: extern "C" void gpuSpMV_CSR(SpMatCSR *sparseMatrixCSR, MY_FLOAT_TYPE vectorB[], MY_FLOAT_TYPE vectorResult[], int procRank, int isFinalIteration);
#include "aprod.h"
#include "FileIO.h"



void gather_vec_x_init (int procRank, int numProc, int iteration,
		PPDataV3_TRI * p, int * check_startIdx1, int * check_endIdx1);

void gather_vec_x(int procRank, int numProc, PPDataV3_TRI * p, const int startIdx, MY_FLOAT_TYPE * vec_x_plus_neighbor);

void gather_vec_y_init (int procRank, int numProc, int iteration,
		PPDataV3_TRI * p, int * check_startIdx1, int * check_endIdx1);

void gather_vec_y (int procRank, int numProc, PPDataV3_TRI * p, const int startIdx, MY_FLOAT_TYPE * vec_y_d_plus_neighbor);

void PrintVecInfo(const int procRank, const int numProc, PPDataV3_TRI * p);

/////////////////////////////////////////////////////////////////////////////////////////


void GatherVecYInitByProjY(int procRank, int numProc,
		PPDataV3_TRI * p, int * check_startIdx1, int * check_endIdx1);

void GatherVecXInitByProjX(int procRank, int numProc,
		 PPDataV3_TRI * p, int * check_startIdx1, int * check_endIdx1);

void GatherVecXCompressed(int procRank, int numProc, PPDataV3_TRI * p, const int startIdx, MY_FLOAT_TYPE * extended_vec_x, int isFinalIteration);

void GatherVecYCompressed(int procRank, int numProc, PPDataV3_TRI * p, const int startIdx, MY_FLOAT_TYPE * extended_vec_y_d, int isFinalIteration);

//void InitSparseVecXInPPDataV3_Tri(int procRank, PPDataV3_TRI * p);

//void InitSparseVecYInPPDataV3_Tri(int rank, PPDataV3_TRI * p);

//void Init_fake_vec_y_d(PPDataV3_TRI * p, int * fake_vec_y_d);

//void Init_fake_vec_x(PPDataV3_TRI * p, int * fake_vec_x);


#endif /* VECTOR_COMM_H_ */
