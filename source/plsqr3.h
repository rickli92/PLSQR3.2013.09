/*
 * mpiio.h
 *
 *  Created on: Sep 17, 2010
 *      Author: hhuang1
 */

#ifndef MPIIO_H_
#define MPIIO_H_

#include "lsqr.h"
#include "main.h"
#include "FileIO.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/*extern int PLSQRV2, PLSQRV3;*/


#define TAG 0
#define FILENAMELEN 1024



typedef struct tagKernelColInfoPerCol{
	int colIdx;
	int nnz;
//	int globalBeginIdx;
//9/9/11
	long long globalBeginIdx;
}KernelColInfoPerCol;


int PLSQR3_main( int argc, char * argv[], int procRank, int numProc);

void initPerProcDataV3(int procRank, int numProc, PerProcDataV3 * p, SpMatCSC * block_k, SpMatCSR * block_d,
		int numRow_k, int startCol_perProc_k, int endCol_perProc_k,
		int startRowIdx_perProc_d, int numRow_perProc_d);

void finalizePerProcDataV3(PerProcDataV3 * p);

void printPerProcDataV3ToFile(PerProcDataV3 * p, int rank);


int min_vec(int * array, int len);

int max_vec(int * array, int len);

void min_vec1(const int * array, const int len, int * index, int * min);

void max_vec1(const int * array, const int len, int * index, int * max);

void init_SparseMatrixCSR_fromBasicSpMat2(SpMatCSR* matrixCSR,BasicSpMat2 * bSpMat2) ;

void basicSpMat2TOSpMatCSR(SpMatCSR* matrixCSR, BasicSpMat2* bSpMat2, int rank);

void finalizeSpMatCSR(SpMatCSR* matrixCSR);

void printSpMatCSR(int rank, SpMatCSR* matrixCSR);

void countDampingData_ascii(const char dampingFileName[], const int numRow,
		int  numNonzeroPerRow_d[], int * totalNumNonzero);

void loadDampingData_ascii(const char dampingFileName[],BasicSpMat2 * basicSpMat_d);

void mpiio_load_damp_info(int procRank, int numProc, char damp_info_filename[], const int startRowIdx_perProc_d,
		const int numRow_perProc_d, int  nnzPerRow_d_array[]);
/*void mpiio_load_damp_once(int procRank, int numProc, char damp_binary_filename[], int nnzPerProc_d,
	SpMatCSR * matCSR	);*/
void mpiio_load_damp_once(int procRank, int numProc, char damp_binary_filename[], long long nnzPerProc_d,
	void * spMat, MATRIXTYPE info);

void read_kernel_col_info(int procRank, char * kernel_col_info_filename, int numCol,
		KernelColInfoPerCol * kernelColInfo);

void mpiio_load_kernelColBin2CSC_once(int procRank, int numProc,	char * kernel_filename,
		long long nnzPerProc_k, KernelColInfoPerCol kernelColInfo_perProc[], SpMatCSC * spMatCSC);

void mpiio_load_kernelColBin2CSC_multi(int procRank, int numProc,	char * kernel_filename,
		long long nnzPerProc_k, KernelColInfoPerCol kernelColInfo_perProc[], SpMatCSC * spMatCSC);
//void reqadBin(char * filename);
void mpiio_load_kernelColBin2ReorderedCSC(int procRank, int numProc,	char * kernel_filename, int numCol,
		KernelColInfoPerCol  kernelColInfo[], int  colPerm_perProc[], SpMatCSC * spMatCSC);

void mpiio_load_kernelColBin2BasicSpMat(int procRank, int numProc,	char * kernel_filename, int numCol,
		KernelColInfoPerCol * kernelColInfo, int startCol, int endCol, BasicSpMat * basicSpMat);

void mpiio_write_result(const int procRank, const int numProc, char * fname, MY_FLOAT_TYPE * local_vec, const int local_n,
		const int local_start_idx, const int tot_len);

void initSpMatCSC(SpMatCSC *, int , int ,long long );

void finalizeSpMatCSC(SpMatCSC *);

void init_BasicSpMat(BasicSpMat *basicSparseMatrix,
		int numOfRows, int numOfColumns, int numOfNonZeroEntries);

void finalizeBasicSpMat(BasicSpMat *basicSparseMatrix);

void read_col_perm(char  filename [], int colPerm[], int numCol);

void init_BasicSpMat2(BasicSpMat2 *basicSparseMatrix,
		int numOfRows, int numOfColumns, int numOfNonZeroEntries);

void finalizeBasicSpMat2(BasicSpMat2 *basicSparseMatrix);

void readPartitionFile(const char * fname, const int rank, const int size, unsigned int * start, unsigned int * end);

#endif /* MPIIO_H_ */
