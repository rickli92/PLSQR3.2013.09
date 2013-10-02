/**
 *
 * @author He Huang
 * @file CPUSpMV.c
 * @brief Perform SpMV. This file contains sample code to optimize SpMV using OpenMP
 */
#include "CPUSpMV.h"
#include <stdlib.h>
#include <stdio.h>


extern int group_size_omp;
/**
 * A[m,n]*x[n,1]
 * @param A
 * @param x
 * @param Ax
 */
void cpuSpMV_CSR(const SpMatCSR * A, const MY_FLOAT_TYPE x[], MY_FLOAT_TYPE Ax[])
{
//JMD #ifdef GPTL
//JMD	GPTLstart("spmv_csr");
//JMD #endif
	// allocate the result vector
	int m = A->numRow;
	int rowIdx, /*colIdx,*/ firstEntryInRow, lastEntryInRow, entryIdx;
	//FOR each row in A
    for ( rowIdx = 0; rowIdx < m; rowIdx++) {
    	 firstEntryInRow = A->row_ptr[rowIdx];
    	 lastEntryInRow = A->row_ptr[rowIdx+1];
    	 MY_FLOAT_TYPE temp_Ax_rowIdx = Ax[rowIdx];
    	 //FOR each entry in current row
    	for (entryIdx = firstEntryInRow; entryIdx < lastEntryInRow; entryIdx++)    	{
    		temp_Ax_rowIdx += A->val[entryIdx] * x[A->col_idx[entryIdx]];//key part: only difference
    	}
    	Ax[rowIdx] = temp_Ax_rowIdx;
    }
//JMD #ifdef GPTL
//JMD	GPTLstop("spmv_csr");
//JMD #endif
}

/**
 * Almost the same as cpuSpMV_CSR, except using startIdxOfX for x
 * @param A
 * @param x
 * @param startIdxOfX
 * @param Ax
 */
void cpuSpMV_CSR_v2(const SpMatCSR * A, const MY_FLOAT_TYPE x[], const int startIdxOfX, MY_FLOAT_TYPE Ax[])
{
//JMD#ifdef GPTL
//JMD		GPTLstart("spmv_csr_v2");
//JMD#endif
	// allocate the result vector
	int m = A->numRow;
	int rowIdx, /*colIdx,*/ firstEntryInRow, lastEntryInRow, entryIdx;
	MY_FLOAT_TYPE temp_Ax_rowIdx;

#ifdef _OPENMP
	int chunk=(m+group_size_omp-1) / (group_size_omp);
#pragma omp parallel for private(rowIdx, temp_Ax_rowIdx, firstEntryInRow, lastEntryInRow, entryIdx) schedule(dynamic, chunk)
	for ( rowIdx = 0; rowIdx < m; rowIdx++) {
		 firstEntryInRow = A->row_ptr[rowIdx];
		 lastEntryInRow = A->row_ptr[rowIdx+1];
		 temp_Ax_rowIdx = Ax[rowIdx];
		 //FOR each entry in current row
		for (entryIdx = firstEntryInRow; entryIdx < lastEntryInRow; entryIdx++)    	{
			temp_Ax_rowIdx += A->val[entryIdx] * x[A->col_idx[entryIdx] - startIdxOfX];//key part: only difference
		}
		Ax[rowIdx] = temp_Ax_rowIdx;
	}
#else
	//FOR each row in A
    for ( rowIdx = 0; rowIdx < m; rowIdx++) {
    	 firstEntryInRow = A->row_ptr[rowIdx];
    	 lastEntryInRow = A->row_ptr[rowIdx+1];
    	 temp_Ax_rowIdx = Ax[rowIdx];
    	 //FOR each entry in current row
    	for (entryIdx = firstEntryInRow; entryIdx < lastEntryInRow; entryIdx++)    	{
    		temp_Ax_rowIdx += A->val[entryIdx] * x[A->col_idx[entryIdx] - startIdxOfX];//key part: only difference
    	}
    	Ax[rowIdx] = temp_Ax_rowIdx;
    }
#endif //OMP
//JMD #ifdef GPTL
//JMD    GPTLstop("spmv_csr_v2");
//JMD #endif
}

/**
 * transpose(A)[n,m]*y[m,1]
 * @param A
 * @param y len=m
 * @param tAy len=column
 */
void cpuSpMVTranspose_CSR(const SpMatCSR *A, const MY_FLOAT_TYPE y[], MY_FLOAT_TYPE tAy[])
{	
//JMD #ifdef GPTL
//JMD	GPTLstart("spmvT_csr");
//JMD #endif
	int m=A->numRow;
	int /*colIdx,*/ rowIdx, firstEntryInRow, lastEntryInRow, entryIndex;
	MY_FLOAT_TYPE y_temp;
	for (rowIdx = 0; rowIdx < m; rowIdx++) {
		 firstEntryInRow = A->row_ptr[rowIdx];
		 lastEntryInRow = A->row_ptr[rowIdx+1];
		 y_temp=y[rowIdx];
#pragma ivdep
		for (entryIndex = firstEntryInRow; entryIndex < lastEntryInRow; entryIndex++) {
			tAy[A->col_idx[entryIndex]] += A->val[entryIndex] * y_temp;//key part: only difference, do not need + if y is tAy is clear to 0.
		}
	}
//JMD #ifdef GPTL
//JMD	GPTLstop("spmvT_csr");
//JMD #endif
}



//OpenMP implementation
//void cpuSpMV_CSC(const SpMatCSC *A, const MY_FLOAT_TYPE x[], MY_FLOAT_TYPE Ax[])
//{
//	int n=A->numCol;
//	int colIdx, /*rowIdx,*/ firstEntryInCol, lastEntryInCol, entryIdx;
//	MY_FLOAT_TYPE x_temp;
//	int chunk=(n+group_size_omp-1) / (group_size_omp);
//	MY_FLOAT_TYPE temp;
//	int temp_idx;
//#pragma omp parallel for private(colIdx, firstEntryInCol, lastEntryInCol, x_temp, entryIdx, temp, temp_idx) schedule(dynamic, chunk)
//	for(colIdx=0; colIdx<n; colIdx++){
//		firstEntryInCol=A->col_ptr[colIdx];
//		lastEntryInCol=A->col_ptr[colIdx+1];
//		//FOR each entry in current column
//		x_temp = x[colIdx];
//		for(entryIdx=firstEntryInCol; entryIdx<lastEntryInCol; entryIdx++){
//			temp = A->val[entryIdx] * x_temp;
//			temp_idx=A->row_idx[entryIdx];
//#pragma omp atomic
//			Ax[temp_idx] += temp;
//		}
//	}
///*	for(colIdx=0; colIdx<n; colIdx++){
//		firstEntryInCol=A->col_ptr[colIdx];
//		lastEntryInCol=A->col_ptr[colIdx+1];
//		//FOR each entry in current column
//		x_temp = x[colIdx];
////		int chunk=(lastEntryInCol - firstEntryInCol + group_size_omp-1) / group_size_omp;
////#pragma omp parallel for private(entryIdx) schedule(dynamic, chunk)
//#pragma omp parallel for private(entryIdx)
//		for(entryIdx=firstEntryInCol; entryIdx<lastEntryInCol; entryIdx++){
//			Ax[A->row_idx[entryIdx]] += A->val[entryIdx] * x_temp;
//		}
//	}*/
//}
/**
 *
 * @param A
 * @param x
 * @param Ax
 */
void cpuSpMV_CSC(const SpMatCSC *A, const MY_FLOAT_TYPE x[], MY_FLOAT_TYPE Ax[])
{
//JMD #ifdef GPTL
//JMD	GPTLstart("spmv_csc");
//JMD #endif
	int n=A->numCol;
	int colIdx, /*rowIdx,*/ firstEntryInCol, lastEntryInCol, entryIdx; 
	MY_FLOAT_TYPE x_temp;
	//FOR each c(lumn in A
	for(colIdx=0; colIdx<n; colIdx++){
		firstEntryInCol=A->col_ptr[colIdx];
		lastEntryInCol=A->col_ptr[colIdx+1];
		//FOR each entry in current column
		x_temp = x[colIdx];
		for(entryIdx=firstEntryInCol; entryIdx<lastEntryInCol; entryIdx++){
			Ax[A->row_idx[entryIdx]] += A->val[entryIdx] * x_temp;
		}
	}
//JMD #ifdef GPTL
//JMD	GPTLstop("spmv_csc");
//JMD #endif
}
//test code
/*
 * 	//test CPUSpMV
	MY_FLOAT_TYPE * tAx =(MY_FLOAT_TYPE *) calloc(perProcData.block_k->numRow, sizeof(MY_FLOAT_TYPE));
	for(i=0; i<perProcData.block_k->numCol ; i++){
		perProcData.vec_x[i]=1.0;
	}
	cpuSpMV_CSC(perProcData.block_k, perProcData.vec_x, tAx);
	free(tAx);
 *

 */

/** @date 4/16/2013
 * @brief tAy=A(transpose)*y
 * @param A
 * @param y
 * @param tAy
 */
void cpuSpMVTranspose_CSC(const SpMatCSC *A, const MY_FLOAT_TYPE y[], MY_FLOAT_TYPE tAy[])
{
//JMD #ifdef GPTL
//JMD	GPTLstart("spmvT_csc");
//JMD #endif
	int n=A->numCol;
	int colIdx, /*rowIdx,*/ firstEntryInCol, lastEntryInCol, entryIdx;
	MY_FLOAT_TYPE temp_tAy_colIdx;
	//FOR each column in A
#ifdef _OPENMP
	int chunk=( n + group_size_omp-1) / (group_size_omp);

#pragma omp parallel for private(colIdx, temp_tAy_colIdx, firstEntryInCol, lastEntryInCol, entryIdx) schedule(dynamic, chunk)
	for(colIdx=0; colIdx<n; colIdx++){
//			printf("%d: colIdx=%d\n", tid, colIdx);
		firstEntryInCol=A->col_ptr[colIdx];
		lastEntryInCol=A->col_ptr[colIdx+1];
		/*temporal variables to store tAy[colIdx]. Use local cache in the following loop */
		temp_tAy_colIdx = tAy[colIdx];
		//FOR each entry in current column
		for(entryIdx=firstEntryInCol; entryIdx<lastEntryInCol; entryIdx++){
			temp_tAy_colIdx += A->val[entryIdx] * y[A->row_idx[entryIdx]];
		}
		tAy[colIdx]=temp_tAy_colIdx;
	}

#else
	for(colIdx=0; colIdx<n; colIdx++){
		firstEntryInCol=A->col_ptr[colIdx];
		lastEntryInCol=A->col_ptr[colIdx+1];
		/*temporal variables to store tAy[colIdx]. Use local cache in the following loop */
		temp_tAy_colIdx = tAy[colIdx];
		//FOR each entry in current column
		for(entryIdx=firstEntryInCol; entryIdx<lastEntryInCol; entryIdx++){
			temp_tAy_colIdx += A->val[entryIdx] * y[A->row_idx[entryIdx]];
		}
		tAy[colIdx]=temp_tAy_colIdx;
	}
#endif
//JMD #ifdef GPTL
//JMD	GPTLstop("spmvT_csc");
//JMD #endif
}
/**
 * @param A
 * @param y
 * @param startIdxOfY
 * @param tAy
 * @date 2/18/2012, 4/16/2013
 * @brief exclusively for multi-diagonal
 * Almost the same as cpuSpMVTranspose_CSC, except using startIdxOfX for y
 */
void cpuSpMVTranspose_CSC_v2(const SpMatCSC *A, const MY_FLOAT_TYPE y[], const int startIdxOfY, MY_FLOAT_TYPE tAy[] )
{
//JMD #ifdef GPTL
//JMD	GPTLstart("spmvT_csc_v2");
//JMD #endif
	int n=A->numCol;
	int colIdx, /*rowIdx,*/ firstEntryInCol, lastEntryInCol, entryIdx;
	MY_FLOAT_TYPE temp_tAy_colIdx;

#ifdef _OPENMP
	int chunk=(n+group_size_omp-1) / (group_size_omp);
#pragma omp parallel for private(colIdx, temp_tAy_colIdx, firstEntryInCol, lastEntryInCol, entryIdx) schedule(dynamic, chunk)
	for(colIdx=0; colIdx<n; colIdx++){
		firstEntryInCol=A->col_ptr[colIdx];
		lastEntryInCol=A->col_ptr[colIdx+1];
		temp_tAy_colIdx = tAy[colIdx];
		//FOR each entry in current column
		for(entryIdx=firstEntryInCol; entryIdx<lastEntryInCol; entryIdx++){
//			rowIdx=A->row_idx[entryIdx]; //global index
			temp_tAy_colIdx += A->val[entryIdx] * y[A->row_idx[entryIdx] - startIdxOfY];
		}
		tAy[colIdx] = temp_tAy_colIdx;
	}

#else
	//FOR each column in A
	for(colIdx=0; colIdx<n; colIdx++){
		firstEntryInCol=A->col_ptr[colIdx];
		lastEntryInCol=A->col_ptr[colIdx+1];
		temp_tAy_colIdx = tAy[colIdx];
		//FOR each entry in current column
		for(entryIdx=firstEntryInCol; entryIdx<lastEntryInCol; entryIdx++){
//			rowIdx=A->row_idx[entryIdx]; //global index
			temp_tAy_colIdx += A->val[entryIdx] * y[A->row_idx[entryIdx] - startIdxOfY];
		}
		tAy[colIdx] = temp_tAy_colIdx;
	}
#endif
//JMD #ifdef GPTL
//JMD	GPTLstop("spmvT_csc_v2");
//JMD #endif
}

//test code
/*
 * 	MY_FLOAT_TYPE * tAy = (MY_FLOAT_TYPE *)calloc(perProcData.block_k->numCol, sizeof(MY_FLOAT_TYPE));
	cpuSpMVTranspose_CSC(perProcData.block_k,  perProcData.vec_y_k, tAy);
	free(tAy);
 */
